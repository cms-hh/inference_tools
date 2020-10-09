# coding: utf-8

"""
Base tasks dedicated to NLO inference.
"""

import os
import sys
import re
import glob
import importlib

import law
import luigi

from dhi.config import poi_data
from dhi.tasks.base import AnalysisTask, CommandTask


class CombineCommandTask(CommandTask):

    exclude_index = True

    combine_stable_options = (
        "--cminDefaultMinimizerType Minuit2"
        " --cminDefaultMinimizerStrategy 0"
        " --cminFallbackAlgo Minuit2,0:1.0"
    )


class DatacardBaseTask(AnalysisTask):
    """
    A task that requires datacards in its downstream dependencies that can have quite longish names
    and are therefore not encoded in the output paths of tasks inheriting from this class. Instead,
    it defines a generic prefix that can be prepended to its outputs, and defines other parameters
    that are significant for the datacard handling.
    """

    datacards = law.CSVParameter(
        default=tuple(os.getenv("DHI_EXAMPLE_CARDS").split()),
        description="path to input datacards separated by comma; supports globbing",
    )
    mass = luigi.FloatParameter(
        default=125.0,
        description="hypothetical mass of the sought resonance, default: 125.",
    )
    dc_prefix = luigi.Parameter(
        default="",
        description="prefix to prepend to output file paths; default: ''",
    )
    hh_model = luigi.Parameter(
        default="hh:HHdefault",
        description="the name of the HH model relative to dhi.models in the format "
        "module:model_name; default: hh:HHdefault",
    )

    hash_datacards_in_repr = True

    @classmethod
    def modify_param_values(cls, params):
        """
        Interpret globbing statements in datacards, expand variables, remove duplicates and sort.
        All of the transformations respect bin statements, e.g. "mybin=datacard.txt".
        """
        _cards = params.get("datacards")
        if isinstance(_cards, (tuple, list)):
            cards = []
            for pattern in _cards:
                pattern, bin_name = cls.split_datacard_path(pattern)
                for _card in glob.glob(pattern):
                    _card = os.path.abspath(os.path.expandvars(os.path.expanduser(_card)))
                    cards.append("{}={}".format(bin_name, _card) if bin_name else _card)
            cards = sorted(law.util.make_unique(cards))

            # complain when cards are empty
            if not cards:
                raise ValueError("datacards parameter did not match any existing datacard files")

            params["datacards"] = tuple(cards)
        return params

    @classmethod
    def _repr_param(cls, name, value, **kwargs):
        if cls.hash_datacards_in_repr and name == "datacards":
            value = "hash:{}".format(law.util.create_hash(value))
        return super(DatacardBaseTask, cls)._repr_param(name, value, **kwargs)

    @classmethod
    def split_datacard_path(cls, path):
        """
        Splits a potential bin name from a datacard path and returns the path and the bin name,
        which is *None* when missing.
        """
        bin_name, path = re.match(r"^(([^\/]+)=)?(.+)$", path).groups()[1:]
        return path, bin_name

    def store_parts(self):
        parts = super(DatacardBaseTask, self).store_parts()
        parts["mass"] = "m{}".format(self.mass)
        parts["hh_model"] = "model_" + self.hh_model.replace(".", "_").replace(":", "_")
        return parts

    def local_target_dc(self, *path, **kwargs):
        cls = law.LocalFileTarget if not kwargs.pop("dir", False) else law.LocalDirectoryTarget
        store = kwargs.pop("store", self.default_store)
        store = os.path.expandvars(os.path.expanduser(store))
        if path:
            # add the dc_prefix to the last path fragment
            last_parts = path[-1].rsplit(os.sep, 1)
            last_parts[-1] = self.dc_prefix + last_parts[-1]
            last_path = os.sep.join(last_parts)

            # add the last path fragment back
            path = path[:-1] + (last_path,)

        return cls(self.local_path(*path, store=store), **kwargs)

    @property
    def mass_int(self):
        return law.util.try_int(self.mass)

    def load_hh_model(self, hh_model=None):
        """
        Returns the module of the requested *hh_model* and the model instance itself in a 2-tuple.
        """
        if not hh_model:
            hh_model = self.hh_model

        full_hh_model = "dhi.models." + hh_model
        module_id, model_name = full_hh_model.split(":", 1)

        with open("/dev/null", "w") as null_stream:
            with law.util.patch_object(sys, "stdout", null_stream):
                mod = importlib.import_module(module_id)

        return mod, getattr(mod, model_name)


class POITask(DatacardBaseTask):

    k_pois = ("kl", "kt", "CV", "C2V")
    r_pois = ("r", "r_qqhh", "r_gghh")
    all_pois = k_pois + r_pois

    @classmethod
    def get_frozen_parameters(cls, other_pois):
        return ",".join(other_pois)

    @classmethod
    def get_set_parameters(cls, other_pois, values=1.0):
        if not isinstance(values, dict):
            values = {poi: values for poi in other_pois}
        return ",".join(["{}={}".format(p, values.get(p, 1.0)) for p in other_pois])

    @property
    def frozen_params(self):
        return self.get_frozen_parameters(self.other_pois)

    @property
    def fixed_params(self):
        return self.get_set_parameters(self.other_pois)


class POITask1D(POITask):

    poi = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the poi; default: kl",
    )
    poi_value = luigi.FloatParameter(
        default=1.0,
        description="initial value of the poi; default: 1.0",
    )

    def __init__(self, *args, **kwargs):
        super(POITask1D, self).__init__(*args, **kwargs)

        # store pois that are not selected
        self.other_pois = [p for p in self.all_pois if p != self.poi]

    def store_parts(self):
        parts = super(POITask1D, self).store_parts()
        parts["poi"] = self.poi
        return parts

    def get_output_postfix(self):
        return "{}_{}".format(self.poi, self.poi_value)


class POIScanTask1D(POITask1D):

    poi_range = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=None,
        min_len=2,
        max_len=2,
        description="the range of the poi given by two comma-separated values; default: range "
        "defined in physics model for poi",
    )
    poi_points = luigi.IntParameter(
        default=None,
        description="number of points to scan; default: int(poi_range[1] - poi_range[0] + 1)",
    )

    poi_value = None

    @classmethod
    def modify_param_values(cls, params):
        params = super(POIScanTask1D, cls).modify_param_values(params)

        # set default range and points
        if "poi" in params:
            data = poi_data[params["poi"]]
            if params.get("poi_range") in [None, (None,), (None, None)]:
                params["poi_range"] = data.range
            if not params.get("poi_points"):
                params["poi_points"] = int(params["poi_range"][1] - params["poi_range"][0] + 1)

        return params

    def get_output_postfix(self):
        return "{}_n{}_{}_{}".format(self.poi, self.poi_points, *self.poi_range)


class POITask2D(POITask):

    poi1 = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the first poi; default: kl",
    )
    poi2 = luigi.ChoiceParameter(
        default="kt",
        choices=POITask.k_pois,
        description="name of the first poi; default: kt",
    )
    poi1_value = luigi.FloatParameter(
        default=1.0,
        description="initial value of the first poi; default: 1.0",
    )
    poi2_value = luigi.FloatParameter(
        default=1.0,
        description="initial value of the second poi; default: 1.0",
    )

    store_pois_sorted = False

    def __init__(self, *args, **kwargs):
        super(POITask2D, self).__init__(*args, **kwargs)

        # poi's should differ
        if self.poi1 == self.poi2:
            raise ValueError("poi1 and poi2 should differ but both are '{}'".format(self.poi1))

        # store pois that are not selected
        self.other_pois = [p for p in self.all_pois if p not in (self.poi1, self.poi2)]

    def get_poi_info(self, attributes=("", "_value")):
        # get info per poi
        info = {}
        info.update({"poi1" + attr: getattr(self, "poi1" + attr) for attr in attributes})
        info.update({"poi2" + attr: getattr(self, "poi2" + attr) for attr in attributes})

        # add same info with keys starting with "poiA_" and "poiB_" to reflect sorting by name
        # i.e. when poi1 is "kl" and poi2 is "C2V", "poiA_" would denote poi2
        names = sorted([self.poi1, self.poi2])
        keys = sorted(["poi1", "poi2"], key=lambda key: names.index(getattr(self, key)))
        info.update({"poiA" + attr: getattr(self, keys[0] + attr) for attr in attributes})
        info.update({"poiB" + attr: getattr(self, keys[1] + attr) for attr in attributes})

        return info

    def store_parts(self):
        parts = super(POITask2D, self).store_parts()

        if self.store_pois_sorted:
            tmpl = "{poiA}__{poiB}"
        else:
            tmpl = "{poi1}__{poi2}"
        parts["pois"] = tmpl.format(**self.get_poi_info())

        return parts

    def get_output_postfix(self):
        if self.store_pois_sorted:
            tmpl = "{poiA}_{poiA_value}__{poiB}_{poiB_value}"
        else:
            tmpl = "{poi1}_{poi1_value}__{poi2}_{poi2_value}"
        return tmpl.format(**self.get_poi_info())


class POIScanTask2D(POITask2D):

    poi1_range = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=None,
        min_len=2,
        max_len=2,
        description="the range of the first poi given by two comma-separated values; default: "
        "range defined in physics model for poi1",
    )
    poi2_range = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=None,
        min_len=2,
        max_len=2,
        description="the range of the second poi given by two comma-separated values; default: "
        "range defined in physics model for poi2",
    )
    poi1_points = luigi.IntParameter(
        default=None,
        description="number of points of the first poi to scan; default: "
        "int(poi1_range[1] - poi1_range[0] + 1)",
    )
    poi2_points = luigi.IntParameter(
        default=None,
        description="number of points of the second poi to scan; default: "
        "int(poi2_range[1] - poi2_range[0] + 1)",
    )

    poi1_value = None
    poi2_value = None

    @classmethod
    def modify_param_values(cls, params):
        params = super(POIScanTask2D, cls).modify_param_values(params)

        # set default range and points
        for n in ["poi1", "poi2"]:
            if n not in params:
                continue
            data = poi_data[params[n]]
            if params.get(n + "_range") in [None, (None,), (None, None)]:
                params[n + "_range"] = data.range
            if not params.get(n + "_points"):
                params[n + "_points"] = int(params[n + "_range"][1] - params[n + "_range"][0] + 1)

        return params

    def get_poi_info(self, attributes=("", "_range", "_points")):
        return super(POIScanTask2D, self).get_poi_info(attributes=attributes)

    def get_output_postfix(self):
        if self.store_pois_sorted:
            tmpl = (
                "{poiA}_n{poiA_points}_{poiA_range[0]}_{poiA_range[1]}"
                "__{poiB}_n{poiB_points}_{poiB_range[0]}_{poiB_range[1]}"
            )
        else:
            tmpl = (
                "{poi1}_n{poi1_points}_{poi1_range[0]}_{poi1_range[1]}"
                "__{poi2}_n{poi2_points}_{poi2_range[0]}_{poi2_range[1]}"
            )
        return tmpl.format(**self.get_poi_info())
