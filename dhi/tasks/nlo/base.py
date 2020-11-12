# coding: utf-8

"""
Base tasks dedicated to NLO inference.
"""

import os
import sys
import re
import glob
import importlib
from collections import defaultdict

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


class DatacardTask(AnalysisTask):
    """
    A task that requires datacards in its downstream dependencies that can have quite longish names
    and are therefore not encoded in the output paths of tasks inheriting from this class. Instead,
    it defines a generic prefix that can be prepended to its outputs, and defines other parameters
    that are significant for the datacard handling.
    """

    datacards = law.CSVParameter(
        description="paths to input datacards separated by comma; supports globbing and brace "
        "expansion; no default",
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
        default="HHModelPinv:HHdefault",
        description="the name of the HH model relative to dhi.models in the format "
        "module:model_name; default: HHModelPinv:HHdefault",
    )

    hash_datacards_in_store = True
    hash_datacards_in_repr = True

    @classmethod
    def modify_param_values(cls, params):
        """
        Interpret globbing statements in datacards, expand variables, remove duplicates and sort.
        When a pattern did not resolve to valid paths it is reconsidered relative to the
        datacards_run2 directory. Bin statements, e.g. "mybin=datacard.txt" are accepted.
        """
        params = super(DatacardTask, cls).modify_param_values(params)

        datacards = params.get("datacards")
        if datacards:
            datacards = cls.resolve_datacards(datacards)

            # complain when datacards are empty
            if not datacards:
                raise ValueError("datacards parameter did not match any existing datacard files")

            params["datacards"] = tuple(datacards)

        return params

    @classmethod
    def _repr_param(cls, name, value, **kwargs):
        if cls.hash_datacards_in_repr and name == "datacards":
            value = "hash:{}".format(law.util.create_hash(value))
        return super(DatacardTask, cls)._repr_param(name, value, **kwargs)

    @classmethod
    def split_datacard_path(cls, path):
        """
        Splits a potential bin name from a datacard path and returns the path and the bin name,
        which is *None* when missing.
        """
        bin_name, path = re.match(r"^(([^\/]+)=)?(.+)$", path).groups()[1:]
        return path, bin_name

    @classmethod
    def resolve_datacards(cls, patterns):
        paths = []
        bin_names = []

        # when a pattern contains a "{", join again and perform brace expansion
        if any("{" in pattern for pattern in patterns):
            patterns = law.util.brace_expand(",".join(patterns), split_csv=True)

        # try to resolve all patterns
        for pattern in patterns:
            # extract a bin name when given
            pattern, bin_name = cls.split_datacard_path(pattern)
            pattern = os.path.expandvars(os.path.expanduser(pattern))

            # get matching paths
            _paths = list(glob.glob(pattern))

            # when the pattern did not match anything, repeat relative to the datacard submodule
            if not _paths:
                dc_path = os.path.expandvars("$DHI_BASE/datacards_run2")
                _paths = list(glob.glob(os.path.join(dc_path, pattern)))

            # when directories are given, assume to find a file "datacard.txt"
            _paths = [
                (os.path.join(path, "datacard.txt") if os.path.isdir(path) else path)
                for path in _paths
            ]

            # keep only existing cards
            _paths = filter(os.path.exists, _paths)

            # complain when no files matched
            if not _paths:
                if law.util.is_pattern(pattern):
                    raise Exception("no matching datacards found for pattern {}".format(pattern))
                else:
                    raise Exception("datacard {} does not exist".format(pattern))

            # resolve paths to make them fully deterministic as a hash might be built later on
            _paths = map(os.path.realpath, _paths)

            # add datacard path and optional bin name
            for path in _paths:
                if path not in paths:
                    paths.append(path)
                    bin_names.append(bin_name)

        # zip datacard paths again with bin names when given
        bin_name_counter = defaultdict(int)
        datacards = []
        for path, bin_name in zip(paths, bin_names):
            if bin_name and bin_names.count(bin_name) > 1:
                bin_name_counter[bin_name] += 1
                bin_name += str(bin_name_counter[bin_name])
            datacards.append("{}={}".format(bin_name, path) if bin_name else path)

        # sort
        datacards = sorted(datacards)

        return datacards

    def store_parts(self):
        parts = super(DatacardTask, self).store_parts()
        parts["hh_model"] = "model_" + self.hh_model.replace(".", "_").replace(":", "_")
        if self.hash_datacards_in_store:
            parts["datacards"] = "datacards_{}".format(law.util.create_hash(self.datacards))
        parts["mass"] = "m{}".format(self.mass)
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


class MultiDatacardTask(DatacardTask):

    multi_datacards = law.MultiCSVParameter(
        description="multiple paths to comma-separated input datacard sequences, each one "
        "separated by colon; supports globbing and brace expansion; no default",
    )
    datacard_order = law.CSVParameter(
        default=(),
        cls=luigi.IntParameter,
        significant=False,
        description="indices of datacard sequences in multi_datacards for reordering; not used "
        "when empty; default: empty",
    )
    datacard_names = law.CSVParameter(
        default=(),
        significant=False,
        description="names of datacard sequences for plotting purposes; applied before reordering "
        "with datacard_order; default: empty",
    )

    datacards = None

    @classmethod
    def modify_param_values(cls, params):
        params = super(MultiDatacardTask, cls).modify_param_values(params)

        multi_datacards = params.get("multi_datacards")
        if multi_datacards:
            _multi_datacards = []
            for i, patterns in enumerate(multi_datacards.split(":")):
                datacards = cls.resolve_datacards(patterns)

                # complain when datacards are empty
                if not datacards:
                    raise ValueError("datacards at position {} of multi_datacards parameter did "
                        "not match any existing datacard files".format(i))

                _multi_datacards.append(tuple(datacards))

            params["multi_datacards"] = tuple(_multi_datacards)

        return params

    def __init__(self, *args, **kwargs):
        super(MultiDatacardTask, self).__init__(*args, **kwargs)

        # the lengths of names and order indices must match multi_datacards when given
        if len(self.datacard_names) not in (0, len(self.multi_datacards)):
            raise Exception("when datacard_names is set, its length ({}) must match that of "
                "multi_datacards ({})".format(len(self.datacard_names), len(self.multi_datacards)))
        if len(self.datacard_order) not in (0, len(self.multi_datacards)):
            raise Exception("when datacard_order is set, its length ({}) must match that of "
                "multi_datacards ({})".format(len(self.datacard_order), len(self.multi_datacards)))

    @classmethod
    def _repr_param(cls, name, value, **kwargs):
        if cls.hash_datacards_in_repr and name == "multi_datacards":
            value = "hash:{}".format(law.util.create_hash(value))
        return super(MultiDatacardTask, cls)._repr_param(name, value, **kwargs)

    def store_parts(self):
        parts = super(MultiDatacardTask, self).store_parts()
        if self.hash_datacards_in_store:
            parts["datacards"] = "multidatacards_{}".format(
                law.util.create_hash(self.multi_datacards))
        return parts


class POITask(DatacardTask):

    k_pois = ("kl", "kt", "CV", "C2V")
    r_pois = ("r", "r_qqhh", "r_gghh")
    all_pois = k_pois + r_pois

    @classmethod
    def get_frozen_pois(cls, other_pois):
        return ",".join(other_pois)

    @classmethod
    def get_set_pois(cls, other_pois, values=1.0):
        if not isinstance(values, dict):
            values = {poi: values for poi in other_pois}
        return ",".join(["{}={}".format(p, values.get(p, 1.0)) for p in other_pois])

    @property
    def frozen_pois(self):
        return self.get_frozen_pois(self.other_pois)

    @property
    def set_pois(self):
        return self.get_set_pois(self.other_pois)


class POITask1D(POITask):

    poi = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the poi; choices: {}; default: kl".format(",".join(POITask.all_pois)),
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

    def get_poi_postfix(self):
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

    def get_poi_postfix(self):
        return "{}_n{}_{}_{}".format(self.poi, self.poi_points, *self.poi_range)


class POITask1DWithR(POITask1D):
    """
    Same as POITask1D but besides the actual poi, it keeps track of a separate r-poi that is also
    encoded in output paths. This is helpful for (e.g.) limit calculations where results are
    extracted for a certain r-poi while scanning an other parameter.
    """

    r_poi = luigi.ChoiceParameter(
        default="r",
        choices=POITask1D.r_pois,
        description="name of the r POI; choices: {}; default: r".format(
            ",".join(POITask1D.r_pois)),
    )

    def store_parts(self):
        parts = super(POITask1DWithR, self).store_parts()
        parts["poi"] = "{}__{}".format(self.r_poi, self.poi)
        return parts

    def get_poi_postfix(self):
        postfix = super(POITask1DWithR, self).get_poi_postfix()
        return "{}__{}".format(self.r_poi, postfix)


class POIScanTask1DWithR(POITask1DWithR, POIScanTask1D):
    """
    Same as POIScanTask1D but besides the actual poi, it keeps track of a separate r-poi that is
    also encoded in output paths. This is helpful for (e.g.) limit calculations where results are
    extracted for a certain r-poi while scanning an other parameter.
    """


class POITask2D(POITask):

    poi1 = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the first poi; choices: {}; default: kl".format(
            ",".join(POITask.all_pois)),
    )
    poi2 = luigi.ChoiceParameter(
        default="kt",
        choices=POITask.k_pois,
        description="name of the second poi; choices: {}; default: kt".format(
            ",".join(POITask.all_pois)),
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

    def get_poi_postfix(self):
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

    def get_poi_postfix(self):
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
