# coding: utf-8

"""
Base tasks dedicated to NLO inference.
"""


import os
import sys
import glob
import importlib

import law
import luigi

from dhi.tasks.base import AnalysisTask


class DatacardBaseTask(AnalysisTask):
    """
    A task that requires datacards in its downstream dependencies that can have quite longish names
    and are therefore not encoded in the output paths of tasks inheriting from this class. Instead,
    it defines a generic prefix that can be prepended to its outputs, and defines other parameters
    that are significant for the datacard handling.
    """

    input_cards = law.CSVParameter(
        default=tuple(os.getenv("DHI_EXAMPLE_CARDS").split(" ")),
        description="path to input datacards separated by comma; supports globbing",
    )
    dc_prefix = luigi.Parameter(
        default="", description="prefix to prepend to output file paths; " "default: ''"
    )
    hh_model = luigi.Parameter(
        default="hh:HHdefault",
        description="the name of the HH model "
        "relative to dhi.models in the format module:model_name; default: hh:HHdefault",
    )

    @classmethod
    def modify_param_values(cls, params):
        """
        Interpret globbing statements in input_cards, expand variables and remove duplicates.
        """
        cards = params.get("input_cards")
        if isinstance(cards, tuple):
            unique_cards = []
            for card in sum((glob.glob(card) for card in cards), []):
                card = os.path.expandvars(os.path.expanduser(card))
                if card not in unique_cards:
                    unique_cards.append(card)
            params["input_cards"] = tuple(unique_cards)
        return params

    def store_parts(self):
        parts = super(DatacardBaseTask, self).store_parts()
        parts["hh_model"] = self.hh_model.replace(".", "_").replace(":", "_")
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

    @staticmethod
    def get_freeze_parameters(other_pois):
        return ",".join(other_pois)

    @staticmethod
    def get_set_parameters(other_pois, values=1.0):
        if not isinstance(values, dict):
            values = {poi: values for poi in other_pois}
        return ",".join(["{}={}".format(p, values.get(p, 1.0)) for p in other_pois])

    @property
    def frozen_params(self):
        return self.get_freeze_parameters(self.other_pois)

    @property
    def fixed_params(self):
        return self.get_set_parameters(self.other_pois)


class POITask1D(POITask):

    poi = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the " "poi; default: kl",
    )
    poi_value = luigi.FloatParameter(
        default=1.0, description="initial value of the poi; default: " "1.0"
    )

    def __init__(self, *args, **kwargs):
        super(POITask1D, self).__init__(*args, **kwargs)

        # store pois that are not selected
        self.other_pois = [p for p in self.all_pois if p != self.poi]

    def store_parts(self):
        parts = super(POITask1D, self).store_parts()
        parts["poi"] = self.poi
        return parts


class POIScanTask1D(POITask1D):

    poi_range = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(-10, 10),
        min_len=2,
        max_len=2,
        description="the range of the poi given by two comma-separated values; default: -10,10",
    )
    points = luigi.IntParameter(default=21, description="number of points to scan; default: 21")

    poi_value = None


class POITask2D(POITask):

    poi1 = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of " "the first poi; default: kl",
    )
    poi2 = luigi.ChoiceParameter(
        default="kt",
        choices=POITask.k_pois,
        description="name of " "the first poi; default: kt",
    )
    poi1_value = luigi.FloatParameter(
        default=1.0, description="initial value of the first poi; " "default: 1.0"
    )
    poi2_value = luigi.FloatParameter(
        default=1.0, description="initial value of the second poi; " "default: 1.0"
    )

    def __init__(self, *args, **kwargs):
        super(POITask2D, self).__init__(*args, **kwargs)

        # poi's should differ
        if self.poi1 == self.poi2:
            raise ValueError("poi1 and poi2 should differ but both are '{}'".format(self.poi1))

        # store pois that are not selected
        self.other_pois = [p for p in self.all_pois if p not in (self.poi1, self.poi2)]

    def store_parts(self):
        parts = super(POITask2D, self).store_parts()
        parts["pois"] = "{}__{}".format(self.poi1, self.poi2)
        return parts


class POIScanTask2D(POITask2D):

    poi1_range = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(-10.0, 10.0),
        min_len=2,
        max_len=2,
        description="the range of the first poi given by two comma-separated values; "
        "default: -10,10",
    )
    poi2_range = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(-10.0, 10.0),
        min_len=2,
        max_len=2,
        description="the range of the second poi given by two comma-separated values; "
        "default: -10,10",
    )
    points1 = luigi.IntParameter(
        default=21,
        description="number of points of the first poi to " "scan; default: 21",
    )
    points2 = luigi.IntParameter(
        default=21,
        description="number of points of the second poi to " "scan; default: 21",
    )

    poi1_value = None
    poi2_value = None
