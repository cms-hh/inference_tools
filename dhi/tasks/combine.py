# coding: utf-8

"""
Base tasks dedicated to inference.
"""

import os
import sys
import re
import glob
import importlib
import functools
from collections import defaultdict

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask, CommandTask
from dhi.config import poi_data, br_hh


class CombineCommandTask(CommandTask):

    exclude_index = True

    combine_stable_options = (
        "--cminDefaultMinimizerType Minuit2"
        " --cminDefaultMinimizerStrategy 0"
        " --cminFallbackAlgo Minuit2,0:1.0"
    )


class HHModelTask(AnalysisTask):
    """
    A task that essentially adds a hh_model parameter to locate the physics model file and that
    provides a few convenience functions for working with it.
    """

    valid_hh_model_options = {"noNNLOscaling", "noBRscaling", "noHscaling", "noklDependentUnc"}

    hh_model = luigi.Parameter(
        default="HHModelPinv:model_default",
        description="the name of the HH model relative to dhi.models with optional configuration "
        "options in the format module.model_name[@opt1[@opt2...]]; valid options are {}; default: "
        "HHModelPinv.model_default".format(",".join(valid_hh_model_options)),
    )

    @classmethod
    def _split_hh_model(cls, hh_model):
        # the format used to be "module:model_name" before so adjust it to support legacy commands
        hh_model = hh_model.replace(":", ".")

        m = re.match(r"^(.+)\.([^\.@]+)(@(.+))?$", hh_model)
        if not m:
            raise Exception("invalid hh_model format '{}'".format(hh_model))

        module_id = m.group(1)
        model_name = m.group(2)
        options = []
        if m.group(4):
            options = [opt for opt in m.group(4).split("@") if opt]

        return module_id, model_name, options

    @classmethod
    def _load_hh_model(cls, hh_model):
        """
        Returns the module of the requested *hh_model* and the model instance itself in a 2-tuple.
        """
        # split into module id, model name and options
        module_id, model_name, options = cls._split_hh_model(hh_model)

        with open("/dev/null", "w") as null_stream:
            with law.util.patch_object(sys, "stdout", null_stream):
                try:
                    # first, try absolute
                    mod = importlib.import_module(module_id)
                except ImportError:
                    # then, try relative to dhi.models
                    mod = importlib.import_module("dhi.models." + module_id)

        # check if options are valid
        for opt in options:
            if opt not in cls.valid_hh_model_options:
                raise Exception("invalid HH model option '{}', valid options are {}".format(
                    opt, ",".join(cls.valid_hh_model_options)))

        # get the model and set the options
        model = getattr(mod, model_name)
        model.doNNLOscaling = "noNNLOscaling" not in options
        model.doBRscaling = "noBRscaling" not in options
        model.doHscaling = "noHscaling" not in options
        model.doklDependentUnc = "noklDependentUnc" not in options

        return mod, model

    @classmethod
    def _create_xsec_func(cls, hh_model, poi, unit, br=None):
        if poi not in ["kl", "C2V"]:
            raise ValueError("cross section conversion not supported for poi {}".format(poi))
        if unit not in ["fb", "pb"]:
            raise ValueError("cross section conversion not supported for unit {}".format(unit))
        if br and br != law.NO_STR and br not in br_hh:
            raise ValueError("unknown decay channel name {}".format(br))

        # compute the scale conversion
        scale = {"pb": 1., "fb": 1000.}[unit]
        if br in br_hh:
            scale *= br_hh[br]

        # get the proper xsec getter for the formula of the current model
        module, model = cls._load_hh_model(hh_model)
        if poi == "kl":
            # get the cross section function and make it a partial with the nnlo value set properly
            get_xsec = module.create_ggf_xsec_func(model.ggf_formula)
            get_xsec = functools.partial(get_xsec, nnlo=model.doNNLOscaling)
        else:  # C2V
            get_xsec = module.create_vbf_xsec_func(model.vbf_formula)

        # create and return the function including scaling
        return lambda *args, **kwargs: get_xsec(*args, **kwargs) * scale

    @classmethod
    def _convert_to_xsecs(cls, hh_model, expected_values, poi, unit, br=None):
        import numpy as np

        # copy values
        expected_values = np.array(expected_values)

        # create the xsec getter
        get_xsec = cls._create_xsec_func(hh_model, poi, unit, br=br)

        # convert values
        limit_keys = [key for key in expected_values.dtype.names if key.startswith("limit")]
        for point in expected_values:
            xsec = get_xsec(point[poi])
            for key in limit_keys:
                point[key] *= xsec

        return expected_values

    @classmethod
    def _get_theory_xsecs(cls, hh_model, poi_values, poi, unit=None, br=None,
            normalize=False):
        import numpy as np

        # set defaults
        if not unit:
            if not normalize:
                raise ValueError("unit must be set when normalize is False")
            unit = "fb"

        # create the xsec getter
        get_xsec = cls._create_xsec_func(hh_model, poi, unit, br=br)

        # for certain cases, also obtain errors
        has_unc = poi == "kl" and cls._load_hh_model(hh_model)[1].doNNLOscaling

        # store as records
        records = []
        dtype = [(poi, np.float32), ("xsec", np.float32)]
        if has_unc:
            dtype.extend([("xsec_p1", np.float32), ("xsec_m1", np.float32)])
        for poi_value in poi_values:
            record = (poi_value, get_xsec(poi_value))
            # add uncertainties
            if has_unc:
                record += (
                    get_xsec(poi_value, unc="up"),
                    get_xsec(poi_value, unc="down"),
                )
            # normalize by nominal value
            if normalize:
                record = (record[0],) + tuple((r / record[1]) for r in record[1:])
            records.append(record)

        return np.array(records, dtype=dtype)

    def split_hh_model(self):
        return self._split_hh_model(self.hh_model)

    def load_hh_model(self):
        return self._load_hh_model(self.hh_model)

    def create_xsec_func(self, *args, **kwargs):
        return self._create_xsec_func(self.hh_model, *args, **kwargs)

    def convert_to_xsecs(self, *args, **kwargs):
        return self._convert_to_xsecs(self.hh_model, *args, **kwargs)

    def get_theory_xsecs(self, *args, **kwargs):
        return self._get_theory_xsecs(self.hh_model, *args, **kwargs)

    def store_parts(self):
        parts = super(HHModelTask, self).store_parts()

        module_id, model_name, options = self.split_hh_model()
        part = "{}__{}".format(module_id.replace(".", "_"), model_name)
        if options:
            part += "__" + "_".join(options)
        parts["hh_model"] = part

        return parts


class MultiHHModelTask(HHModelTask):

    hh_models = law.CSVParameter(
        description="multiple names of the HH models relative to dhi.models in the format "
        "module.model_name[@opt]; default: empty",
        brace_expand=True,
    )

    hh_model = None
    split_hh_model = None
    load_hh_model = None
    create_xsec_func = None
    convert_to_xsecs = None
    get_theory_xsecs = None

    def store_parts(self):
        parts = AnalysisTask.store_parts(self)

        # replace the hh_model store part with a hash
        parts["hh_model"] = "models_" + law.util.create_hash(self.hh_models)

        return parts


class DatacardTask(HHModelTask):
    """
    A task that requires datacards in its downstream dependencies that can have quite longish names
    and are therefore not encoded in the output paths of tasks inheriting from this class. Instead,
    it defines a generic prefix that can be prepended to its outputs, and defines other parameters
    that are significant for the datacard handling.
    """

    datacards = law.CSVParameter(
        description="paths to input datacards separated by comma; supported formats are "
        "'[bin=]path', '[bin=]paths@store_directory for the last datacard in the sequence, and "
        "'@store_directory' for easier configuration; supports globbing and brace expansion",
        brace_expand=True,
    )
    mass = luigi.FloatParameter(
        default=125.0,
        description="hypothetical mass of the resonance, default: 125.",
    )
    datacards_store_dir = luigi.Parameter(
        default=law.NO_STR,
        description="do not set manually",
    )

    hash_datacards_in_repr = True

    exclude_params_index = {"datacards_store_dir"}
    exclude_params_repr = {"datacards_store_dir"}

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
            datacards, store_dir = cls.resolve_datacards(datacards)

            # update the store dir when newly set
            if params.get("datacards_store_dir", law.NO_STR) == law.NO_STR and store_dir:
                params["datacards_store_dir"] = store_dir

            # store resovled datacards
            params["datacards"] = tuple(datacards)

        return params

    def _repr_params(self, *args, **kwargs):
        params = super(DatacardTask, self)._repr_params(*args, **kwargs)

        if not params.get("datacards") and self.datacards_store_dir != law.NO_STR:
            params["datacards"] = self.datacards_store_dir

        return params

    @classmethod
    def _repr_param(cls, name, value, **kwargs):
        if name == "datacards":
            if isinstance(value, six.string_types):
                value = "@" + value
            elif cls.hash_datacards_in_repr:
                value = "hash:{}".format(law.util.create_hash(value))
            kwargs["serialize"] = False
        return super(DatacardTask, cls)._repr_param(name, value, **kwargs)

    @classmethod
    def split_datacard_path(cls, path):
        """
        Splits a datacard *path* into a maximum of three components: the path itself, leading bin
        name, and a training storage directory. Missing components are returned as *None*.
        """
        m = re.match(r"^(([^\/]+)=)?([^@]*)(@(.+))?$", path)
        path, bin_name, store_dir = None, None, None
        if m:
            path = m.group(3) or path
            bin_name = m.group(2) or bin_name
            store_dir = m.group(5) or store_dir

        return path, bin_name, store_dir

    @classmethod
    def resolve_datacards(cls, patterns):
        paths = []
        bin_names = []
        store_dir = None

        # try to resolve all patterns
        for i, pattern in enumerate(patterns):
            is_last = i == len(patterns) - 1

            # extract bin name and store dir when given
            pattern, bin_name, _store_dir = cls.split_datacard_path(pattern)

            # save the store_dir when this is the last pattern
            if is_last and _store_dir:
                store_dir = _store_dir

            # continue when when no pattern exists
            if not pattern:
                continue

            # get matching paths
            pattern = os.path.expandvars(os.path.expanduser(pattern))
            _paths = list(glob.glob(pattern))

            # when the pattern did not match anything, repeat relative to the datacards_run2 dir
            if not _paths and "DHI_DATACARDS_RUN2" in os.environ:
                dc_path = os.path.expandvars("$DHI_DATACARDS_RUN2")
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

        # complain when no paths were provided and store dir is invalid
        if not paths and not store_dir:
            raise Exception("a store_dir is required when no datacard paths are provided")

        # join to pairs and sorted by path
        pairs = sorted(list(zip(paths, bin_names)), key=lambda pair: pair[0])

        # merge bin names and paths again
        bin_name_counter = defaultdict(int)
        datacards = []
        for path, bin_name in pairs:
            if bin_name and bin_names.count(bin_name) > 1:
                bin_name_counter[bin_name] += 1
                bin_name += str(bin_name_counter[bin_name])
            datacards.append("{}={}".format(bin_name, path) if bin_name else path)

        return datacards, store_dir

    def store_parts(self):
        parts = super(DatacardTask, self).store_parts()
        if self.datacards_store_dir != law.NO_STR:
            parts["datacards"] = self.datacards_store_dir
        else:
            parts["datacards"] = "datacards_{}".format(law.util.create_hash(self.datacards))
        parts["mass"] = "m{}".format(self.mass)
        return parts

    @property
    def mass_int(self):
        return law.util.try_int(self.mass)


class MultiDatacardTask(DatacardTask):

    multi_datacards = law.MultiCSVParameter(
        description="multiple path sequnces to input datacards separated by a colon; supported "
        "formats are '[bin=]path', '[bin=]paths@store_directory for the last datacard in the "
        "sequence, and '@store_directory' for easier configuration; supports globbing and brace "
        "expansion",
        brace_expand=True,
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
        brace_expand=True,
    )

    datacards = None
    datacards_store_dir = law.NO_STR

    @classmethod
    def modify_param_values(cls, params):
        params = super(MultiDatacardTask, cls).modify_param_values(params)

        multi_datacards = params.get("multi_datacards")
        if multi_datacards:
            _multi_datacards = []
            for i, patterns in enumerate(multi_datacards):
                datacards, store_dir = cls.resolve_datacards(patterns)

                # add back the store dir for transparent forwarding to dependencies
                if store_dir:
                    if datacards:
                        datacards[-1] += "@" + store_dir
                    else:
                        datacards.append("@" + store_dir)

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
            kwargs["serialize"] = False
        return super(MultiDatacardTask, cls)._repr_param(name, value, **kwargs)

    def store_parts(self):
        parts = super(MultiDatacardTask, self).store_parts()
        parts["datacards"] = "multidatacards_{}".format(law.util.create_hash(self.multi_datacards))
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
        choices=POITask.all_pois,
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

    poi = luigi.ChoiceParameter(
        default="kl",
        choices=POITask.k_pois,
        description="name of the poi; choices: {}; default: kl".format(",".join(POITask.k_pois)),
    )
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
        choices=POITask.all_pois,
        description="name of the first poi; choices: {}; default: kl".format(
            ",".join(POITask.all_pois)),
    )
    poi2 = luigi.ChoiceParameter(
        default="kt",
        choices=POITask.all_pois,
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
