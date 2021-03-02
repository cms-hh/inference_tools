# coding: utf-8

"""
Base tasks dedicated to inference.
"""

import os
import sys
import re
import glob
import importlib
import itertools
import functools
from collections import defaultdict, OrderedDict

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask, CommandTask, PlotTask
from dhi.config import poi_data, br_hh
from dhi.util import linspace, try_int
from dhi.datacard_tools import bundle_datacard
from dhi.scripts.remove_processes import remove_processes as remove_processes_script


def require_hh_model(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.hh_model_empty:
            raise Exception("calls to {}() are invalid with empty hh_model".format(func.__name__))
        return func(self, *args, **kwargs)

    return wrapper


class HHModelTask(AnalysisTask):
    """
    A task that essentially adds a hh_model parameter to locate the physics model file and that
    provides a few convenience functions for working with it.
    """

    valid_hh_model_options = {"noNNLOscaling", "noBRscaling", "noHscaling", "noklDependentUnc"}

    hh_model = luigi.Parameter(
        default="HHModelPinv.model_default",
        description="the location of the HH model to use with optional configuration options in "
        "the format module.model_name[@opt1[@opt2...]]; when no module name is given, the default "
        "one 'dhi.models.HHModelPinv' is assumed; valid options are {}; default: "
        "HHModelPinv.model_default".format(",".join(valid_hh_model_options)),
    )

    allow_empty_hh_model = False

    def __init__(self, *args, **kwargs):
        super(HHModelTask, self).__init__(*args, **kwargs)

        if self.hh_model_empty and not self.allow_empty_hh_model:
            raise Exception("hh_model is not allowed to be empty")

    @property
    def hh_model_empty(self):
        return self.hh_model in ("", law.NO_STR)

    @classmethod
    def _split_hh_model(cls, hh_model):
        # the format used to be "module:model_name" before so adjust it to support legacy commands
        hh_model = hh_model.replace(":", ".")

        # when there is no "." in the string, assume it to be the name of a model in the default
        # model file "HHModelPinv"
        if "." not in hh_model:
            hh_model = "HHModelPinv.{}".format(hh_model)

        # split into module, model name and options
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
                raise Exception(
                    "invalid HH model option '{}', valid options are {}".format(
                        opt, ",".join(cls.valid_hh_model_options)
                    )
                )

        # get the model and set the options
        model = getattr(mod, model_name)
        model.doNNLOscaling = "noNNLOscaling" not in options
        model.doBRscaling = "noBRscaling" not in options
        model.doHscaling = "noHscaling" not in options
        model.doklDependentUnc = "noklDependentUnc" not in options

        return mod, model

    @classmethod
    def _create_xsec_func(cls, hh_model, r_poi, unit, br=None, safe_signature=False):
        if r_poi not in cls.r_pois:
            raise ValueError("cross section conversion not supported for POI {}".format(r_poi))
        if unit not in ["fb", "pb"]:
            raise ValueError("cross section conversion not supported for unit {}".format(unit))
        if br and br != law.NO_STR and br not in br_hh:
            raise ValueError("unknown decay channel name {}".format(br))

        # get the module and the hh model object
        module, model = cls._load_hh_model(hh_model)

        # get the proper xsec getter, based on poi
        if r_poi == "r_gghh":
            get_ggf_xsec = module.create_ggf_xsec_func(model.ggf_formula)
            get_xsec = functools.partial(get_ggf_xsec, nnlo=model.doNNLOscaling)
            signature_kwargs = {"kl", "kt", "unc"}
        elif r_poi == "r_qqhh":
            get_xsec = module.create_vbf_xsec_func(model.vbf_formula)
            signature_kwargs = {"C2V", "CV"}
        else:  # r
            get_hh_xsec = module.create_hh_xsec_func(model.ggf_formula, model.vbf_formula)
            get_xsec = functools.partial(get_hh_xsec, nnlo=model.doNNLOscaling)
            signature_kwargs = {"kl", "kt", "C2V", "CV", "unc"}

        # compute the scale conversion
        scale = {"pb": 1.0, "fb": 1000.0}[unit]
        if br in br_hh:
            scale *= br_hh[br]

        # create a wrapper to apply the scale and to protect the signature if requested
        def wrapper(**kwargs):
            if safe_signature:
                # remove all kwargs that are not accepted
                kwargs = {k: v for k, v in kwargs.items() if k in signature_kwargs}
            return get_xsec(**kwargs) * scale

        return wrapper

    @classmethod
    def _convert_to_xsecs(
        cls, hh_model, r_poi, data, unit, br=None, param_keys=None, xsec_kwargs=None
    ):
        import numpy as np

        # create the xsec getter
        get_xsec = cls._create_xsec_func(hh_model, r_poi, unit, br=br)

        # copy values
        data = np.array(data)

        # convert values
        for point in data:
            # gather parameter values to pass to the xsec getter
            _xsec_kwargs = dict(xsec_kwargs or {})
            if param_keys:
                _xsec_kwargs.update({key: point[key] for key in param_keys})

            # get the xsec and apply it to every
            xsec = get_xsec(**_xsec_kwargs)
            for key in point.dtype.names:
                if key.startswith("limit") or key == "observed":
                    point[key] *= xsec

        return data

    @classmethod
    def _get_theory_xsecs(
        cls,
        hh_model,
        r_poi,
        param_names,
        param_values,
        unit=None,
        br=None,
        normalize=False,
        xsec_kwargs=None,
    ):
        import numpy as np

        # set defaults
        if not unit:
            if not normalize:
                raise ValueError("unit must be set when normalize is False")
            unit = "fb"

        # create the xsec getter
        get_xsec = cls._create_xsec_func(hh_model, r_poi, unit, br=br)

        # for certain cases, also obtain errors
        has_unc = r_poi in ("r", "r_gghh") and cls._load_hh_model(hh_model)[1].doNNLOscaling

        # store as records
        records = []
        dtype = [(p, np.float32) for p in param_names] + [("xsec", np.float32)]
        if has_unc:
            dtype.extend([("xsec_p1", np.float32), ("xsec_m1", np.float32)])
        for values in param_values:
            # prepare the xsec kwargs
            values = law.util.make_tuple(values)
            _xsec_kwargs = dict(xsec_kwargs or {})
            _xsec_kwargs.update(dict(zip(param_names, values)))

            # create the record, potentially with uncertainties and normalization
            record = values + (get_xsec(**_xsec_kwargs),)
            if has_unc:
                record += (
                    get_xsec(unc="up", **_xsec_kwargs),
                    get_xsec(unc="down", **_xsec_kwargs),
                )
            if normalize:
                record = (record[0],) + tuple((r / record[1]) for r in record[1:])
            records.append(record)

        return np.array(records, dtype=dtype)

    @require_hh_model
    def split_hh_model(self):
        return self._split_hh_model(self.hh_model)

    @require_hh_model
    def load_hh_model(self):
        return self._load_hh_model(self.hh_model)

    @require_hh_model
    def create_xsec_func(self, *args, **kwargs):
        return self._create_xsec_func(self.hh_model, *args, **kwargs)

    @require_hh_model
    def convert_to_xsecs(self, *args, **kwargs):
        return self._convert_to_xsecs(self.hh_model, *args, **kwargs)

    @require_hh_model
    def get_theory_xsecs(self, *args, **kwargs):
        return self._get_theory_xsecs(self.hh_model, *args, **kwargs)

    def store_parts(self):
        parts = super(HHModelTask, self).store_parts()

        if not self.hh_model_empty:
            module_id, model_name, options = self.split_hh_model()
            part = "{}__{}".format(module_id.replace(".", "_"), model_name)
            if options:
                part += "__" + "_".join(options)
            parts["hh_model"] = part

        return parts


class MultiHHModelTask(HHModelTask):

    hh_models = law.CSVParameter(
        description="comma-separated locations of HH models to use with optional configurations, "
        "each in the format module.model_name[@opt1[@opt2...]]; no default",
        brace_expand=True,
    )
    hh_model_order = law.CSVParameter(
        default=(),
        cls=luigi.IntParameter,
        significant=False,
        description="comma-separated indices of models in hh_models for reordering; not used when "
        "empty; no default",
    )
    hh_model_names = law.CSVParameter(
        default=(),
        significant=False,
        description="comma-separated names of hh models for plotting purposes; applied before "
        "reordering with hh_model_order; no default",
        brace_expand=True,
    )

    hh_model = None
    split_hh_model = None
    load_hh_model = None
    create_xsec_func = None
    convert_to_xsecs = None
    get_theory_xsecs = None

    def __init__(self, *args, **kwargs):
        super(MultiHHModelTask, self).__init__(*args, **kwargs)

        # the lengths of names and order indices must match hh_models when given
        if len(self.hh_model_names) not in (0, len(self.hh_models)):
            raise Exception(
                "when hh_model_names is set, its length ({}) must match that of "
                "hh_models ({})".format(len(self.hh_model_names), len(self.hh_models))
            )
        if len(self.hh_model_order) not in (0, len(self.hh_models)):
            raise Exception(
                "when hh_model_order is set, its length ({}) must match that of "
                "hh_models ({})".format(len(self.hh_model_order), len(self.hh_models))
            )

    def store_parts(self):
        parts = AnalysisTask.store_parts(self)

        # replace the hh_model store part with a hash
        if not self.hh_model_empty:
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
        dc_path = os.path.expandvars("$DHI_DATACARDS_RUN2")
        single_dc_matched = []

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
                _paths = list(glob.glob(os.path.join(dc_path, pattern)))
                single_dc_matched.append(len(_paths) == 1)
            else:
                single_dc_matched.append(False)

            # when directories are given, assume to find a file "datacard.txt"
            # when such a file does not exist, but a directory "latest" does, use it and try again
            __paths = []
            for path in _paths:
                if os.path.isdir(path):
                    dir_path = path
                    while True:
                        card = os.path.join(dir_path, "datacard.txt")
                        latest = os.path.join(dir_path, "latest")
                        if os.path.isfile(card):
                            path = card
                        elif os.path.isdir(latest):
                            dir_path = os.path.realpath(latest)
                            continue
                        break
                __paths.append(path)
            _paths = __paths

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

        # join to pairs and sort by path
        pairs = sorted(list(zip(paths, bin_names)), key=lambda pair: pair[0])

        # merge bin names and paths again
        bin_name_counter = defaultdict(int)
        datacards = []
        for path, bin_name in pairs:
            if bin_name and bin_names.count(bin_name) > 1:
                bin_name_counter[bin_name] += 1
                bin_name += str(bin_name_counter[bin_name])
            datacards.append("{}={}".format(bin_name, path) if bin_name else path)

        # when all matched datacards were found in the dc path and no store dir is set,
        # set it to a readable value
        if not store_dir and single_dc_matched and all(single_dc_matched):
            parts = [
                os.path.relpath(os.path.dirname(p), dc_path).replace(os.sep, "_")
                for p in sorted(paths)
            ]
            store_dir = "__".join(parts)

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
        return try_int(self.mass)


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
            raise Exception(
                "when datacard_names is set, its length ({}) must match that of "
                "multi_datacards ({})".format(len(self.datacard_names), len(self.multi_datacards))
            )
        if len(self.datacard_order) not in (0, len(self.multi_datacards)):
            raise Exception(
                "when datacard_order is set, its length ({}) must match that of "
                "multi_datacards ({})".format(len(self.datacard_order), len(self.multi_datacards))
            )

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


class ParameterValuesTask(AnalysisTask):

    parameter_values = law.CSVParameter(
        default=(),
        description="comma-separated parameter names and values to be set; the accepted format is "
        "'name1=value1,name2=value2,...'",
    )

    sort_parameter_values = True

    @classmethod
    def modify_param_values(cls, params):
        params = super(ParameterValuesTask, cls).modify_param_values(params)

        # sort parameters
        if "parameter_values" in params:
            parameter_values = params["parameter_values"]
            for p in parameter_values:
                if "=" not in p:
                    raise ValueError("invalid parameter value format '{}'".format(p))
            if cls.sort_parameter_values:
                parameter_values = sorted(parameter_values, key=lambda p: p.split("=", 1)[0])
            params["parameter_values"] = tuple(parameter_values)

        return params

    def __init__(self, *args, **kwargs):
        super(ParameterValuesTask, self).__init__(*args, **kwargs)

        # store parameter values in a dict, check for duplicates
        self.parameter_values_dict = OrderedDict()
        for p in self.parameter_values:
            name, value = p.split("=", 1)
            if name in self.parameter_values_dict:
                raise ValueError("duplicate parameter value '{}'".format(name))
            self.parameter_values_dict[name] = float(value)

    def get_output_postfix(self, join=True):
        parts = []
        if self.parameter_values:
            parts = [
                ["params"] + ["{}{}".format(*tpl) for tpl in self.parameter_values_dict.items()]
            ]

        return self.join_postfix(parts) if join else parts

    def _joined_parameter_values(self, join=True):
        values = OrderedDict(self.parameter_values_dict)
        return ",".join("{}={}".format(*tpl) for tpl in values.items()) if join else values

    @property
    def joined_parameter_values(self):
        return self._joined_parameter_values()


class ParameterScanTask(AnalysisTask):

    scan_parameters = law.MultiCSVParameter(
        default=(("kl",),),
        description="colon-separated parameters to scan, each in the format "
        "'name[,start,stop][,points]'; defaults for start and stop values are taken from the used "
        "physics model; the default number of points is inferred from that range so that there is "
        "one measurement per integer step; default: (kl,)",
    )

    force_n_scan_parameters = None
    sort_scan_parameters = True

    @classmethod
    def modify_param_values(cls, params):
        params = super(ParameterScanTask, cls).modify_param_values(params)

        # set default range and points
        if "scan_parameters" in params:
            _scan_parameters = []
            for p in params["scan_parameters"]:
                name, start, stop, points = None, None, None, None
                if len(p) == 1:
                    name = p[0]
                elif len(p) == 2:
                    name, points = p[0], int(p[1])
                elif len(p) == 3:
                    name, start, stop = p[0], float(p[1]), float(p[2])
                elif len(p) == 4:
                    name, start, stop, points = p[0], float(p[1]), float(p[2]), int(p[3])
                else:
                    raise ValueError("invalid scan parameter format '{}'".format(",".join(p)))

                # get range defaults
                if start is None or stop is None:
                    if name not in poi_data:
                        raise Exception(
                            "cannot infer default range of scan parameter {}".format(name)
                        )
                    start, stop = poi_data[name].range

                # get default points
                if points is None:
                    points = int(stop - start + 1)

                _scan_parameters.append((name, start, stop, points))

            if cls.sort_scan_parameters:
                _scan_parameters = sorted(_scan_parameters, key=lambda tpl: tpl[0])

            params["scan_parameters"] = tuple(_scan_parameters)

        return params

    def __init__(self, *args, **kwargs):
        super(ParameterScanTask, self).__init__(*args, **kwargs)

        # check the number of scan parameters if restricted
        n = self.force_n_scan_parameters
        if isinstance(n, six.integer_types):
            if self.n_scan_parameters != n:
                raise Exception(
                    "{} requires exactly {} scan parameters but got '{}'".format(
                        self.__class__.__name__, n, self.joined_scan_parameter_names
                    )
                )
        elif isinstance(n, tuple) and len(n) == 2:
            if not (n[0] <= self.n_scan_parameters <= n[1]):
                raise Exception(
                    "{} requires between {} and {} scan parameters but got '{}'".format(
                        self.__class__.__name__, n[0], n[1], self.joined_scan_parameter_names
                    )
                )

    def get_output_postfix(self, join=True):
        parts = [["scan"] + ["{}_{}_{}_n{}".format(*tpl) for tpl in self.scan_parameters]]

        return self.join_postfix(parts) if join else parts

    @property
    def n_scan_parameters(self):
        return len(self.scan_parameters)

    @property
    def scan_parameter_names(self):
        return [p for p, _, _, _ in self.scan_parameters]

    @property
    def joined_scan_parameter_names(self):
        return ",".join(self.scan_parameter_names)

    @property
    def joined_scan_values(self):
        # only valid when this is a workflow branch
        if not isinstance(self, law.BaseWorkflow) or self.is_workflow():
            return AttributeError(
                "{} has no attribute '{}' when not a workflow branch".format(
                    self.__class__.__name__, "joined_scan_values"
                )
            )

        return ",".join(
            "{}={}".format(*tpl) for tpl in zip(self.scan_parameter_names, self.branch_data)
        )

    @property
    def joined_scan_ranges(self):
        return ":".join(
            "{}={},{}".format(p, start, stop) for p, start, stop, _ in self.scan_parameters
        )

    @property
    def joined_scan_points(self):
        return ",".join(str(points) for _, _, _, points in self.scan_parameters)

    @classmethod
    def _get_scan_linspace(cls, scan_parameters, step_size=None):
        if isinstance(step_size, six.integer_types + (float,)):
            step_size = len(scan_parameters) * [step_size]

        def get_points(i, start, stop, points):
            # when step_size is set, use this value to define the resolution between points
            # otherwise, use the number of points given in scan parameters (the default)
            if not step_size:
                return points
            else:
                points = (stop - start) / float(step_size[i]) + 1
                if points % 1 != 0:
                    raise Exception(
                        "step size {} does not equally divide range [{}, {}]".format(
                            step_size[i], start, stop
                        )
                    )
                return points

        return list(
            itertools.product(
                *[
                    linspace(start, stop, get_points(i, start, stop, points))
                    for i, (_, start, stop, points) in enumerate(scan_parameters)
                ]
            )
        )

    def get_scan_linspace(self, step_size=None):
        return self._get_scan_linspace(self.scan_parameters, step_size=step_size)

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class POITask(DatacardTask, ParameterValuesTask):

    r_pois = ("r", "r_qqhh", "r_gghh")
    k_pois = ("kl", "kt", "CV", "C2V")
    all_pois = r_pois + k_pois

    pois = law.CSVParameter(
        default=("r",),
        unique=True,
        choices=all_pois,
        description="names of POIs; choices: {}; default: (r,)".format(",".join(all_pois)),
    )
    frozen_parameters = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of parameters to be frozen in addition to non-POI and "
        "scan parameters",
    )
    frozen_groups = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of groups of parameters to be frozen",
    )
    unblinded = luigi.BoolParameter(
        default=False,
        description="unblinded computation and plotting of results; default: False",
    )

    force_n_pois = None
    allow_parameter_values_in_pois = False
    sort_pois = True

    @classmethod
    def modify_param_values(cls, params):
        params = DatacardTask.modify_param_values.__func__.__get__(cls)(params)
        # no call to ParameterValuesTask.modify_param_values as its functionality is replaced below

        # sort pois
        if "pois" in params:
            if cls.sort_pois:
                params["pois"] = tuple(sorted(params["pois"]))

        # remove r and k pois from parameter values that are one, sort the rest
        if "parameter_values" in params:
            parameter_values = []
            for p in params["parameter_values"]:
                if "=" not in p:
                    raise ValueError("invalid parameter value format '{}'".format(p))
                name, value = p.split("=", 1)
                if name in cls.all_pois and float(value) == 1:
                    continue
                parameter_values.append(p)
            if cls.sort_parameter_values:
                parameter_values = sorted(parameter_values, key=lambda p: p.split("=", 1)[0])
            params["parameter_values"] = tuple(parameter_values)

        # remove r and k pois from frozen parameters as they are frozen by default, sort the rest
        if "frozen_parameters" in params:
            params["frozen_parameters"] = tuple(
                sorted(p for p in params["frozen_parameters"] if p not in cls.all_pois)
            )

        # sort frozen groups
        if "frozen_groups" in params:
            params["frozen_groups"] = tuple(sorted(params["frozen_groups"]))

        return params

    def __init__(self, *args, **kwargs):
        super(POITask, self).__init__(*args, **kwargs)

        # check the number of pois if restricted
        n = self.force_n_pois
        if isinstance(n, six.integer_types):
            if self.n_pois != n:
                raise Exception(
                    "{} requires exactly {} POIs but got '{}'".format(
                        self.__class__.__name__, n, self.joined_pois
                    )
                )
        elif isinstance(n, tuple) and len(n) == 2:
            if not (n[0] <= self.n_pois <= n[1]):
                raise Exception(
                    "{} requires between {} and {} POIs but got '{}'".format(
                        self.__class__.__name__, n[0], n[1], self.joined_pois
                    )
                )

        # check if parameter values are allowed in pois
        if not self.allow_parameter_values_in_pois:
            for p in self.parameter_values_dict:
                if p in self.pois:
                    raise Exception(
                        "parameter values are not allowed to be in POIs, but found "
                        "'{}'".format(p)
                    )

    def store_parts(self):
        parts = super(POITask, self).store_parts()
        parts["poi"] = "poi_{}".format("_".join(self.pois))
        return parts

    def get_output_postfix_pois(self):
        return self.all_pois if self.allow_parameter_values_in_pois else self.other_pois

    def get_output_postfix(self, join=True, exclude_params=None, include_params=None):
        parts = []

        # add the unblinded flag
        if self.unblinded:
            parts.append(["unblinded"])

        # add pois
        parts.append(["poi"] + list(self.pois))

        # add parameters, taking into account excluded and included ones
        params = OrderedDict((p, 1.0) for p in self.get_output_postfix_pois())
        params.update(self.parameter_values_dict)
        if exclude_params:
            params = OrderedDict((p, v) for p, v in params.items() if p not in exclude_params)
        if include_params:
            params.update((k, include_params[k]) for k in sorted(include_params.keys()))
        parts.append(["params"] + ["{}{}".format(*tpl) for tpl in params.items()])

        # add frozen paramaters
        if self.frozen_parameters:
            parts.append(["fzp"] + list(self.frozen_parameters))

        # add frozen groups
        if self.frozen_groups:
            parts.append(["fzg"] + list(self.frozen_groups))

        return self.join_postfix(parts) if join else parts

    @property
    def n_pois(self):
        return len(self.pois)

    @property
    def joined_pois(self):
        return ",".join(self.pois)

    @property
    def other_pois(self):
        return [p for p in self.all_pois if p not in self.pois]

    def _joined_parameter_values_pois(self):
        return self.all_pois if self.allow_parameter_values_in_pois else self.other_pois

    def _joined_parameter_values(self, join=True):
        # all unused pois with a value of one
        values = OrderedDict((p, 1.0) for p in self._joined_parameter_values_pois())

        # manually set parameters
        values.update(self.parameter_values_dict)

        return ",".join("{}={}".format(*tpl) for tpl in values.items()) if join else values

    @property
    def joined_parameter_values(self):
        return self._joined_parameter_values()

    def _joined_frozen_parameters(self, join=True):
        # unused pois
        params = tuple(self.other_pois)

        # manually frozen parameters
        params += tuple(self.frozen_parameters)

        return ",".join(params) if join else params

    @property
    def joined_frozen_parameters(self):
        return self._joined_frozen_parameters()

    @property
    def joined_frozen_groups(self):
        return ",".join(self.frozen_groups) or '""'

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class POIScanTask(POITask, ParameterScanTask):

    force_scan_parameters_equal_pois = False
    force_scan_parameters_unequal_pois = False
    allow_parameter_values_in_scan_parameters = False

    @classmethod
    def modify_param_values(cls, params):
        params = POITask.modify_param_values.__func__.__get__(cls)(params)
        params = ParameterScanTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def __init__(self, *args, **kwargs):
        super(POIScanTask, self).__init__(*args, **kwargs)

        # check if scan parameters exactly match pois
        if self.force_scan_parameters_equal_pois:
            missing = set(self.pois) - set(self.scan_parameter_names)
            if missing:
                raise Exception(
                    "scan parameter(s) '{}' must match POI(s) '{}' or vice versa".format(
                        self.joined_scan_parameter_names, self.joined_pois
                    )
                )
            unknown = set(self.scan_parameter_names) - set(self.pois)
            if unknown:
                raise Exception(
                    "scan parameter(s) '{}' must match POI(s) '{}' or vice versa".format(
                        self.joined_scan_parameter_names, self.joined_pois
                    )
                )

        # check if scan parameters and pois diverge
        if self.force_scan_parameters_unequal_pois:
            if set(self.pois).intersection(set(self.scan_parameter_names)):
                raise Exception(
                    "scan parameter(s) '{}' and POI(s) '{}' must not overlap".format(
                        self.joined_scan_parameter_names, self.joined_pois
                    )
                )

        # check if parameter values are in scan parameters
        if not self.allow_parameter_values_in_scan_parameters:
            for p in self.parameter_values_dict:
                if p in self.scan_parameter_names:
                    raise Exception(
                        "parameter values are not allowed to be in scan parameters, "
                        "but found '{}'".format(p)
                    )

    def get_output_postfix_pois(self):
        use_all_pois = self.allow_parameter_values_in_pois or self.force_scan_parameters_equal_pois
        return self.all_pois if use_all_pois else self.other_pois

    def get_output_postfix(self, join=True):
        if isinstance(self, law.BaseWorkflow) and self.is_branch():
            # include scan values
            scan_values = OrderedDict(zip(self.scan_parameter_names, self.branch_data))
            parts = POITask.get_output_postfix(self, join=False, include_params=scan_values)
        else:
            # exclude scan values when not explicitely allowed
            exclude_params = None
            if not self.allow_parameter_values_in_scan_parameters:
                exclude_params = self.scan_parameter_names
            parts = POITask.get_output_postfix(self, join=False, exclude_params=exclude_params)

            # insert the scan configuration
            i = 2 if self.unblinded else 1
            parts = parts[:i] + ParameterScanTask.get_output_postfix(self, join=False) + parts[i:]

        return self.join_postfix(parts) if join else parts

    def _joined_parameter_values_pois(self):
        pois = super(POIScanTask, self)._joined_parameter_values_pois()

        # skip scan parameters
        pois = [p for p in pois if p not in self.scan_parameter_names]

        return pois


class POIPlotTask(PlotTask, POITask):

    show_parameters = law.CSVParameter(
        default=(),
        significant=False,
        description="comma-separated list of parameters that are shown in the plot even if they "
        "are 1; default: empty",
    )

    def get_shown_parameters(self):
        parameter_values = self._joined_parameter_values(join=False)
        for p, v in list(parameter_values.items()):
            if v == 1 and p not in self.show_parameters:
                parameter_values.pop(p, None)
        return parameter_values


class InputDatacards(DatacardTask, law.ExternalTask):
    version = None
    hh_model = None
    mass = None

    def output(self):
        return law.TargetCollection(
            [law.LocalFileTarget(self.split_datacard_path(card)[0]) for card in self.datacards]
        )


class CombineCommandTask(CommandTask):

    toys = luigi.IntParameter(
        default=-1,
        significant=False,
        description="the number of toys to sample; -1 will use the asymptotic method; default: -1",
    )
    optimize_discretes = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when set, use additional combine flags to optimize the minimization of "
        "likelihoods with containing discrete parameters; default: False",
    )

    combine_stable_options = (
        "--cminDefaultMinimizerType Minuit2"
        " --cminDefaultMinimizerStrategy 0"
        " --cminFallbackAlgo Minuit2,0:1.0"
    )

    combine_discrete_options = (
        "--X-rt MINIMIZER_freezeDisassociatedParams"
        " --X-rtd MINIMIZER_multiMin_hideConstants"
        " --X-rtd MINIMIZER_multiMin_maskConstraints"
        " --X-rtd MINIMIZER_multiMin_maskChannels=2"
    )

    exclude_index = True

    @property
    def combine_optimization_args(self):
        args = self.combine_stable_options
        if self.optimize_discretes:
            args += " " + self.combine_discrete_options
        return args


class CombineDatacards(DatacardTask, CombineCommandTask):

    priority = 100

    def __init__(self, *args, **kwargs):
        super(DatacardTask, self).__init__(*args, **kwargs)

        # complain when no datacard paths are given but the store path does not exist yet
        if not self.datacards and not self.local_target(dir=True).exists():
            raise Exception(
                "store directory {} does not exist which is required when no datacard "
                "paths are provided".format(self.local_target(dir=True).path)
            )

    def requires(self):
        return InputDatacards.req(self)

    def output(self):
        return self.local_target("datacard.txt")

    def build_command(self, input_cards=None, output_card=None):
        if not input_cards:
            input_cards = self.datacards
        if not output_card:
            output_card = self.output().path

        return "combineCards.py {} {} > {}".format(
            self.custom_args,
            " ".join(input_cards),
            output_card,
        )

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        # immediately complain when no datacard paths were set but just a store directory
        if not self.datacards:
            raise Exception("{} task requires datacard paths to be set".format(self.task_family))

        # before running the actual card combination command, copy shape files and handle collisions
        # first, create a tmp dir to work in
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # remove any bin name from the datacard paths
        datacards = [self.split_datacard_path(card)[0] for card in self.datacards]
        bin_names = [self.split_datacard_path(card)[1] for card in self.datacards]

        # run the bundling for all cards which handles collision-free copying on its own
        datacards = [os.path.basename(bundle_datacard(card, tmp_dir.path)) for card in datacards]

        # add back bin names
        datacards = [
            ("{}={}".format(bin_name, card) if bin_name else card)
            for bin_name, card in zip(bin_names, datacards)
        ]

        # build and run the command
        output_card = tmp_dir.child("merged_XXXXXX.txt", type="f", mktemp_pattern=True)
        self.run_command(self.build_command(datacards, output_card.path), cwd=tmp_dir.path)

        # remove ggf and vbf processes that are not covered by the physics model
        mod, model = self.load_hh_model()
        all_hh_processes = {sample.label for sample in mod.ggf_samples.values()}
        all_hh_processes |= {sample.label for sample in mod.vbf_samples.values()}
        model_hh_processes = {sample.label for sample in model.ggf_formula.sample_list}
        model_hh_processes |= {sample.label for sample in model.vbf_formula.sample_list}
        to_remove = all_hh_processes - model_hh_processes
        if to_remove:
            self.logger.info(
                "trying to remove processe(s) '{}' from the combined datacard as they "
                "are not part of the phyics model {}".format(",".join(to_remove), self.hh_model)
            )
            remove_processes_script(output_card.path, map("{}*".format, to_remove))

        # copy shape files and the datacard to the output location
        output = self.output()
        output.parent.touch()
        for basename in tmp_dir.listdir(pattern="*.root", type="f"):
            tmp_dir.child(basename).copy_to(output.parent)
        output_card.copy_to(output)


class CreateWorkspace(DatacardTask, CombineCommandTask):

    priority = 90

    run_command_in_tmp = True

    def requires(self):
        return CombineDatacards.req(self)

    def output(self):
        return self.local_target("workspace.root")

    def build_command(self):
        # build physics model arguments when not empty
        model_args = ""
        if not self.hh_model_empty:
            model_args = (
                " -P {model.__module__}:{model.name}"
                " --PO doNNLOscaling={model.doNNLOscaling}"
                " --PO doBRscaling={model.doBRscaling}"
                " --PO doHscaling={model.doHscaling}"
                " --PO doklDependentUnc={model.doklDependentUnc}"
            ).format(model=self.load_hh_model()[1])

        return (
            "text2workspace.py {datacard}"
            " -o workspace.root"
            " -m {self.mass}"
            " {model_args}"
            " {self.custom_args}"
            " && "
            "mv workspace.root {workspace}"
        ).format(
            self=self,
            datacard=self.input().path,
            workspace=self.output().path,
            model_args=model_args,
        )
