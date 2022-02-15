# coding: utf-8

"""
Base tasks dedicated to inference.
"""

import os
import sys
import re
import glob
import shlex
import importlib
import itertools
import functools
import inspect
from abc import abstractproperty
from collections import defaultdict, OrderedDict

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask, CommandTask, PlotTask, ModelParameters
from dhi.config import poi_data, br_hh
from dhi.util import linspace, try_int, real_path, expand_path, get_dcr2_path
from dhi.datacard_tools import bundle_datacard


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

    DEFAULT_HH_MODULE = "hh_model"
    DEFAULT_HH_MODEL = "model_default"

    valid_hh_model_options = {
        "noNNLOscaling", "noBRscaling", "noHscaling", "noklDependentUnc",
        "doProfilergghh", "doProfilerqqhh", "doProfilervhh",
        "doProfilekl", "doProfilekt", "doProfileCV", "doProfileC2V", "doProfileC2",
    }

    hh_model = luigi.Parameter(
        default="{}.{}".format(DEFAULT_HH_MODULE, DEFAULT_HH_MODEL),
        description="the location of the HH model to use with optional configuration options in "
        "the format [module.]model_name[@opt1[@opt2...]]; when no module is given, the default "
        "'{0}' is assumed; valid options are {2}; default: {0}.{1}".format(
            DEFAULT_HH_MODULE, DEFAULT_HH_MODEL, ",".join(valid_hh_model_options),
        ),
    )

    allow_empty_hh_model = False

    def __init__(self, *args, **kwargs):
        super(HHModelTask, self).__init__(*args, **kwargs)

        if self.hh_model_empty and not self.allow_empty_hh_model:
            raise Exception("{!r}: hh_model is not allowed to be empty".format(self))

        # cached hh model instance
        self._cached_model = None

    @property
    def hh_model_empty(self):
        return self.hh_model in ("", law.NO_STR, None)

    @classmethod
    def _split_hh_model(cls, hh_model):
        # the format used to be "module:model_name" before so adjust it to support legacy commands
        hh_model = hh_model.replace(":", ".")

        # when there is no "." in the string, assume it to be the default module
        if "." not in hh_model:
            hh_model = "{}.{}".format(cls.DEFAULT_HH_MODULE, hh_model)

        # split into module, model name and options
        front, options = hh_model.split("@", 1) if "@" in hh_model else (hh_model, "")
        module_id, model_name = front.rsplit(".", 1) if "." in front else (cls.DEFAULT_HH_MODULE, front)
        options = options.split("@") if options else []

        # check if options are valid and split them into key value pairs
        opts = OrderedDict()
        for opt in options:
            # strip potential values
            key, value = opt.split("=", 1) if "=" in opt else (opt, True)
            if key not in cls.valid_hh_model_options:
                raise Exception("invalid HH model option '{}', valid options are {}".format(
                    key, ",".join(cls.valid_hh_model_options)))
            opts[key] = value

        return module_id, model_name, opts

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

        # get the model
        model = getattr(mod, model_name)

        # helper to set options when existing
        def set_opt(name, value):
            if name in model.hh_options:
                model.set_opt(name, value)

        # set physics options
        set_opt("doNNLOscaling", not options.get("noNNLOscaling", False))
        set_opt("doBRscaling", not options.get("noBRscaling", False))
        set_opt("doHscaling", not options.get("noHscaling", False))
        set_opt("doklDependentUnc", not options.get("noklDependentUnc", False))
        set_opt("doProfilergghh", options.get("doProfilergghh"))
        set_opt("doProfilerqqhh", options.get("doProfilerqqhh"))
        set_opt("doProfilervhh", options.get("doProfilervhh"))
        set_opt("doProfilekl", options.get("doProfilekl"))
        set_opt("doProfilekt", options.get("doProfilekt"))
        set_opt("doProfileCV", options.get("doProfileCV"))
        set_opt("doProfileC2V", options.get("doProfileC2V"))
        set_opt("doProfileC2", options.get("doProfileC2"))

        # reset pois
        model.reset_pois()

        return mod, model

    @classmethod
    def _create_xsec_func(cls, hh_model, r_poi, unit, br=None, safe_signature=True):
        if r_poi not in cls.r_pois:
            raise ValueError("cross section conversion not supported for POI {}".format(r_poi))
        if unit not in ["fb", "pb"]:
            raise ValueError("cross section conversion not supported for unit {}".format(unit))
        if br and br != law.NO_STR and br not in br_hh:
            raise ValueError("unknown decay channel name {}".format(br))

        # get the module and the hh model object
        module, model = cls._load_hh_model(hh_model)

        # check that the r POI is handled by the model
        if r_poi not in model.r_pois:
            raise ValueError("r POI {} is not covered by the HH model {}".format(r_poi, hh_model))

        # get the proper xsec getter, based on poi
        if r_poi == "r_gghh":
            get_ggf_xsec = module.create_ggf_xsec_func(model.ggf_formula)
            get_xsec = functools.partial(get_ggf_xsec, nnlo=model.opt("doNNLOscaling"))
            signature_kwargs = set(inspect.getargspec(get_ggf_xsec).args) - {"nnlo"}
            has_unc = bool(model.opt("doNNLOscaling"))
        elif r_poi == "r_qqhh":
            get_xsec = module.create_vbf_xsec_func(model.vbf_formula)
            signature_kwargs = set(inspect.getargspec(get_xsec).args)
            has_unc = True
        elif r_poi == "r_vhh":
            get_xsec = module.create_vhh_xsec_func(model.vhh_formula)
            signature_kwargs = set(inspect.getargspec(get_xsec).args)
            has_unc = False
        else:  # r
            _get_xsec = model.create_hh_xsec_func()
            get_xsec = functools.partial(_get_xsec, nnlo=model.opt("doNNLOscaling"))
            signature_kwargs = set(inspect.getargspec(_get_xsec).args) - {"nnlo"}
            has_unc = True

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

        # store whether it accepts an uncertainty
        wrapper.has_unc = has_unc

        return wrapper

    @classmethod
    def _convert_to_xsecs(cls, hh_model, r_poi, data, unit, br=None, param_keys=None,
            xsec_kwargs=None):
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
    def _get_theory_xsecs(cls, hh_model, r_poi, param_names, param_values, unit=None, br=None,
            normalize=False, skip_unc=False, xsec_kwargs=None):
        import numpy as np

        # set defaults
        if not unit:
            if not normalize:
                raise ValueError("unit must be set when normalize is False")
            unit = "fb"

        # create the xsec getter
        get_xsec = cls._create_xsec_func(hh_model, r_poi, unit, br=br)

        # check if uncertainties can / should be added
        add_unc = get_xsec.has_unc and not skip_unc

        # store as records
        records = []
        dtype = [(p, np.float32) for p in param_names] + [("xsec", np.float32)]
        if add_unc:
            dtype.extend([("xsec_p1", np.float32), ("xsec_m1", np.float32)])
        for values in param_values:
            # prepare the xsec kwargs
            values = law.util.make_tuple(values)
            _xsec_kwargs = dict(xsec_kwargs or {})
            _xsec_kwargs.update(dict(zip(param_names, values)))

            # create the record, potentially with uncertainties and normalization
            record = values + (get_xsec(**_xsec_kwargs),)
            if add_unc:
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
        if self._cached_model is None:
            self._cached_model = self._load_hh_model(self.hh_model)
        return self._cached_model

    @require_hh_model
    def create_xsec_func(self, *args, **kwargs):
        return self._create_xsec_func(self.hh_model, *args, **kwargs)

    @require_hh_model
    def convert_to_xsecs(self, *args, **kwargs):
        if kwargs.get("br") and self.load_hh_model()[1].opt("doBRscaling"):
            self.logger.warning("the computation of cross sections involving the branching ratio "
                "'{}' does not consider any kappa dependence of the branching ratio itself; this "
                "is, however, part of the configured physics model '{}' and is therefore used "
                "internally by combine".format(kwargs.get("br"), self.hh_model))

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
                def fmt(k, v):
                    if isinstance(v, six.string_types):
                        return (k, v.replace("=", "").replace("/", "").replace(",", ""))
                    return (k, v)

                # sort keys for reproducible paths
                keys = sorted(options.keys())
                part += "__" + "_".join("{}{}".format(*fmt(k, options[k])) for k in keys)
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
    allow_empty_hh_model = True
    split_hh_model = None
    load_hh_model = None
    create_xsec_func = None
    convert_to_xsecs = None
    get_theory_xsecs = None

    def __init__(self, *args, **kwargs):
        super(MultiHHModelTask, self).__init__(*args, **kwargs)

        # the lengths of names and order indices must match hh_models when given
        if len(self.hh_model_names) not in (0, len(self.hh_models)):
            raise Exception("{!r}: when hh_model_names is set, its length ({}) must match that of "
                "hh_models ({})".format(self, len(self.hh_model_names), len(self.hh_models)))
        if len(self.hh_model_order) not in (0, len(self.hh_models)):
            raise Exception("{!r}: when hh_model_order is set, its length ({}) must match that of "
                "hh_models ({})".format(self, len(self.hh_model_order), len(self.hh_models)))

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
        description="paths to input datacards separated by comma or to a single workspace; "
        "datacard paths support formats '[bin=]path', '[bin=]paths@store_directory for the last "
        "datacard in the sequence, and '@store_directory' for easier configuration; supports "
        "globbing and brace expansion",
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
    workspace_inject_files = law.CSVParameter(
        default=tuple(),
        description="do not set manually",
    )

    hash_datacards_in_repr = True
    allow_workspace_input = True
    skip_inject_files = False

    exclude_params_index = {"datacards_store_dir", "workspace_inject_files"}
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
            datacards, inject, store_dir = cls.resolve_datacards(datacards)

            # update the store dir when newly set
            if params.get("datacards_store_dir", law.NO_STR) == law.NO_STR and store_dir:
                params["datacards_store_dir"] = store_dir

            # update inject files when newly set
            if params.get("workspace_inject_files", law.NO_STR) in (law.NO_STR, tuple()) and inject:
                params["workspace_inject_files"] = tuple(inject)

            # remove inject files again (or initially) when requested
            if cls.skip_inject_files:
                params["workspace_inject_files"] = tuple()

            # store resovled datacards
            params["datacards"] = tuple(datacards)

        return params

    @classmethod
    def split_datacard_path(cls, path):
        """
        Splits a datacard *path* into a maximum of four components: the path itself, leading bin
        name, injection files, and a training storage directory. Missing components are returned as
        *None*. When the *path* refers to a workspace, do not apply any splitting.
        """
        _path, bin_name, inject, store_dir = None, None, None, None

        if cls.file_is_workspace(path):
            _path = path
        else:
            m = re.match(r"^(([^\/]+)=)?([^@\<]*)(\<([^@]*))?(@(.+))?$", path)
            if m:
                _path = m.group(3) or path
                bin_name = m.group(2) or bin_name
                inject = (m.group(5) and list(filter(bool, m.group(5).split("|")))) or inject
                store_dir = m.group(7) or store_dir

        return _path, bin_name, inject, store_dir

    @classmethod
    def resolve_datacards(cls, patterns):
        paths = []
        bin_names = []
        inject = []
        store_dir = None
        dc_path = get_dcr2_path()
        has_dcr2 = dc_path is not None
        in_dc_path = has_dcr2 and dc_path == real_path(os.getcwd())
        single_dc_matched = []

        # try to resolve all patterns
        for i, pattern in enumerate(patterns):
            is_last = i == len(patterns) - 1

            # extract bin name, injection files, and a store dir when given
            pattern, bin_name, _inject, _store_dir = cls.split_datacard_path(pattern)

            # save the store_dir when this is the last pattern
            if is_last and _store_dir:
                store_dir = _store_dir

            # continue when when no pattern exists
            if not pattern:
                continue

            # get matching paths
            pattern = expand_path(pattern)
            _paths = [] if has_dcr2 and in_dc_path else list(glob.glob(pattern))
            _paths = map(os.path.realpath, _paths)

            # when the pattern did not match anything, repeat relative to the datacards_run2 dir
            if not _paths and has_dcr2:
                _paths = list(glob.glob(os.path.join(dc_path, pattern)))

            # special handling when a single, non-workspace file from datacards_run2 was matched
            _single_dc_matched = False
            if len(_paths) == 1 and has_dcr2 and _paths[0].startswith(dc_path + os.sep) \
                    and not cls.file_is_workspace(_paths[0]):
                _single_dc_matched = True

                # special case: when there is no bin_name defined, use the analysis channel name
                if not bin_name:
                    bin_name = os.path.relpath(_paths[0], dc_path).split(os.path.sep)[0]
            single_dc_matched.append(_single_dc_matched)

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

            # keep only existing cards and resolve them to get deterministic hashes later on
            _paths = filter(os.path.exists, _paths)
            _paths = map(os.path.realpath, _paths)

            # complain when no files matched
            if not _paths:
                if law.util.is_pattern(pattern):
                    raise Exception("no matching datacards found for pattern {}".format(pattern))
                else:
                    raise Exception("datacard {} does not exist".format(pattern))

            # add datacard path and optional bin name
            for path in _paths:
                if path not in paths:
                    paths.append(path)
                    bin_names.append(bin_name)

            # resolve injection files relative to the first path when not existing
            if _inject:
                for inject_file in _inject:
                    # alias
                    if inject_file.lower() == "i":
                        inject_file = "inject.json"
                    real_inject_file = real_path(inject_file)
                    if not os.path.exists(real_inject_file):
                        _real_inject_file = os.path.join(os.path.dirname(_paths[0]), inject_file)
                        if os.path.exists(_real_inject_file):
                            real_inject_file = _real_inject_file
                    inject.append(real_inject_file)

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
            parts = []
            for p in sorted(paths):
                p = os.path.relpath(p, dc_path)
                if os.path.basename(p) == "datacard.txt":
                    # when p refers to the default "datacard.txt", only use the dir name
                    p = os.path.dirname(p)
                else:
                    # remove the file extension
                    p = os.path.splitext(p)[0]
                parts.append(p)
            store_dir = "__".join(p.replace(os.sep, "_") for p in parts)

        return datacards, inject, store_dir

    @classmethod
    def file_is_workspace(cls, path):
        return real_path(path).endswith(".root")

    def __init__(self, *args, **kwargs):
        super(DatacardTask, self).__init__(*args, **kwargs)

        # complain when the input is a workspace and it's not allowed
        if not self.allow_workspace_input and self.input_is_workspace:
            raise Exception("{!r}: input {} is a workspace which is not allowed".format(
                self, self.split_datacard_path(self.datacards[0])[0]))

        # generic hook to change task parameters in a customizable way
        self.call_hook("init_datacard_task")

    @property
    def input_is_workspace(self):
        return self.datacards and len(self.datacards) == 1 \
            and self.file_is_workspace(self.split_datacard_path(self.datacards[0])[0])

    def _repr_param(self, name, value, **kwargs):
        if name == "datacards":
            if isinstance(value, six.string_types):
                value = "@" + value
            elif self.datacards_store_dir != law.NO_STR:
                value = self.datacards_store_dir
            elif self.hash_datacards_in_repr:
                value = "hash:{}".format(law.util.create_hash(value))
            kwargs["serialize"] = False
        return super(DatacardTask, self)._repr_param(name, value, **kwargs)

    def store_parts(self):
        parts = super(DatacardTask, self).store_parts()

        # set the "inputs" part depending on what is passed
        if self.datacards_store_dir != law.NO_STR:
            parts["inputs"] = self.datacards_store_dir
        else:
            prefix = "workspace" if self.input_is_workspace else "datacards"
            parts["inputs"] = "{}_{}".format(prefix, law.util.create_hash(self.datacards))

        # append inject paths
        if self.workspace_inject_files:
            parts["inputs"] += "__inject_" + law.util.create_hash(self.workspace_inject_files)

        parts["mass"] = "m{}".format(self.mass)

        return parts

    @property
    def mass_int(self):
        return try_int(self.mass)


class MultiDatacardTask(DatacardTask):

    multi_datacards = law.MultiCSVParameter(
        description="multiple path sequences to input datacards separated by a colon; supported "
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
        "when empty; no default",
    )
    datacard_names = law.CSVParameter(
        default=(),
        significant=False,
        description="names of datacard sequences for plotting purposes; applied before reordering "
        "with datacard_order; no default",
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
                datacards, inject, store_dir = cls.resolve_datacards(patterns)

                # add back inject files back to let dependencies resolve them
                if inject and datacards:
                    datacards[-1] += "<" + "<".join(inject)

                # add back the store dir back to let dependencies resolve it
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
        n = self.n_datacard_entries
        if self.datacard_names and len(self.datacard_names) != n:
            raise Exception("found {} entries in datacard_names whereas {} is expected".format(
                len(self.datacard_names), n))
        if self.datacard_order and len(self.datacard_order) != n:
            raise Exception("found {} entries in datacard_order whereas {} is expected".format(
                len(self.datacard_order), n))

    @property
    def n_datacard_entries(self):
        return len(self.multi_datacards)

    def _repr_param(self, name, value, **kwargs):
        if self.hash_datacards_in_repr and name == "multi_datacards":
            value = "hash:{}".format(law.util.create_hash(value))
            kwargs["serialize"] = False
        return super(MultiDatacardTask, self)._repr_param(name, value, **kwargs)

    def store_parts(self):
        parts = super(MultiDatacardTask, self).store_parts()
        parts["inputs"] = "multidatacards_{}".format(law.util.create_hash(self.multi_datacards))
        return parts


class ParameterValuesTask(AnalysisTask):

    parameter_values = ModelParameters(
        default=(),
        min_len=1,
        max_len=1,
        description="colon-separated parameter names and values to be set; the accepted format is "
        "'name1=value1:name2=value2:...'",
    )
    parameter_ranges = ModelParameters(
        default=(),
        min_len=2,
        max_len=2,
        significant=False,
        description="colon-separated parameter names and ranges to be set; the accepted format is "
        "'name1=min1,max1:name2=min2,max2:...'",
    )

    sort_parameter_values = True
    sort_parameter_ranges = True

    @classmethod
    def modify_param_values(cls, params):
        params = super(ParameterValuesTask, cls).modify_param_values(params)

        # sort parameters values
        if "parameter_values" in params:
            parameter_values = params["parameter_values"]
            if cls.sort_parameter_values:
                parameter_values = sorted(parameter_values, key=lambda p: p[0])
            params["parameter_values"] = tuple(parameter_values)

        # sort parameters ranges
        if "parameter_ranges" in params:
            parameter_ranges = params["parameter_ranges"]
            if cls.sort_parameter_ranges:
                parameter_ranges = sorted(parameter_ranges, key=lambda p: p[0])
            params["parameter_ranges"] = tuple(parameter_ranges)

        return params

    def __init__(self, *args, **kwargs):
        super(ParameterValuesTask, self).__init__(*args, **kwargs)

        # store parameter values in a dict, check for duplicates
        self.parameter_values_dict = OrderedDict()
        for p in self.parameter_values:
            if p[0] in self.parameter_values_dict:
                raise ValueError("{!r}: duplicate parameter value '{}'".format(self, p[0]))
            self.parameter_values_dict[p[0]] = float(p[1])

        # store parameter ranges in a dict, check for duplicates
        self.parameter_ranges_dict = OrderedDict()
        for p in self.parameter_ranges:
            if p[0] in self.parameter_ranges_dict:
                raise ValueError("{!r}: duplicate parameter range '{}'".format(self, p[0]))
            self.parameter_ranges_dict[p[0]] = (float(p[1]), float(p[2]))

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
        return self._joined_parameter_values() or '""'

    def _joined_parameter_ranges(self, join=True):
        values = OrderedDict(self.parameter_ranges_dict)
        return ":".join("{}={},{}".format(k, *v) for k, v in values.items()) if join else values

    @property
    def joined_parameter_ranges(self):
        return self._joined_parameter_ranges() or '""'


class ParameterScanTask(AnalysisTask):

    scan_parameters = ModelParameters(
        default=(("kl",),),
        max_len=3,
        description="colon-separated parameters to scan, each in the format "
        "'name[,start,stop[,points]]'; defaults for start and stop values are taken from the used "
        "physics model; the default number of points is inferred from that range so that there is "
        "one measurement per integer step; when the same scan parameter name is used multiple "
        "times, the corresponding scan ranges are joined and the order of first occurrence is "
        "preserved; default: (kl,)",
    )

    force_n_scan_parameters = None
    sort_scan_parameters = True
    allow_multiple_scan_ranges = False
    find_scan_parameter_patches = True

    @classmethod
    def modify_param_values(cls, params):
        params = super(ParameterScanTask, cls).modify_param_values(params)

        # set default range and points
        if "scan_parameters" in params:
            _scan_parameters = []
            names = []
            for p in params["scan_parameters"]:
                name, start, stop, points = None, None, None, None
                if len(p) == 1:
                    name = p[0]
                elif len(p) == 3:
                    name, start, stop = p[0], float(p[1]), float(p[2])
                elif len(p) == 4:
                    name, start, stop, points = p[0], float(p[1]), float(p[2]), int(p[3])
                else:
                    raise ValueError("invalid scan parameter format '{}'".format(",".join(p)))

                # check if the parameter name was already used
                if name in names and not cls.allow_multiple_scan_ranges:
                    raise Exception("scan parameter {} configured more than once".format(name))

                # get range defaults
                if start is None or stop is None:
                    if name not in poi_data:
                        raise Exception("cannot infer default range of scan parameter {}".format(
                            name))
                    start, stop = poi_data[name].range

                # get default points
                if points is None:
                    points = int(stop - start + 1)

                _scan_parameters.append((name, start, stop, points))
                names.append(name)

            if cls.sort_scan_parameters:
                _scan_parameters = sorted(_scan_parameters, key=lambda tpl: tpl[0])

            params["scan_parameters"] = tuple(_scan_parameters)

        return params

    def __init__(self, *args, **kwargs):
        super(ParameterScanTask, self).__init__(*args, **kwargs)

        # store scan parameters and ranges as a dict
        self.scan_parameters_dict = OrderedDict()
        for p in self.scan_parameters:
            if len(p) < 4:
                raise Exception("invalid scan parameter size '{}'".format(p))
            self.scan_parameters_dict.setdefault(p[0], []).append(p[1:])

        # check the number of scan parameters if restricted
        n = self.force_n_scan_parameters
        if isinstance(n, six.integer_types):
            if self.n_scan_parameters != n:
                raise Exception("exactly {} scan parameters required but got '{}'".format(
                    n, self.joined_scan_parameter_names))
        elif isinstance(n, tuple) and len(n) == 2:
            if not (n[0] <= self.n_scan_parameters <= n[1]):
                raise Exception("between {} and {} scan parameters required but got "
                    "'{}'".format(n[0], n[1], self.joined_scan_parameter_names))

    def has_multiple_scan_ranges(self):
        return any(len(ranges) > 1 for ranges in self.scan_parameters_dict.values())

    def _assert_single_scan_range(self, attr):
        for name, ranges in self.scan_parameters_dict.items():
            if len(ranges) > 1:
                raise Exception("{} is only available for scan parameters with one range, "
                    "but {} has {} ranges".format(attr, name, len(ranges)))

    def get_output_postfix(self, join=True):
        parts = [["scan"] + [
            "_".join([name] + ["{}_{}_n{}".format(*r) for r in ranges])
            for name, ranges in self.scan_parameters_dict.items()
        ]]

        return self.join_postfix(parts) if join else parts

    @property
    def n_scan_parameters(self):
        return len(self.scan_parameters_dict)

    @property
    def scan_parameter_names(self):
        return list(self.scan_parameters_dict)

    @property
    def joined_scan_parameter_names(self):
        return ",".join(self.scan_parameter_names)

    @property
    def joined_scan_values(self):
        # only valid when this is a workflow branch
        if not isinstance(self, law.BaseWorkflow) or self.is_workflow():
            raise AttributeError("{!r}: has no attribute '{}' when not a workflow branch".format(
                self, "joined_scan_values"))

        return ",".join(
            "{}={}".format(*tpl) for tpl in zip(self.scan_parameter_names, self.branch_data)
        )

    def _joined_scan_ranges(self, join=True):
        self._assert_single_scan_range("joined_scan_ranges")
        vals = OrderedDict((name, r[0][:2]) for name, r in self.scan_parameters_dict.items())
        return ":".join("{}={},{}".format(name, *r) for name, r in vals.items()) if join else vals

    @property
    def joined_scan_ranges(self):
        return self._joined_scan_ranges()

    @property
    def joined_scan_points(self):
        self._assert_single_scan_range("joined_scan_points")
        return ",".join(str(ranges[0][2]) for ranges in self.scan_parameters_dict.values())

    @classmethod
    def _get_scan_linspace(cls, ranges, step_sizes=None):
        if not isinstance(step_sizes, (list, tuple)):
            step_sizes = len(ranges) * [step_sizes]
        elif len(step_sizes) != len(ranges):
            raise Exception("number of step_sizes {} must match number of ranges {}".format(
                len(step_sizes), len(ranges)))

        def get_points(r, step_size):
            # when step_size is set, use this value to define the resolution between points
            # otherwise, use the number of points given by the range
            start, stop = r[:2]
            if step_size:
                points = (stop - start) / float(step_size) + 1
                if points % 1 != 0:
                    raise Exception("step size {} does not equally divide range [{}, {}]".format(
                        step_size, start, stop))
                return points
            elif len(r) < 3:
                return int(stop - start + 1)
            else:
                return r[2]

        return list(itertools.product(*[
            linspace(r[0], r[1], get_points(r, step_size))
            for r, step_size in zip(ranges, step_sizes)
        ]))

    def get_scan_linspace(self, step_sizes=None):
        self._assert_single_scan_range("get_scan_linspace")
        return self._get_scan_linspace([ranges[0] for ranges in self.scan_parameters_dict.values()],
            step_sizes=step_sizes)

    def get_scan_parameter_combinations(self, find_patches=None):
        if find_patches is None:
            find_patches = self.find_scan_parameter_patches

        n_params = len(self.scan_parameter_names)
        if len(self.scan_parameters) > n_params > 1 and find_patches:
            # patches are identified when parameters of scan ranges are repeated multiple times
            # in the exact same order, e.g. for two parameters a and b:
            # (a, b, b) or (a, b, a) or (a, b, b, a)
            #   -> no patch, fallback to the full product between all a's and b's
            # (a, b, a, b) or (a, b, a, b, a, b)
            #   -> two or three (a, b) patches found
            names = tuple(p[0] for p in self.scan_parameters)
            n_patches = len(self.scan_parameters) / float(n_params)
            is_int = lambda n: int(n) == n
            if is_int(n_patches) and names == int(n_patches) * tuple(self.scan_parameter_names):
                self.logger.debug("identified {} patches in scan parameter ranges".format(
                    int(n_patches)))
                return [
                    self.scan_parameters[i * n_params:(i + 1) * n_params]
                    for i in range(int(n_patches))
                ]

        # build the product of all scan ranges
        return [
            tuple((name,) + _values for name, _values in zip(self.scan_parameter_names, values))
            for values in itertools.product(*self.scan_parameters_dict.values())
        ]

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class POITask(DatacardTask, ParameterValuesTask):

    # class-level sequence of all available pois
    # instances will have potentially reduced sequences, depending on the physics model
    r_pois = ("r", "r_gghh", "r_qqhh", "r_vhh")
    k_pois = ("kl", "kt", "CV", "C2V", "C2")
    all_pois = r_pois + k_pois

    pois = law.CSVParameter(
        default=("r",),
        unique=True,
        choices=all_pois,
        description="names of POIs; choices: {}; default: (r,)".format(",".join(all_pois)),
    )
    unblinded = luigi.BoolParameter(
        default=False,
        description="unblinded computation and plotting of results; default: False",
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

    force_n_pois = None
    allow_parameter_values_in_pois = False
    freeze_pois_with_parameter_values = False
    sort_pois = True

    @classmethod
    def modify_param_values(cls, params):
        params = DatacardTask.modify_param_values.__func__.__get__(cls)(params)
        # no call to ParameterValuesTask.modify_param_values as its functionality is replaced below

        # sort pois
        if "pois" in params:
            if cls.sort_pois:
                params["pois"] = tuple(sorted(params["pois"]))

        # remove r and k pois with SM values from parameters, sort the rest
        if "parameter_values" in params:
            parameter_values = []
            for p in params["parameter_values"]:
                name, value = p
                if name in cls.all_pois and float(value) == poi_data[name].sm_value:
                    continue
                parameter_values.append(p)
            if cls.sort_parameter_values:
                parameter_values = sorted(parameter_values, key=lambda p: p[0])
            params["parameter_values"] = tuple(parameter_values)

        # remove r and k pois from frozen parameters as they are frozen by default, sort the rest
        if "frozen_parameters" in params:
            multi = isinstance(getattr(cls, "frozen_parameters"), law.MultiCSVParameter)
            if not multi:
                params["frozen_parameters"] = (params["frozen_parameters"],)
            params["frozen_parameters"] = tuple(
                tuple(sorted(p for p in fp if p not in cls.all_pois))
                for fp in params["frozen_parameters"]
            )
            if not multi:
                params["frozen_parameters"] = params["frozen_parameters"][0]

        # sort frozen groups
        if "frozen_groups" in params:
            multi = isinstance(getattr(cls, "frozen_groups"), law.MultiCSVParameter)
            if not multi:
                params["frozen_groups"] = (params["frozen_groups"],)
            params["frozen_groups"] = tuple(tuple(sorted(fg)) for fg in params["frozen_groups"])
            if not multi:
                params["frozen_groups"] = params["frozen_groups"][0]

        return params

    def __init__(self, *args, **kwargs):
        super(POITask, self).__init__(*args, **kwargs)

        # store available pois on task level and potentially update them according to the model
        if self.hh_model_empty:
            self.r_pois, self.k_pois = self.get_empty_hh_model_pois()
        else:
            model = self.load_hh_model()[1]
            self.r_pois = tuple(model.r_pois)
            self.k_pois = tuple(model.k_pois)
        self.all_pois = self.r_pois + self.k_pois

        # check again of the chosen pois are available
        for p in self.pois:
            if p not in self.all_pois:
                raise Exception("{!r}: parameter {} is not a POI in model {}".format(
                    self, p, self.hh_model))

        # check the number of pois if restricted
        n = self.force_n_pois
        if isinstance(n, six.integer_types):
            if self.n_pois != n:
                raise Exception("{!r}: exactly {} POIs required but got '{}'".format(
                    self, n, self.joined_pois))
        elif isinstance(n, tuple) and len(n) == 2:
            if not (n[0] <= self.n_pois <= n[1]):
                raise Exception("{!r}: between {} and {} POIs required but got '{}'".format(
                    self, n[0], n[1], self.joined_pois))

        # check if parameter values are allowed in pois
        if not self.allow_parameter_values_in_pois:
            for p in self.parameter_values_dict:
                if p in self.pois:
                    raise Exception("{!r}: parameter values are not allowed to be in POIs, but "
                        "found '{}'".format(self, p))

        # check the type of the unblinded parameter (for downstream extensibility)
        if self.unblinded is not None and not isinstance(self.unblinded, (bool, tuple)):
            raise TypeError("{!r}: unblinded must refer to a bool or tuple, but found '{}'".format(
                self, self.unblinded))

    def get_empty_hh_model_pois(self):
        # hook that can be implemented to configure the r (and possibly k) POIs to be used
        # when not hh model is configured
        return tuple(self.__class__.r_pois), tuple(self.__class__.k_pois)

    def store_parts(self):
        parts = super(POITask, self).store_parts()
        parts["poi"] = "poi_{}".format("_".join(self.pois))
        return parts

    def get_output_postfix_pois(self):
        return self.all_pois if self.allow_parameter_values_in_pois else self.other_pois

    def get_output_postfix(self, join=True, exclude_params=None, include_params=None):
        parts = []

        # add the unblinded flag, or a hash in case of multiple values
        if isinstance(self.unblinded, bool):
            if self.unblinded:
                parts.append(["unblinded"])
        elif self.unblinded:
            if all(self.unblinded):
                parts.append(["unblinded"])
            elif any(self.unblinded):
                parts.append(["unblinded_{}".format("".join(map(str, map(int, self.unblinded))))])

        # add pois
        parts.append(["poi"] + list(self.pois))

        # add parameters, taking into account excluded and included ones
        params = OrderedDict((p, poi_data[p].sm_value) for p in self.get_output_postfix_pois())
        params.update(self.parameter_values_dict)
        if exclude_params:
            params = OrderedDict((p, v) for p, v in params.items() if p not in exclude_params)
        if include_params:
            params.update((k, include_params[k]) for k in sorted(include_params.keys()))
        parts.append(["params"] + ["{}{}".format(*tpl) for tpl in params.items()])

        # add frozen paramaters
        if self.frozen_parameters:
            # multi-safe
            multi = isinstance(getattr(self.__class__, "frozen_parameters"), law.MultiCSVParameter)
            fp = self.frozen_parameters if multi else (self.frozen_parameters,)
            for _fp in fp:
                # just join the first five and continue with an optional hash
                fp_str = "_".join(_fp[:5])
                if len(_fp) > 5:
                    fp_str += "_" + law.util.create_hash(_fp)
                parts.append(["fzp", fp_str])

        # add frozen groups
        if self.frozen_groups:
            # multi-safe
            multi = isinstance(getattr(self.__class__, "frozen_groups"), law.MultiCSVParameter)
            fg = self.frozen_groups if multi else (self.frozen_groups,)
            for _fg in fg:
                # just join the first five and continue with an optional hash
                fg_str = "_".join(_fg[:5])
                if len(_fg) > 5:
                    fg_str += "_" + law.util.create_hash(_fg)
                parts.append(["fzg", fg_str])

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
        values = OrderedDict((p, poi_data[p].sm_value) for p in self._joined_parameter_values_pois())
        # manually set parameters
        values.update(self.parameter_values_dict)

        return ",".join("{}={}".format(*tpl) for tpl in values.items()) if join else values

    @property
    def joined_parameter_values(self):
        return self._joined_parameter_values()

    def _joined_frozen_parameters(self, join=True):
        params = tuple()

        # optionally add pois whose value is set explicitly in parameter_values
        if self.freeze_pois_with_parameter_values:
            params += tuple(p for p in self.pois if p in self.parameter_values_dict)

        # unused pois
        params += tuple(self.other_pois)

        # manually frozen parameters
        params += tuple(self.frozen_parameters)

        return ",".join(params) if join else params

    @property
    def joined_frozen_parameters(self):
        return self._joined_frozen_parameters()

    @property
    def joined_frozen_groups(self):
        return ",".join(self.frozen_groups) or '""'

    @property
    def blinded(self):
        if self.unblinded is None:
            raise Exception("cannot infer attribute 'blinded' when 'unblinded' is None")

        # trivial case
        if isinstance(self.unblinded, bool):
            return not self.unblinded

        # at this point, it must be tuple
        if not self.unblinded:
            raise Exception("cannot convert empty tuple 'unblinded' to property 'blinded'")

        # flip flags
        return tuple(map((lambda b: not b), self.unblinded))

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class POIMultiTask(POITask):

    unblinded = law.CSVParameter(
        cls=luigi.BoolParameter,
        default=(False,),
        min_len=1,
        description="comma-separated list of booleans defining which set of results should be "
        "unblinded; the length should be one or match the number of datacard sequences or models; "
        "default: (False,)",
    )
    frozen_parameters = law.MultiCSVParameter(
        description="multiple comma-separated sequences of names of parameters to be frozen in "
        "addition to non-POI parameters; sequences should be colon-separated and the number of "
        "sequences should be zero, one or match the number of datacard sequences or models; "
        "no default",
        default=(),
    )
    frozen_groups = law.MultiCSVParameter(
        description="multiple comma-separated sequences of names of parameter groups to be frozen "
        "in addition to non-POI parameters; sequences should be colon-separated and the number of "
        "sequences should be zero, one or match the number of datacard sequences or models; "
        "no default",
        default=(),
    )

    def __init__(self, *args, **kwargs):
        super(POIMultiTask, self).__init__(*args, **kwargs)

        # check unblinded
        n = len(getattr(self, self.compare_multi_sequence))
        if self.unblinded is not None and len(self.unblinded) not in (1, n):
            raise Exception("{!r}: the number of --unblinded values ({}) must be one or match "
                "that of {} ({})".format(self, len(self.unblinded),
                self.compare_multi_sequence, n))

        # check frozen_parameters
        if self.frozen_parameters is not None and len(self.frozen_parameters) not in (0, 1, n):
            raise Exception("{!r}: the number of --frozen-parameters sequences ({}) must be zero, "
                "one or match that of {} ({})".format(self, len(self.frozen_parameters),
                self.compare_multi_sequence, n))

        # check frozen_groups
        if self.frozen_groups is not None and len(self.frozen_groups) not in (0, 1, n):
            raise Exception("{!r}: the number of --frozen-groups sequences ({}) must be zero, "
                "one or match that of {} ({})".format(self, len(self.frozen_groups),
                self.compare_multi_sequence, n))

    @abstractproperty
    def compare_multi_sequence(self):
        return

    def get_multi_task_kwargs(self):
        n = len(getattr(self, self.compare_multi_sequence))

        attrs = ["unblinded", "frozen_parameters", "frozen_groups"]
        keys = []
        values = []

        for i, attr in enumerate(attrs):
            value = getattr(self, attr)
            if value is None:
                continue
            if len(value) == 0:
                value = ((),) * n
            elif len(value) == 1:
                value *= n
            keys.append(attr)
            values.append(value)

        return [
            {key: _values[i] for key, _values in zip(keys, values)}
            for i in range(n)
        ]


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
                raise Exception("scan parameters {} must match POIs {} and vice versa".format(
                    self.joined_scan_parameter_names, self.joined_pois))
            unknown = set(self.scan_parameter_names) - set(self.pois)
            if unknown:
                raise Exception("scan parameters {} must match POIs {} and vice versa".format(
                    self.joined_scan_parameter_names, self.joined_pois))

        # check if scan parameters and pois diverge
        if self.force_scan_parameters_unequal_pois:
            if set(self.pois).intersection(set(self.scan_parameter_names)):
                raise Exception("scan parameters {} and POIs {} must not overlap".format(
                    self.joined_scan_parameter_names, self.joined_pois))

        # check if parameter values are in scan parameters
        if not self.allow_parameter_values_in_scan_parameters:
            for p in self.parameter_values_dict:
                if p in self.scan_parameter_names:
                    raise Exception("parameter values are not allowed to be in scan "
                        "parameters, but found {}".format(p))

        # check if no scan parameter is in custom parameter ranges
        for p in self.scan_parameter_names:
            if p in self.parameter_ranges_dict:
                raise Exception("scan parameters are not allowed to be in parameter ranges, "
                    "but found {}".format(p))

    def _joined_parameter_values_pois(self):
        pois = super(POIScanTask, self)._joined_parameter_values_pois()

        # skip scan parameters
        pois = [p for p in pois if p not in self.scan_parameter_names]

        return pois

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


class POIPlotTask(PlotTask, POITask):

    show_parameters = law.MultiCSVParameter(
        default=(),
        significant=False,
        description="colon-separated list of parameters that are shown in the plot even if they "
        "are 1; parameters separated by comma are shown in the same line when their value is "
        "identical same; no default",
    )

    def get_shown_parameters(self):
        parameter_values = self._joined_parameter_values(join=False)
        shown_parameters = OrderedDict()

        # add those requested explicitly
        for names in self.show_parameters:
            groups = OrderedDict()
            for name in names:
                if name not in parameter_values:
                    continue
                groups.setdefault(parameter_values[name], []).append(name)
                parameter_values.pop(name)
            for value, names in groups.items():
                shown_parameters[tuple(names)] = value

        # add remaining ones
        for name, value in list(parameter_values.items()):
            if name in poi_data and value != poi_data[name].sm_value:
                shown_parameters[(name,)] = value

        return shown_parameters


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
    minimizer = law.MultiCSVParameter(
        default=(("0", "0.1"), ("0", "0.2"), ("0", "0.4")),
        min_len=2,
        max_len=4,
        significant=False,
        description="multiple, colon-separated minimizer configurations in the format "
        "'[algo,][subalgo,]strategy,tolerance'; algo defaults to Minuit2; subalgo has no default; "
        "default: 0,0.1:0,0.2:0,0.4",
    )

    combine_discrete_options = (
        " --X-rtd MINIMIZER_freezeDisassociatedParams"
        " --X-rtd MINIMIZER_multiMin_hideConstants"
        " --X-rtd MINIMIZER_multiMin_maskConstraints"
        " --X-rtd MINIMIZER_multiMin_maskChannels=2"
    )

    option_aliases = {
        "-d": ["--datacard"],
        "-M": ["--method"],
        "-t": ["--toys"],
        "-v": ["--verbose"],
        "-m": ["--mass"],
        "-s": ["--seed"],
        "-w": ["--workspaceName"],
        "-n": ["--name"],
        "-g": ["-group"],
        "-C": ["--cl"],
        "-D": ["--dataset"],
        "-H": ["--hintMethod"],
        "-L": ["--LoadLibrary"],
        "--blind": ["--run expected", "--noFitAsimov"],
    }

    multi_options = ["--cminFallbackAlgo", "--LoadLibrary", "--keyword-value", "--X-rtd", "--X-rt"]

    exclude_index = True

    def __init__(self, *args, **kwargs):
        super(CombineCommandTask, self).__init__(*args, **kwargs)

        # generic hook to change task parameters in a customizable way
        self.call_hook("init_combine_command_task")

    def get_minimizer_args(self, skip_default=False):
        args = ""
        if not self.minimizer:
            return args

        def parse(m):
            # returns algo, subalgo, strategy, tolerance
            if len(m) == 2:
                return "Minuit2", None, m[0], m[1]
            elif len(m) == 3:
                return m[0], None, m[1], m[2]
            return m

        if not skip_default:
            algo, subalgo, strat, tol = parse(self.minimizer[0])
            args += " --cminDefaultMinimizerType {}".format(algo)
            args += " --cminDefaultMinimizerStrategy {}".format(strat)
            args += " --cminDefaultMinimizerTolerance {}".format(tol)
            if subalgo:
                args += " --cminDefaultMinimizerAlgo {}".format(subalgo)

        for m in self.minimizer[(0 if skip_default else 1):]:
            algo, subalgo, strat, tol = parse(m)
            subalgo = ("," + subalgo) if subalgo else ""
            args += " --cminFallbackAlgo {}{},{}:{}".format(algo, subalgo, strat, tol)

        return args

    @property
    def combine_optimization_args(self):
        # start with minimizer args
        args = self.get_minimizer_args()

        # additional args
        args += " --X-rtd REMOVE_CONSTANT_ZERO_POINT=1"

        # optimization for discrete parameters (as suggested by bbgg)
        if self.optimize_discretes:
            args += " " + self.combine_discrete_options

        return args

    def get_command(self):
        cmd = super(CombineCommandTask, self).get_command()

        def is_number(s):
            try:
                return isinstance(float(s), float)
            except:
                return False

        def is_opt(s):
            return s.startswith("-") and not is_number(s[1:])

        # highlighting
        cyan = lambda s: law.util.colored(s, "cyan")
        cyan_bright = lambda s: law.util.colored(s, "cyan", style="bright")
        bright = lambda s: law.util.colored(s, "default", style="bright")

        # extract the "combine ..." part
        cmds = []
        highlighted_cmds = []
        for c in (c.strip() for c in cmd.split(" && ")):
            if not c.startswith("combine "):
                cmds.append(c)
                highlighted_cmds.append(cyan(c))
                continue

            # first, replace aliases
            parts = []
            for part in shlex.split(c)[1:]:
                if is_opt(part) and part in self.option_aliases:
                    parts.extend(self.option_aliases[part])
                else:
                    parts.append(part)

            # group into (opt, value|None) pairs
            leading_values = []
            params = []
            for part in parts:
                if is_opt(part):
                    params.append([part])
                elif params:
                    params[-1].append(part)
                else:
                    leading_values.append(part)

            # list indices of occurence per option
            indices = defaultdict(list)
            for j, param in enumerate(params):
                indices[param[0]].append(j)

            # remove all but the first occurrence of options that are allowed only once
            params = [
                param for j, param in enumerate(params)
                if param[0] in self.multi_options or j == min(indices[param[0]])
            ]

            # remove params with existing, but all empty values
            params = [
                param for param in params
                if len(param) == 1 or any(param[1:])
            ]

            # allow lazy, custom modification via hooks
            _params = self.call_hook("modify_combine_params", params=params)
            if _params:
                params = _params

            # build the full combine command again
            combine_cmd = law.util.quote_cmd(["combine"] + leading_values + law.util.flatten(params))

            # build the highlighted command
            highlighted_cmd = []
            for i, p in enumerate(shlex.split(combine_cmd)):
                func = cyan_bright if (i == 0 and p == "combine") or p.startswith("--") else cyan
                highlighted_cmd.append(func(p))

            # add them
            cmds.append(combine_cmd)
            highlighted_cmds.append(" ".join(highlighted_cmd))

        return " && ".join(cmds), bright(" && ").join(highlighted_cmds)


class InputDatacards(DatacardTask, law.ExternalTask):

    version = None
    mass = None
    hh_model = law.NO_STR

    allow_empty_hh_model = True
    skip_inject_files = True

    def output(self):
        return [law.LocalFileTarget(self.split_datacard_path(card)[0]) for card in self.datacards]


class CombineDatacards(DatacardTask, CombineCommandTask):

    priority = 100

    allow_empty_hh_model = True
    allow_workspace_input = False
    skip_inject_files = True

    def __init__(self, *args, **kwargs):
        super(DatacardTask, self).__init__(*args, **kwargs)

        # complain when no datacard paths are given but the store path does not exist yet
        if not self.datacards and not self.local_target(dir=True).exists():
            raise Exception("{!r}: store directory {} does not exist which is required when no "
                "datacard paths are provided".format(self, self.local_target(dir=True).path))

    def requires(self):
        return InputDatacards.req(self)

    def output(self):
        return self.local_target("datacard.txt")

    def build_command(self, input_cards=None, output_card=None):
        if not input_cards:
            input_cards = self.datacards
        if not output_card:
            output_card = self.output().path

        # when there is only one input datacard without a bin name and no custom args, copy the card
        if not self.custom_args and len(input_cards) == 1 \
                and not re.match(r"^[^\/]+=.+$", input_cards[0]):
            return "cp {} {}".format(
                input_cards[0],
                output_card,
            )
        else:
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
            raise Exception("{!r}: datacard paths required".format(self))

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
        if not self.hh_model_empty:
            mod, model = self.load_hh_model()

            # build two sets of all available hh processes and those actually used in the model
            all_procs = set()
            model_procs = set()
            for r_poi, proc in [("r_gghh", "ggf"), ("r_qqhh", "vbf"), ("r_vhh", "vhh")]:
                all_samples = getattr(mod, proc + "_samples", None)
                formula = getattr(model, proc + "_formula", None)
                if not all_samples or not formula:
                    continue
                all_procs |= {s.label for s in all_samples.values()}
                model_procs |= {s.label for s in formula.samples}

            # subtract the sets to see which processes to remove
            to_remove = all_procs - model_procs
            if to_remove:
                from dhi.scripts.remove_processes import remove_processes
                self.logger.info("trying to remove processe(s) '{}' from the combined datacard as "
                    "they are not part of the phyics model {}".format(
                        ",".join(to_remove), self.hh_model))
                remove_processes(output_card.path, map("{}*".format, to_remove))

            # remove the THU_HH nuisance if not added (probably in listed in nuisances group)
            if not model.opt("doklDependentUnc"):
                from dhi.scripts.remove_parameters import remove_parameters
                self.logger.info("trying to remove '{}' from the combined datacard as the model "
                    "does not add need it".format(model.ggf_kl_dep_unc))
                remove_parameters(output_card.path, [model.ggf_kl_dep_unc])

        # copy shape files and the datacard to the output location
        output = self.output()
        output.parent.touch()
        with self.publish_step("moving shape files ..."):
            for basename in tmp_dir.listdir(pattern="*.root", type="f"):
                tmp_dir.child(basename).move_to(output.parent)
        output_card.copy_to(output)


class CreateWorkspace(DatacardTask, CombineCommandTask):

    priority = 90

    allow_empty_hh_model = True
    run_command_in_tmp = True

    def requires(self):
        if self.input_is_workspace:
            return []
        else:
            return CombineDatacards.req(self)

    def output(self):
        if self.input_is_workspace:
            return law.LocalFileTarget(self.datacards[0], external=True)
        else:
            return self.local_target("workspace.root")

    def build_command(self):
        if self.input_is_workspace:
            return None

        # build physics model arguments when not empty
        model_args = []
        if not self.hh_model_empty:
            model = self.load_hh_model()[1]
            model_args.append("--physics-model {model.__module__}:{model.name}".format(model=model))
            # add options
            for name, opt in model.hh_options.items():
                model_args.append("--physics-option {}={}".format(name, opt["value"]))

        # build the t2w command
        cmd = (
            "text2workspace.py {datacard}"
            " {self.custom_args}"
            " --out workspace.root"
            " --mass {self.mass}"
            " {model_args}"
        ).format(
            self=self,
            datacard=self.input().path,
            model_args=" ".join(model_args),
        )

        # add optional workspace injection commands
        for path in self.workspace_inject_files:
            cmd += " && inject_fit_result.py {} workspace.root w".format(path)

        # add the move command
        cmd += " && mv workspace.root {}".format(self.output().path)

        return cmd

    @law.decorator.log
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self):
        if self.input_is_workspace:
            raise Exception("{!r}: the input is a workspace which should already exist before this "
                "task is executed")

        # check if inject files exist
        for path in self.workspace_inject_files:
            if not os.path.exists(path):
                raise Exception("workspace inject file {} does not exist".format(path))

        return super(CreateWorkspace, self).run()
