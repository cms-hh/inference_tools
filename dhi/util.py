# coding: utf-8

"""
Helpers and utilities.
"""

import os
import sys
import re
import shutil
import itertools
import array
import contextlib
import tempfile
import operator
import logging
from collections import OrderedDict

import six
from law.util import multi_match, make_unique, make_list  # noqa

# modules and objects from lazy imports
_plt = None
_ROOT = None


def import_plt():
    """
    Lazily imports and configures matplotlib pyplot.
    """
    global _plt

    if not _plt:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rc("text", usetex=True)
        matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
        matplotlib.rcParams["legend.edgecolor"] = "white"
        import matplotlib.pyplot as plt

        _plt = plt

    return _plt


def import_ROOT():
    """
    Lazily imports and configures ROOT.
    """
    global _ROOT

    if not _ROOT:
        import ROOT

        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gROOT.SetBatch()

        _ROOT = ROOT

    return _ROOT


def import_file(path, attr=None):
    """
    Loads the content of a python file located at *path* and returns its package content as a
    dictionary. When *attr* is set, only the attribute with that name is returned.

    The file is not required to be importable as its content is loaded directly into the
    interpreter. While this approach is not necessarily clean, it can be useful in places where
    custom user code must be loaded.
    """
    # load the package contents (do not try this at home)
    path = os.path.expandvars(os.path.expanduser(path))
    pkg = {}
    with open(path, "r") as f:
        exec(f.read(), pkg)

    # extract a particular attribute
    if attr:
        if attr not in pkg:
            raise AttributeError("no local member '{}' found in file {}".format(attr, path))
        return pkg[attr]

    return pkg


class DotDict(dict):
    """
    Dictionary providing item access via attributes.
    """

    def __getattr__(self, attr):
        return self[attr]

    def copy(self):
        return self.__class__(super(DotDict, self).copy())


def expand_path(path):
    """
    Takes a *path* and recursively expands all contained environment variables.
    """
    while "$" in path or "~" in path:
        path = os.path.expandvars(os.path.expanduser(path))
    return path


def real_path(path):
    """
    Takes a *path* and returns its real, absolute location with all variables expanded.
    """
    return os.path.realpath(expand_path(path))


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def to_root_latex(s):
    """
    Converts latex expressions in a string *s* to ROOT-compatible latex.
    """
    s = re.sub(r"(\$|\\,|\\;)", "", s)
    s = re.sub(r"\~", " ", s)
    s = re.sub(r"\\(\{|\})", r"\1", s)
    s = re.sub(r"\\", "#", s)
    return s


shell_colors = {
    "default": 39,
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "light_gray": 37,
    "dark_gray": 90,
    "light_red": 91,
    "light_green": 92,
    "light_yellow": 93,
    "light_blue": 94,
    "light_magenta": 95,
    "light_cyan": 96,
    "white": 97,
}


def colored(msg, color=None, force=False):
    """
    Return the colored version of a string *msg*. Unless *force* is *True*, the *msg* string is
    returned unchanged in case the output is not a tty. Simplified from law.util.colored.
    """
    if not force:
        try:
            tty = os.isatty(sys.stdout.fileno())
        except:
            tty = False

        if not tty:
            return msg

    color = shell_colors.get(color, shell_colors["default"])

    return "\033[{}m{}\033[0m".format(color, msg)


def warn(msg, color="yellow", force=False):
    """
    Prints a warning *msg* with a default *color*. *force* is forwarded to *colored*.
    """
    print(colored(msg, color=color, force=force))


def linspace(start, stop, steps, precision=7):
    """
    Same as np.linspace with *start*, *stop* and *steps* being directly forwarded but the generated
    values are rounded to a certain *precision* and returned in a list.
    """
    import numpy as np

    return np.linspace(start, stop, steps).round(precision).tolist()


def get_neighbor_coordinates(shape, i, j):
    """
    Given a 2D shape and the coordinates *i* and *j* of a "pixel", returns a list of coordinates of
    neighboring pixels in a 3x3 grid.
    """
    # check inputs
    if len(shape) != 2:
        raise ValueError("shape must have length 2, got {}".format(shape))
    if any(l <= 0 for l in shape):
        raise ValueError("shape must contain only positive numbers, got {}".format(shape))
    if not (0 <= i < shape[0]):
        raise ValueError("i must be within interval [0, shape[0])")
    if not (0 <= j < shape[1]):
        raise ValueError("j must be within interval [0, shape[1])")

    # determine coordinates of the window
    i_start = max(0, i - 1)
    i_end = min(shape[0] - 1, i + 1)
    j_start = max(0, j - 1)
    j_end = min(shape[1] - 1, j + 1)

    # build neighbors
    neighbors = list(itertools.product(range(i_start, i_end + 1), range(j_start, j_end + 1)))

    # remove (i, j) again
    neighbors.remove((i, j))

    return neighbors


def minimize_1d(objective, bounds, start=None, **kwargs):
    """
    Performs a 1D minimization of an *objective* using scipy.optimize.basinhopping (fowarding all
    *kwargs*) within certain parameter *bounds* provided as a 2-tuple. When *start* is *None*, these
    bounds are used initially to get a good starting point. The result of the best optimization
    iteration is returned.
    """
    import numpy as np
    import scipy.optimize

    # get the minimal starting point from a simple scan within bounds
    if start is None:
        x = np.linspace(bounds[0], bounds[1], 100)
        y = objective(x).flatten()
        start = x[np.argmin(y)]

    # minimization using basin hopping
    kwargs.setdefault("niter", 50)
    minimizer_kwargs = kwargs.setdefault("minimizer_kwargs", {})
    minimizer_kwargs["bounds"] = [bounds]
    minimizer_kwargs.setdefault("tol", 0.00001)
    res = scipy.optimize.basinhopping(objective, start, **kwargs)

    return res.lowest_optimization_result


def minimize_2d(objective, bounds, start=None, **kwargs):
    """
    Performs a 2D minimization of an *objective* using scipy.optimize.basinhopping (fowarding all
    *kwargs*) within certain parameter *bounds* provided as two 2-tuples in a list. When *start* is
    *None*, these bounds are used initially to get a good starting point. The result of the best
    optimization iteration is returned.
    """
    import numpy as np
    import scipy.optimize

    # get the minimal starting point from a simple scan within bounds
    if start is None:
        x = np.array(np.meshgrid(
            np.linspace(bounds[0][0], bounds[0][1], 100),
            np.linspace(bounds[1][0], bounds[1][1], 100),
        )).T.reshape(-1, 2)
        y = objective(x).flatten()
        start = tuple(x[np.argmin(y)].tolist())

    # minimization using basin hopping
    kwargs.setdefault("niter", 50)
    minimizer_kwargs = kwargs.setdefault("minimizer_kwargs", {})
    minimizer_kwargs["bounds"] = bounds
    minimizer_kwargs.setdefault("tol", 0.00001)
    res = scipy.optimize.basinhopping(objective, start, **kwargs)

    return res.lowest_optimization_result


def create_tgraph(n, *args, **kwargs):
    """create_tgraph(n, *args, pad=None, insert=None)
    Creates a ROOT graph with *n* points, where the type is *TGraph* for two, *TGraphErrors* for
    4 and *TGraphAsymmErrors* for six *args*. Each argument is converted to a python array with
    typecode "f". When *pad* is *True*, the graph is padded by one additional point on each side
    with the same edge value. When *insert* is given, it should be a list of tuples with values
    ``(index, values...)`` denoting the index, coordinates and errors of points to be inserted.
    """
    ROOT = import_ROOT()

    if len(args) <= 2:
        cls = ROOT.TGraph
    elif len(args) <= 4:
        cls = ROOT.TGraphErrors
    else:
        cls = ROOT.TGraphAsymmErrors

    # expand single values
    _args = []
    for a in args:
        if not getattr(a, "__len__", None):
            a = n * [a]
        elif len(a) == 1:
            a = n * list(a)
        _args.append(list(a))

    # apply edge padding when requested with a configurable width
    pad = kwargs.get("pad")
    if pad:
        w = 1 if not isinstance(pad, int) else int(pad)
        n += 2 * w
        _args = [(w * a[:1] + a + w * a[-1:]) for a in _args]

    # insert custom points
    insert = kwargs.get("insert")
    if insert:
        for values in insert:
            idx, values = values[0], values[1:]
            for i, v in enumerate(values):
                _args[i].insert(idx, v)
            n += 1

    if n == 0:
        return cls(n)
    else:
        return cls(n, *(array.array("f", a) for a in _args))


def convert_rooargset(argset):
    """
    Helper to convert a RooArgSet into a dictionary mapping names to value-errors pairs.
    """
    data = OrderedDict()

    it = argset.createIterator()
    while True:
        param = it.Next()
        if not param:
            break
        data[param.GetName()] = (param.getVal(), param.getErrorHi(), param.getErrorLo())

    return data


def copy_no_collisions(path, directory, postfix_format="_{}"):
    """
    Copies a file given by *path* into a *directory*. When a file with the same basename already
    exists in this directory, a number is added as a postfix using *postfix_format* before the last
    file extension. The full path to the created file is returned.
    """
    # prepare the dst directory
    directory = os.path.expandvars(os.path.expanduser(directory))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # get the expanded src path
    src_path = os.path.expandvars(os.path.expanduser(path))

    # determine the collision-free dst path
    dst_path = os.path.join(directory, os.path.basename(src_path))
    dst_tmpl = "{}{{}}{}".format(*os.path.splitext(dst_path))
    i = 0
    while os.path.exists(dst_path):
        i += 1
        dst_path = dst_tmpl.format(postfix_format.format(i))

    # copy
    shutil.copy2(src_path, dst_path)

    return dst_path


def create_console_logger(name, level="INFO", formatter=None):
    """
    Creates a console logger named *name* and returns it. The initial log *level* can either be an
    integer or a string denoting a Python log level. When *formatter* is not set, the default log
    formatter of law is used.
    """
    if formatter is None:
        import law

        formatter = law.logger.LogFormatter()

    logger = logging.getLogger(name)

    # add handler and its formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # patch setLevel to accept more level types
    set_level_orig = logger.setLevel

    def set_level(level):
        if not isinstance(level, int):
            try:
                level = int(level)
            except:
                level = getattr(logging, level.upper(), 0)
        set_level_orig(level)

    # set the initial level
    logger.setLevel(level)

    return logger


@contextlib.contextmanager
def patch_object(obj, attr, value):
    """
    Context manager that temporarily patches an object *obj* by replacing its attribute *attr* with
    *value*. The original value is recovered again when the context terminates.
    """
    no_value = object()
    orig = getattr(obj, attr, no_value)

    try:
        setattr(obj, attr, value)
        yield obj

    finally:
        try:
            if orig is no_value:
                delattr(obj, attr)
            else:
                setattr(obj, attr, orig)
        except:
            pass


@contextlib.contextmanager
def disable_output():
    """
    Context manager that redirects all logs to stdout and stderr in its context.
    """
    with open("/dev/null", "w") as f:
        with patch_object(sys, "stdout", f):
            yield


def try_int(n):
    """
    Takes a number *n* and tries to convert it to an integer. When *n* has no decimals, an integer
    is returned with the same value as *n*. Otherwise, a float is returned.
    """
    n_int = int(n)
    return n_int if n == n_int else n


def poisson_asym_errors(v):
    """
    Returns asymmetric poisson errors for a value *v* in a tuple (up, down) following the Garwoord
    prescription (1936). For more info, see
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars and
    https://root.cern.ch/doc/master/TH1_8cxx_source.html#l08537.
    """
    ROOT = import_ROOT()

    v_int = int(v)
    alpha = 1.0 - 0.682689492

    err_up = ROOT.Math.gamma_quantile_c(alpha / 2.0, v_int + 1, 1) - v
    err_down = 0.0 if v == 0 else (v - ROOT.Math.gamma_quantile(alpha / 2, v_int, 1.0))

    return err_up, err_down


def unique_recarray(a, cols=None, sort=True, test_metric=None):
    import numpy as np

    metric, test_fn = test_metric or (None, None)

    # concatenate multiple input arrays
    if isinstance(a, (list, tuple)):
        a = np.concatenate(a, axis=0)

    # use all columns by default, except for the optional test metric
    if not cols:
        cols = list(a.dtype.names)
        if metric and metric in cols:
            cols.remove(metric)
    else:
        cols = list(cols)

    # get the indices of unique entries and sort them
    indices = np.unique(a[cols], return_index=True)[1]

    # by default, indices are ordered such that the columns used to identify duplicates are sorted
    # so when sort is True, keep it that way, and otherwise sort indices to preserve the old order
    if not sort:
        indices = sorted(indices)

    # create the unique array
    b = np.array(a[indices])

    # perform a check to see if removed values differ in a certain metric
    if metric:
        removed_indices = set(range(a.shape[0])) - set(indices)
        for i in removed_indices:
            # get the removed metric value
            removed_metric = float(a[i][metric])
            # get the kept metric value
            j = six.moves.reduce(operator.and_, [b[c] == v for c, v in zip(cols, a[i][cols])])
            j = np.argwhere(j).flatten()[0]
            kept_metric = float(b[j][metric])
            # call test_fn except when both values are nan
            both_nan = np.isnan(removed_metric) and np.isnan(kept_metric)
            if not both_nan and not test_fn(kept_metric, removed_metric):
                raise Exception("duplicate entries identified by columns {} with '{}' values of {} "
                    "(kept) and {} (removed at row {}) differ".format(
                        cols, metric, kept_metric, removed_metric, i))

    return b


def dict_to_recarray(dicts):
    """
    Converts one (or multiple) dictionaries into a recarray with arrays fields being interpreted as
    columns. When multiple dictionaries are given, the resulting recarrays are concatenated if their
    dtypes are identical.
    """
    import numpy as np

    dicts = make_list(dicts)
    if not dicts:
        return None

    first_keys = list(dicts[0].keys())
    dtype = [(key, np.float32) for key in first_keys]

    arrays = []
    for i, d in enumerate(dicts):
        # check if keys are identical to first dict
        if set(d.keys()) != set(first_keys):
            raise Exception("keys of dictionary {} ({}) do not match those of first dictionary "
                "({})".format(i, ",".join(d.keys()), ",".join(first_keys)))

        # check if all values (lists) have the same length
        value_lengths = set(map(len, d.values()))
        if len(value_lengths) != 1:
            raise Exception("dictionary {} found to map to lists with unequal lengths: {}".format(
                d, value_lengths))

        # construct the recarray
        records = [tuple(v[i] for v in d.values()) for i in range(list(value_lengths)[0])]
        arrays.append(np.array(records, dtype=dtype))

    # concatenate
    return np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]


def extend_recarray(a, *columns, **kwargs):
    """
    Extends a recarray *a* by one or more additional *columns*. A column should be a 3-tuple
    containing the name, the dtype and the data of the column to add. Additional *kwargs* are
    forwarded to numpy.lib.recfunctions.append_fields. The new recarray is returned.
    """
    import numpy.lib.recfunctions as rf

    names = [col[0] for col in columns]
    dtypes = [col[1] for col in columns]
    data = [col[2] for col in columns]

    return rf.append_fields(a, names, data=data, dtypes=dtypes, **kwargs)


def convert_dnll2(dnll2, n=1):
    """
    Converts a -2∆ln(L) value *dnll2* from a likelihood profile scanning *n* parameters (with its
    minimum shifted to zero) into a p-value and the corresponding significance in gaussian standard
    deviations, returned in a 2-tuple. For reference, see Fig. 40.4 and Tab. 40.2 in
    https://pdg.lbl.gov/2020/reviews/rpp2020-rev-statistics.pdf.
    """
    import numpy as np
    from scipy import stats

    # convert to array
    single_value = isinstance(dnll2, six.integer_types + (float,))
    dnll2 = np.array([dnll2] if single_value else dnll2)

    # compute the significance
    # the precision of pdf/ppf can't handle very high dnll2 (== low p-values)
    # so using (inverse) survival function instead
    alpha = stats.chi2.sf(dnll2, n)  # same as 1 - chi2.cdf(dnll2)
    p_value = alpha / 2.
    sig = stats.norm.isf(p_value)  # same as chi2.ppf(1 - p_value)

    # replace values where dnll2 is <= 0 by 0 (probably nan)
    sig[dnll2 <= 0] = 0.

    # optionally convert back to a single value
    if single_value:
        p_value = float(p_value)
        sig = float(sig)

    return p_value, sig


class TFileCache(object):

    def __init__(self, logger=None):
        super(TFileCache, self).__init__()

        self.logger = logger or logging.getLogger(
            "{}_{}".format(self.__class__.__name__, hex(id(self)))
        )

        # cache of files opened for reading
        # abs_path -> {tfile: TFile}
        self._r_cache = {}

        # cache of files opened for writing
        # abs_path -> {
        #     tmp_path: str or None,
        #     tfile: TFile,
        #     write_objects: [(tobj, towner, name), ...],
        #     delete_objects: [abs_key, ...],
        # }
        self._w_cache = {}

    def __enter__(self):
        return self

    def __exit__(self, err_type, err_value, traceback):
        self.finalize(skip_write=err_type is not None, skip_delete=err_type is not None)

    def _clear(self):
        # close and clear r_cache
        for data in self._r_cache.values():
            if data["tfile"] and data["tfile"].IsOpen():
                data["tfile"].Close()
        self._r_cache.clear()

        # close and clear w_cache
        for data in self._w_cache.values():
            if data["tfile"] and data["tfile"].IsOpen():
                data["tfile"].Close()
        self._w_cache.clear()

    def open_tfile(self, path, mode, tmp=True):
        ROOT = import_ROOT()

        abs_path = real_path(path)

        if mode == "READ":
            if abs_path not in self._r_cache:
                # just open the file and cache it
                tfile = ROOT.TFile(abs_path, mode)
                self._r_cache[abs_path] = {"tfile": tfile}

                self.logger.debug("opened tfile {} with mode {}".format(abs_path, mode))

            return self._r_cache[abs_path]["tfile"]

        else:
            if abs_path not in self._w_cache:
                if tmp:
                    # determine a temporary location
                    suffix = "_" + os.path.basename(abs_path)
                    tmp_path = tempfile.mkstemp(suffix=suffix)[1]
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    # copy the file when existing
                    if os.path.exists(abs_path):
                        shutil.copy2(abs_path, tmp_path)

                    # open the file
                    tfile = ROOT.TFile(tmp_path, mode)

                    self.logger.debug("opened tfile {} with mode {} in temporary location {}".format(
                        abs_path, mode, tmp_path))
                else:
                    # open the file
                    tfile = ROOT.TFile(abs_path, mode)
                    tmp_path = None

                    self.logger.debug("opened tfile {} with mode {}".format(abs_path, mode))

                # store it
                self._w_cache[abs_path] = {
                    "tmp_path": tmp_path,
                    "tfile": tfile,
                    "write_objects": [],
                    "delete_objects": [],
                }

            return self._w_cache[abs_path]["tfile"]

    def write_tobj(self, path, tobj, towner=None, name=None):
        ROOT = import_ROOT()

        if isinstance(path, ROOT.TFile):
            # lookup the cache entry by the tfile reference
            for data in self._w_cache.values():
                if data["tfile"] == path:
                    data["write_objects"].append((tobj, towner, name))
                    break
            else:
                raise Exception("cannot write object {} into unknown TFile {}".format(
                    tobj, path))

        else:
            abs_path = real_path(path)
            if abs_path not in self._w_cache:
                raise Exception("cannot write object {} into unopened file {}".format(tobj, path))

            self._w_cache[abs_path]["write_objects"].append((tobj, towner, name))

    def delete_tobj(self, path, abs_key):
        ROOT = import_ROOT()

        if isinstance(path, ROOT.TFile):
            # lookup the cache entry by the tfile reference
            for data in self._w_cache.values():
                if data["tfile"] == path:
                    data["delete_objects"].append(abs_key)
                    break
            else:
                raise Exception("cannot delete object {} from unknown TFile {}".format(
                    abs_key, path))

        else:
            abs_path = real_path(path)
            if abs_path not in self._w_cache:
                raise Exception("cannot delete object {} from unopened file {}".format(
                    abs_key, path))

            self._w_cache[abs_path]["delete_objects"].append(abs_key)

    def finalize(self, skip_write=False, skip_delete=False):
        if self._r_cache:
            # close files opened for reading
            for abs_path, data in self._r_cache.items():
                if data["tfile"] and data["tfile"].IsOpen():
                    data["tfile"].Close()
            self.logger.debug("closed {} cached file(s) opened for reading".format(
                len(self._r_cache)))

        if self._w_cache:
            # close files opened for reading, write objects and move to actual location
            ROOT = import_ROOT()
            ignore_level_orig = ROOT.gROOT.ProcessLine("gErrorIgnoreLevel;")
            ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")

            for abs_path, data in self._w_cache.items():
                # stop when the tfile is empty
                if not data["tfile"]:
                    self.logger.warning("could not write empty tfile with data {}".format(data))
                    continue

                # issue a warning when the file was closed externally
                if not data["tfile"].IsOpen():
                    self.logger.warning("could not write tfile {}, already closed".format(
                        data["tfile"]))
                else:
                    # write objects
                    if not skip_write and data["write_objects"]:
                        data["tfile"].cd()
                        self.logger.debug("writing {} objects".format(len(data["write_objects"])))
                        for tobj, towner, name in data["write_objects"]:
                            if towner:
                                towner.cd()
                            args = (name,) if name else ()
                            tobj.Write(*args)

                    # delete objects
                    if not skip_delete and data["delete_objects"]:
                        data["tfile"].cd()
                        self.logger.debug("deleting {} objects".format(len(data["delete_objects"])))
                        for abs_key in data["delete_objects"]:
                            data["tfile"].Delete(abs_key)
                            self.logger.debug("deleted {} from tfile at {}".format(abs_key, abs_path))

                    # close the file
                    data["tfile"].Close()

                # move back to original place when it was temporary and something changed
                if data["tmp_path"] and (not skip_write or not skip_delete):
                    shutil.move(data["tmp_path"], abs_path)
                    self.logger.debug("moving back temporary file {} to {}".format(
                        data["tmp_path"], abs_path))

            self.logger.debug("closed {} cached file(s) opened for writing".format(
                len(self._w_cache)))
            ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = {};".format(ignore_level_orig))

        # clear
        self._clear()


class ROOTColorGetter(object):

    def __init__(self, **cache):
        super(ROOTColorGetter, self).__init__()

        self.cache = cache or {}

    def _get_color(self, attr):
        if attr not in self.cache:
            self.cache[attr] = self.create_color(attr)
        elif not isinstance(self.cache[attr], int):
            self.cache[attr] = self.create_color(self.cache[attr])

        return self.cache[attr]

    def __call__(self, *args, **kwargs):
        return self._get_color(*args, **kwargs)

    def __getattr__(self, attr):
        return self._get_color(attr)

    def __getitem__(self, key):
        return self._get_color(key)

    @classmethod
    def create_color(cls, obj):
        ROOT = import_ROOT()

        if isinstance(obj, int):
            return obj
        elif isinstance(obj, str):
            c = getattr(ROOT, "k" + obj.capitalize(), None)
            if c is not None:
                return c
        elif isinstance(obj, tuple) and len(obj) in [2, 3, 4]:
            c = ROOT.TColor.GetColor(*obj[:3]) if len(obj) >= 3 else obj[0]
            if len(obj) in [2, 4]:
                c = ROOT.TColor.GetColorTransparent(c, obj[-1])
            return c

        raise AttributeError("cannot interpret '{}' as color".format(obj))
