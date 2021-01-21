# coding: utf-8

"""
Helpers and utilities.
"""

import os
import sys
import re
import fnmatch
import shutil
import itertools
import array
import contextlib
import logging


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


class DotDict(dict):
    """
    Dictionary providing item access via attributes.
    """

    def __getattr__(self, attr):
        return self[attr]

    def copy(self):
        return self.__class__(super(DotDict, self).copy())


def real_path(path):
    """
    Takes a *path* and returns its real, absolute location with all variables expanded.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    path = os.path.realpath(path)
    return path


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def multi_match(name, patterns, mode=any, regex=False):
    """
    Compares *name* to multiple *patterns* and returns *True* in case of at least one match (*mode*
    = *any*, the default), or in case all patterns match (*mode* = *all*). Otherwise, *False* is
    returned. When *regex* is *True*, *re.match* is used instead of *fnmatch.fnmatch*.
    """
    if not isinstance(patterns, (list, tuple, set)):
        patterns = [patterns]
    if not regex:
        return mode(fnmatch.fnmatch(name, pattern) for pattern in patterns)
    else:
        return mode(re.match(pattern, name) for pattern in patterns)


def to_root_latex(s):
    """
    Converts latex expressions in a string *s* to ROOT-compatible latex.
    """
    s = re.sub(r"(\$|\\,|\\;)", "", s)
    s = re.sub(r"\~", " ", s)
    s = re.sub(r"\\", "#", s)
    return s


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


def minimize_1d(objective, bounds, start=None, niter=10, **kwargs):
    """
    Performs a 1D minimization of an *objective* using scipy.optimize.basinhopping over *niter*
    iterations within certain parameter *bounds*. When *start* is *None*, these bounds are used
    initially to get a good starting point. *kwargs* are forwarded as *minimizer_kwargs* to the
    underlying minimizer function. The optimizer result of the lowest optimization iteration is
    returned.
    """
    import numpy as np
    import scipy.optimize

    # get the minimal starting point from a simple scan within bounds
    if start is None:
        x = np.linspace(bounds[0], bounds[1], 100)
        y = objective(x).flatten()
        start = x[np.argmin(y)]

    # minimization using basin hopping
    kwargs["bounds"] = [bounds]
    res = scipy.optimize.basinhopping(objective, start, niter=10, minimizer_kwargs=kwargs)

    return res.lowest_optimization_result


def create_tgraph(n, *args, **kwargs):
    """
    Creates a ROOT graph with *n* points, where the type is *TGraph* for two, *TGraphErrors* for
    4 and *TGraphAsymmErrors* for six *args*. Each argument is converted to a python array with
    typecode "f".
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

    # apply edge padding when requested
    if kwargs.get("pad"):
        n += 2
        _args = [(a[:1] + a + a[-1:]) for a in _args]

    if n == 0:
        return cls(n)
    else:
        return cls(n, *(array.array("f", a) for a in _args))


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
    *value*. The original value is set again when the context is closed.
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


class ROOTColorGetter(object):
    def __init__(self, **cache):
        super(ROOTColorGetter, self).__init__()

        self.cache = cache or {}

    def __getattr__(self, attr):
        ROOT = import_ROOT()

        if attr not in self.cache:
            self.cache[attr] = self.create_color(attr)
        elif not isinstance(self.cache[attr], int):
            self.cache[attr] = self.create_color(self.cache[attr])

        return self.cache[attr]

    @classmethod
    def create_color(cls, obj):
        ROOT = import_ROOT()

        if isinstance(obj, int):
            return obj
        elif isinstance(obj, str):
            return getattr(ROOT, "k" + obj.capitalize())
        elif isinstance(obj, tuple) and len(obj) in [2, 3, 4]:
            c = ROOT.TColor.GetColor(*obj[:3]) if len(obj) >= 3 else obj[0]
            if len(obj) in [2, 4]:
                c = ROOT.TColor.GetColorTransparent(c, obj[-1])
            return c
        else:
            raise AttributeError("cannot interpret '{}' as color".format(obj))
