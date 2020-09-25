# coding: utf-8

"""
Helpers and utilities.
"""

import law
import luigi


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


def WIP(cls):
    """
    This decorator overwrites the constructor of any
    class and quits the program once it is instantiated.
    Additionally it removes the task from `law index`.
    It is useful to prohibit the execution and instantiation
    of law.Tasks/luigi.Tasks, which are currently Work in Progress (WIP).
    """

    def _wip(self, *args, **kwargs):
        raise SystemExit(
            "{} is under development. You can not instantiated this class.".format(cls.__name__)
        )

    if issubclass(cls, (law.Task, luigi.Task)):
        # remove from law index
        cls.exclude_index = True
        # also prevent instantiation
        cls.__init__ = _wip
    return cls


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def is_pow2(num):
    return (num & (num - 1) == 0) and num != 0


def next_pow2(num):
    k = 1
    while k < num:
        k = k << 1
    return k
