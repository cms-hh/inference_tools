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


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def get_ggf_xsec(ggf_formula, kl=1.0, kt=1.0):
    """
    Returns the ggF cross section for a combination of *kl* and *kt*, given a *ggf_formula* object.
    """
    return ggf_formula.sigma.evalf(
        subs={
            "kl": kl,
            "kt": kt,
            "s1": ggf_formula.sample_list[0].val_xs,
            "s2": ggf_formula.sample_list[1].val_xs,
            "s3": ggf_formula.sample_list[2].val_xs,
        }
    )[0]


def get_vbf_xsec(vbf_formula, c2v=1.0, cv=1.0, kl=1.0):
    """
    Returns the VBF cross section for a combination of *c2v*, *cv* and *kl*, given a *vbf_formula*
    object.
    """
    return vbf_formula.sigma.evalf(
        subs={
            "C2V": c2v,
            "CV": cv,
            "kl": kl,
            "s1": vbf_formula.sample_list[0].val_xs,
            "s2": vbf_formula.sample_list[1].val_xs,
            "s3": vbf_formula.sample_list[2].val_xs,
            "s4": vbf_formula.sample_list[3].val_xs,
            "s5": vbf_formula.sample_list[4].val_xs,
            "s6": vbf_formula.sample_list[5].val_xs,
        }
    )[0]
