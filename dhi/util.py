# coding: utf-8

"""
Helpers and utilities.
"""


import itertools


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


def rgb(r, g, b):
    """
    This function norms the rgb color inputs for
    matplotlib in case they are not normalized to 1.
    Otherwise just return inputs as tuple.
    Additionally this method displays the color inline,
    when using the atom package "pigments"!
    """
    return tuple(v if v <= 1.0 else float(v) / 255.0 for v in (r, g, b))


def linspace(start, stop, steps, precision=7):
    """
    Same as np.linspace with *start*, *stop* and *steps* being directly forwarded but the generated
    values are rounded to a certain *precision* and returned in a list.
    """
    import numpy as np
    return np.linspace(start, stop, steps).round(precision).tolist()


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


def minimize_1d(objective, bounds, niter=10, **kwargs):
    """
    Performs a 1D minimization of an *objective* using scipy.optimize.basinhopping over *niter*
    iterations within certain parameter *bounds*. These bounds are used initially to get a good
    starting point. *kwargs* are forwarded as *minimizer_kwargs* to the underlying minimizer
    function. The optimizer result of the lowest optimization iteration is returned.
    """
    import numpy as np
    import scipy.optimize

    # get the minimal starting point from a simple scan
    x = np.linspace(bounds[0], bounds[1], 100)
    y = objective(x).flatten()
    start = x[np.argmin(y)]

    # minimization using basin hopping
    kwargs["bounds"] = [bounds]
    res = scipy.optimize.basinhopping(objective, start, niter=10, minimizer_kwargs=kwargs)

    return res.lowest_optimization_result
