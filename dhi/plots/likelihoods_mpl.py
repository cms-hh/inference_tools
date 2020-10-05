# coding: utf-8

"""
Likelihood plots using matplotlib.
"""

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib
from scinum import Number

from dhi.config import poi_labels, poi_labels_math, campaign_labels
from dhi.util import import_plt, DotDict, rgb


plt = import_plt()

# cumulative, inverse chi2 values in a mapping "n_dof -> n_sigma -> level"
# for the geometrical determination of errors of nll curves
chi2_levels = {
    1: {1: 1.000, 2: 4.000},
    2: {1: 2.296, 2: 6.180},
}


def plot_likelihood_scan_1d(path, poi, data, poi_min=None, campaign="2017", x_min=None,
        x_max=None):
    """
    Creates a likelihood plot of the 1D scan of a *poi* and saves it at *path*. *data* should be a
    mapping to lists of values or a record array with keys "<poi_name>" and "dnll2". When *poi_min*
    is set, it should be the value of the poi that leads to the best likelihood. Otherwise, it is
    estimated from the interpolated curve. *campaign* should refer to the name of a campaign label
    defined in dhi.config.campaign_labels. *x_min* and *x_max* define the x-axis range and default
    to the range of poi values.

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/?search=nll1d
    """
    # get poi and delta nll values
    poi_values = data[poi]
    dnll2_values = data["dnll2"]

    # set default x range
    if x_min is None:
        x_min = poi_values.min()
    if x_max is None:
        x_max = poi_values.max()

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_1d(poi_values, dnll2_values, poi_min=poi_min)

    # start plotting
    fig, ax = plt.subplots()

    # 1 and 2 sigma indicators
    for value in [scan.poi_p1, scan.poi_m1, scan.poi_p2, scan.poi_m2]:
        if value is not None:
            ax.plot(
                (value, value),
                (0.0, scan.interp(value)),
                linestyle="--",
                color=rgb(161, 16, 53),
            )

    # lines at 1 and 4
    for n in [1., 4.]:
        if n < dnll2_values.max():
            ax.plot(
                (x_min, x_max),
                (n, n),
                linestyle="--",
                color=rgb(161, 16, 53),
            )

    # line and text indicating the best fit value
    dnll2_max_visible = dnll2_values[(poi_values >= x_min) & (poi_values <= x_max)].max()
    best_line_max = dnll2_max_visible * 0.85
    best_label = r"$\mu_{{{}}} = {}$".format(
        poi_labels.get(poi, poi),
        scan.num_min.str(format="%.2f", style="latex"),
    )
    ax.plot(
        (scan.poi_min, scan.poi_min),
        (0, best_line_max),
        linestyle="-",
        color=rgb(161, 16, 53),
    )
    ax.annotate(best_label, (scan.poi_min, best_line_max * 1.02), ha="center")

    # nll curve
    ax.plot(
        poi_values,
        dnll2_values,
        linestyle="-",
        marker=".",
        color=rgb(0, 84, 159),
    )

    # legend, labels, titles, etc
    ax.set_xlabel(poi_labels_math.get(poi, poi))
    ax.set_ylabel(r"$-2 \Delta \text{ln}\mathcal{L}$")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    ax.set_title(r"\textbf{CMS} \textit{preliminary}", loc="left")
    ax.set_title(campaign_labels.get(campaign, campaign), loc="right")
    ax.grid()

    # save
    fig.tight_layout()
    fig.savefig(path)


def plot_likelihood_scan_2d(path, poi1, poi2, data, poi1_min=None, poi2_min=None, campaign="2017",
        x1_min=None, x1_max=None, x2_min=None, x2_max=None):
    """
    Creates a likelihood plot of the 2D scan of two pois *poi1* and *poi2*, and saves it at *path*.
    *data* should be a mapping to lists of values or a record array with keys "<poi1_name>",
    "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be the values
    of the pois that lead to the best likelihood. Otherwise, they are  estimated from the
    interpolated curve. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels. *x1_min*, *x1_max*, *x2_min* and *x2_max* define the axis range of
    poi1 and poi2, respectively, and default to the ranges of the poi values.

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/?search=nll2d
    """
    # get poi and delta nll values
    poi1_values = data[poi1]
    poi2_values = data[poi2]
    dnll2_values = data["dnll2"]

    # set default x ranges
    if x1_min is None:
        x1_min = poi1_values.min()
    if x1_max is None:
        x1_max = poi1_values.max()
    if x2_min is None:
        x2_min = poi2_values.min()
    if x2_max is None:
        x2_max = poi2_values.max()

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_2d(poi1_values, poi2_values, dnll2_values, poi1_min=poi1_min,
        poi2_min=poi2_min)

    # start plotting
    fig, ax = plt.subplots()

    counts, bins_x, bins_y, img = ax.hist2d(
        poi1_values,
        poi2_values,
        weights=dnll2_values,
        range=[[x1_min, x1_max], [x2_min, x2_max]],
        bins=[np.unique(poi1_values).size - 1, np.unique(poi2_values).size - 1],
        norm=matplotlib.colors.LogNorm(),
        cmap="viridis",
    )
    fig.colorbar(img, ax=ax, label=r"$-2 \Delta \text{ln}\mathcal{L}$")

    # contours
    contours = ax.contour(counts.transpose(),
        extent=[bins_x.min(), bins_x.max(), bins_y.min(), bins_y.max()],
        levels=np.log([chi2_levels[2][1], chi2_levels[2][2]]),
        linewidths=2,
    )
    fmt = {l: s for l, s in zip(contours.levels, [r"$1 \sigma$", r"$2 \sigma$"])}
    ax.clabel(contours, inline=True, fontsize=8, fmt=fmt)

    # best fit value marker with errors
    ax.errorbar(
        [scan.num1_min()],
        [scan.num2_min()],
        xerr=[[scan.num1_min("down", diff=True)], [scan.num1_min("up", diff=True)]],
        yerr=[[scan.num2_min("down", diff=True)], [scan.num2_min("up", diff=True)]],
        marker="o",
        color="red",
        markersize=4,
        elinewidth=1,
    )

    # best fit texts
    def best_fit_text(poi, num_min, dx2=0.05):
        best_label = r"$\mu_{{{}}} = {}$".format(
            poi_labels.get(poi, poi),
            num_min.str(format="%.2f", style="latex"),
        )
        ax.annotate(
            best_label,
            (x1_min + (x1_max - x1_min) * 0.75, x2_min + (x2_max - x2_min) * (0.97 - dx2)),
            ha="left",
            va="top",
        )
    best_fit_text(poi1, scan.num1_min, 0.)
    best_fit_text(poi2, scan.num2_min, 0.06)

    # legend, labels, titles, etc
    ax.set_xlabel(poi_labels_math.get(poi1, poi1))
    ax.set_ylabel(poi_labels_math.get(poi2, poi2))
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    ax.set_title(r"\textbf{CMS} \textit{preliminary}", loc="left")
    ax.set_title(campaign_labels.get(campaign, campaign), loc="right")

    # save
    fig.tight_layout()
    fig.savefig(path)


def evaluate_likelihood_scan_1d(poi_values, dnll2_values, poi_min=None):
    """
    Takes the results of a 1D likelihood profiling scan given by the *poi_values* and the
    corresponding *delta_2nll* values, performs an interpolation and returns certain results of the
    scan in a dict. When *poi_min* is *None*, it is estimated from the interpolated curve.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``poi_min``: The poi value corresponding to the minimum delta nll value.
    - ``poi_p1``: The poi value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi_m1``: The poi value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi_p2``: The poi value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi_m2``: The poi value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``num_min``: A Number instance representing the best fit value and its 1 sigma uncertainty.
    """
    # first, obtain an interpolation function and sample new points
    mask = ~np.isnan(dnll2_values)
    interp = scipy.interpolate.interp1d(poi_values[mask], dnll2_values[mask], kind="cubic")

    # get the minimum when not set
    bounds = (poi_values.min() + 1e-4, poi_values.max() - 1e-4)
    if poi_min is None:
        objective = lambda x: abs(interp(x))
        res = scipy.optimize.minimize(objective, 1., tol=1e-7, bounds=[bounds])
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi_min = res.x[0]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v):
        def minimize(start):
            objective = lambda x: abs(interp(x) - v)
            res = scipy.optimize.minimize(objective, start, tol=1e-7, bounds=[bounds])
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        return minimize(poi_values.max() - 1), minimize(poi_values.min() + 1)

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi_p1, poi_m1 = get_intersections(chi2_levels[1][1])
    poi_p2, poi_m2 = get_intersections(chi2_levels[1][2])

    # create a Number object wrapping the best fit value and its 1 sigma error when given
    unc = None
    if poi_p1 is not None and poi_m1 is not None:
        unc = (poi_p1 - poi_min, poi_min - poi_m1)
    num_min = Number(poi_min, unc)

    return DotDict(
        interp=interp,
        poi_min=poi_min,
        poi_p1=poi_p1,
        poi_m1=poi_m1,
        poi_p2=poi_p2,
        poi_m2=poi_m2,
        num_min=num_min,
    )


def evaluate_likelihood_scan_2d(poi1_values, poi2_values, dnll2_values, poi1_min=None,
        poi2_min=None):
    """
    Takes the results of a 2D likelihood profiling scan given by *poi1_values*, *poi2_values* and
    the corresponding *dnll2_values* values, performs an interpolation and returns certain results
    of the scan in a dict. The two lists of poi values should represent an expanded grid, so that
    *poi1_values*, *poi2_values* and *dnll2_values* should all be 1D with the same length. When
    *poi1_min* and *poi2_min* are *None*, they are estimated from the interpolated curve.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``poi1_min``: The poi1 value corresponding to the minimum delta nll value.
    - ``poi2_min``: The poi2 value corresponding to the minimum delta nll value.
    - ``poi1_p1``: The poi1 value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_m1``: The poi1 value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_p2``: The poi1 value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_m2``: The poi1 value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_p1``: The poi2 value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_m1``: The poi2 value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_p2``: The poi2 value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_m2``: The poi2 value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``num1_min``: A Number instance representing the poi1 minimum and its 1 sigma uncertainty.
    - ``num2_min``: A Number instance representing the poi2 minimum and its 1 sigma uncertainty.
    """
    # first, obtain an interpolation function and sample new points
    mask = ~np.isnan(dnll2_values)
    interp = scipy.interpolate.interp2d(poi1_values[mask], poi2_values[mask], dnll2_values[mask],
        kind="cubic")

    # get the minima
    if poi1_min is None or poi2_min is None:
        objective = lambda x: abs(interp(*x))
        bounds1 = (poi1_values.min() + 1e-4, poi1_values.max() - 1e-4)
        bounds2 = (poi2_values.min() + 1e-4, poi2_values.max() - 1e-4)
        res = scipy.optimize.minimize(objective, [1., 1.], tol=1e-7, bounds=[bounds1, bounds2])
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi1_min = res.x[0]
        poi2_min = res.x[1]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v, poi_values, poi1=None, poi2=None):
        def minimize(start):
            if poi1 is None:
                objective = lambda x: abs(interp(x, poi2) - v)
                bounds = (poi1_values.min() + 1e-4, poi1_values.max() - 1e-4)
            elif poi2 is None:
                objective = lambda x: abs(interp(poi1, x) - v)
                bounds = (poi2_values.min() + 1e-4, poi2_values.max() - 1e-4)
            else:
                raise ValueError("either poi1 or poi2 must be set")
            res = scipy.optimize.minimize(objective, start, tol=1e-3, bounds=[bounds])
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        return minimize(poi_values.max() - 1), minimize(poi_values.min() + 1)

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi1_p1, poi1_m1 = get_intersections(chi2_levels[2][1], poi1_values, poi2=poi2_min)
    poi2_p1, poi2_m1 = get_intersections(chi2_levels[2][1], poi2_values, poi1=poi1_min)
    poi1_p2, poi1_m2 = get_intersections(chi2_levels[2][2], poi1_values, poi2=poi2_min)
    poi2_p2, poi2_m2 = get_intersections(chi2_levels[2][2], poi2_values, poi1=poi1_min)

    # create Number objects wrapping the best fit values and their 1 sigma error when given
    unc1 = None
    unc2 = None
    if poi1_p1 is not None and poi1_m1 is not None:
        unc1 = (poi1_p1 - poi1_min, poi1_min - poi1_m1)
    if poi2_p1 is not None and poi2_m1 is not None:
        unc2 = (poi2_p1 - poi2_min, poi2_min - poi2_m1)
    num1_min = Number(poi1_min, unc1)
    num2_min = Number(poi2_min, unc2)

    return DotDict(
        interp=interp,
        poi1_min=poi1_min,
        poi2_min=poi2_min,
        poi1_p1=poi1_p1,
        poi1_m1=poi1_m1,
        poi1_p2=poi1_p2,
        poi1_m2=poi1_m2,
        poi2_p1=poi2_p1,
        poi2_m1=poi2_m1,
        poi2_p2=poi2_p2,
        poi2_m2=poi2_m2,
        num1_min=num1_min,
        num2_min=num2_min,
    )
