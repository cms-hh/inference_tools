# coding: utf-8

"""
Likelihood plots using matplotlib.
"""

import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib
from scinum import Number

from dhi.config import poi_data, campaign_labels
from dhi.util import import_plt, DotDict, rgb, minimize_1d, get_neighbor_coordinates


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

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/mpl/?search=nll1d
    """
    # get valid poi and delta nll values
    poi_values = data[poi]
    dnll2_values = data["dnll2"]

    # set default range
    if x_min is None:
        x_min = poi_values.min()
    if x_max is None:
        x_max = poi_values.max()

    # select valid points
    mask = ~np.isnan(dnll2_values)
    poi_values = poi_values[mask]
    dnll2_values = dnll2_values[mask]

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

    # lines at chi2_1 intervals
    for n in [chi2_levels[1][1], chi2_levels[1][2]]:
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
        poi_data[poi].label,
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
    ax.set_xlabel(poi_data[poi].label_math)
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
        x1_min=None, x1_max=None, x2_min=None, x2_max=None, fill_nans=True):
    """
    Creates a likelihood plot of the 2D scan of two pois *poi1* and *poi2*, and saves it at *path*.
    *data* should be a mapping to lists of values or a record array with keys "<poi1_name>",
    "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be the values
    of the pois that lead to the best likelihood. Otherwise, they are  estimated from the
    interpolated curve. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels. *x1_min*, *x1_max*, *x2_min* and *x2_max* define the axis range of
    poi1 and poi2, respectively, and default to the ranges of the poi values. When *fill_nans* is
    *True*, points with failed fits, denoted by nan values, are filled with the averages of
    neighboring fits.

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/mpl/?search=nll2d
    """
    # get poi and delta nll values
    poi1_values = data[poi1]
    poi2_values = data[poi2]
    dnll2_values = data["dnll2"]

    # set default ranges
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

    # transform the poi coordinates and dnll2 values into a 2d array
    # where the inner (outer) dimension refers to poi1 (poi2)
    e1 = np.unique(poi1_values)
    e2 = np.unique(poi2_values)
    i1 = np.searchsorted(e1, poi1_values, side="right") - 1
    i2 = np.searchsorted(e2, poi2_values, side="right") - 1
    data = np.zeros((e2.size, e1.size), dtype=np.float32)
    data[i2, i1] = dnll2_values

    # set slightly negative numbers (usually in the order of e-5) to a new minimal value
    # (not needed for the moment)
    # cur_min = data[data > 0].min()
    # data[(data < 0) & (data > -1e-5)] = cur_min * 0.9

    # optionally fill nans with averages over neighboring points
    if fill_nans:
        nans = np.argwhere(np.isnan(data))
        npoints = {tuple(p): get_neighbor_coordinates(data.shape, *p) for p in nans}
        nvals = {p: data[[c[0] for c in cs], [c[1] for c in cs]] for p, cs in npoints.items()}
        nmeans = {p: vals[~np.isnan(vals)].mean() for p, vals in nvals.items()}
        data[[p[0] for p in nmeans], [p[1] for p in nmeans]] = nmeans.values()

    # start plotting
    fig, ax = plt.subplots()

    img = ax.imshow(
        data,
        norm=matplotlib.colors.LogNorm(),
        aspect="auto",
        origin="lower",
        extent=[poi1_values.min(), poi1_values.max(), poi2_values.min(), poi2_values.max()],
        cmap="viridis",
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, label=r"$-2 \Delta \text{ln}\mathcal{L}$")

    # contours
    contours = plt.contour(e1, e2, data, levels=[chi2_levels[2][1], chi2_levels[2][2]])
    fmt = {l: s for l, s in zip(contours.levels, [r"$1 \sigma$", r"$2 \sigma$"])}
    plt.clabel(contours, inline=True, fontsize=8, fmt=fmt)

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
            poi_data[poi].label,
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
    ax.set_xlabel(poi_data[poi1].label_math)
    ax.set_ylabel(poi_data[poi2].label_math)
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

    # store sane extrema
    poi_values_min = poi_values[mask].min()
    poi_values_max = poi_values[mask].max()

    # get the minimum when not set
    if poi_min is None:
        objective = lambda x: abs(interp(x))
        bounds = (poi_values_min + 1e-4, poi_values_max - 1e-4)
        res = minimize_1d(objective, bounds)
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi_min = res.x[0]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v):
        def minimize(bounds):
            objective = lambda x: (interp(x) - v)**2.
            res = minimize_1d(objective, bounds)
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        return (
            minimize((poi_min, poi_values_max - 1e-4)),
            minimize((poi_values_min + 1e-4, poi_min)),
        )

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

    # store sane extrema
    poi1_values_min = poi1_values[mask].min()
    poi1_values_max = poi1_values[mask].max()
    poi2_values_min = poi2_values[mask].min()
    poi2_values_max = poi2_values[mask].max()

    # get the minima
    if poi1_min is None or poi2_min is None:
        objective = lambda x: interp(*x)**2.
        bounds1 = (poi1_values_min + 1e-4, poi1_values_max - 1e-4)
        bounds2 = (poi2_values_min + 1e-4, poi2_values_max - 1e-4)
        res = scipy.optimize.minimize(objective, [1., 1.], tol=1e-7, bounds=[bounds1, bounds2])
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi1_min = res.x[0]
        poi2_min = res.x[1]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v, n_poi):
        def minimize(bounds):
            res = minimize_1d(objective, bounds)
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        if n_poi == 1:
            poi_values_min, poi_values_max = poi1_values_min, poi1_values_max
            poi_min = poi1_min
            objective = lambda x: (interp(x, poi2_min) - v)**2.
        else:
            poi_values_min, poi_values_max = poi2_values_min, poi2_values_max
            poi_min = poi2_min
            objective = lambda x: (interp(poi1_min, x) - v)**2.

        return (
            minimize((poi_min, poi_values_max - 1e-4)),
            minimize((poi_values_min + 1e-4, poi_min)),
        )

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi1_p1, poi1_m1 = get_intersections(chi2_levels[2][1], 1)
    poi2_p1, poi2_m1 = get_intersections(chi2_levels[2][1], 2)
    poi1_p2, poi1_m2 = get_intersections(chi2_levels[2][2], 1)
    poi2_p2, poi2_m2 = get_intersections(chi2_levels[2][2], 2)

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
