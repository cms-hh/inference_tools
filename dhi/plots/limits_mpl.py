# coding: utf-8

"""
Limit plots using matplotlib.
"""

import numpy as np

from dhi.config import poi_data, campaign_labels
from dhi.util import import_plt

plt = import_plt()


def plot_limit_scan(
    path,
    poi,
    data,
    injected_values=None,
    theory_values=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    is_xsec=False,
    campaign="2017",
):
    """
    Creates a plot for the upper limit scan of a *poi* and saves it at *path*. *data* should be a
    mapping to lists of values or a record array with keys "<poi_name>" and "limit", and optionally
    "limit_p1" (plus 1 sigma), "limit_m1" (minus 1 sigma), "limit_p2" and "limit_m2". When the
    variations by 1 or 2 sigma are missing, the plot is created without them. When *injected_values*
    or *theory_values* are given, they should be single lists of values. Therefore, they must have
    the same length as the lists given in *data*. When *y_log* is *True*, the y-axis is plotted with
    a logarithmic scale. *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to
    the range of the given values. *is_xsec* denotes whether the passed values are given as real
    cross sections or, when *False*, as a ratio over the theory prediction. *campaign* should refer
    to the name of a campaign label defined in dhi.config.campaign_labels.

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/mpl/?search=limits
    """
    # convert record array to dict
    if isinstance(data, np.ndarray):
        data = {key: data[key] for key in data.dtype.names}

    # input checks
    assert poi in data
    poi_values = data[poi]
    n_points = len(poi_values)
    assert "limit" in data
    assert all(len(d) == n_points for d in data.values())
    if injected_values:
        assert len(injected_values) == n_points
    if theory_values:
        assert len(theory_values) == n_points

    # set default ranges
    if x_min is None:
        x_min = poi_values.min()
    if x_max is None:
        x_max = poi_values.max()

    # start plotting
    fig, ax = plt.subplots()
    legend_handles = []

    # central limit
    p = ax.plot(
        poi_values,
        data["limit"],
        label="Expected limit",
        color="black",
        linestyle="dashed",
    )
    legend_handles.append(p[0])

    # 2 sigma band
    if "limit_p2" in data and "limit_m2" in data:
        p = ax.fill_between(
            poi_values,
            data["limit_p2"],
            data["limit_m2"],
            color="yellow",
            label=r"$\pm 95\%$ expected",
            interpolate=True,
        )
        legend_handles.append(p)

    # 1 sigma band
    if "limit_p1" in data and "limit_m1" in data:
        p = ax.fill_between(
            poi_values,
            data["limit_p1"],
            data["limit_m1"],
            color="limegreen",
            label=r"$\pm 68\%$ expected",
            interpolate=True,
        )
        legend_handles.insert(1, p)

    # injected limits
    if injected_values:
        p = ax.plot(
            poi_values,
            injected_values,
            label=r"Injected limits",
            color="black",
            linestyle="-",
        )
        legend_handles.append(p[0])

    # theory prediction
    if theory_values:
        p = ax.plot(
            poi_values,
            theory_values,
            label=r"Theory prediction",
            color="red",
            linestyle="-",
        )
        legend_handles.append(p[0])

    # legend, labels, titles, etc
    ax.set_xlabel(poi_data[poi].label_math)
    y_unit = "fb" if is_xsec else r"$\sigma_{SM}$"
    ax.set_ylabel(r"Upper $95\%$ CLs limit on $\sigma$ / " + y_unit)
    if y_log:
        ax.set_yscale("log")
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.set_xlim(left=x_min, right=x_max)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    ax.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")
    ax.set_title(r"\textbf{CMS} \textit{preliminary}", loc="left")
    ax.set_title(campaign_labels.get(campaign, campaign), loc="right")
    ax.grid()

    # save
    fig.tight_layout()
    fig.savefig(path)
