# coding: utf-8

"""
Limit plots using matplotlib.
"""

import numpy as np

from dhi.config import poi_labels_math, campaign_labels
from dhi.util import import_plt

plt = import_plt()


def plot_limit_scan(
    path,
    poi,
    data,
    injected_values=None,
    theory_values=None,
    log=False,
    is_xsec=True,
    campaign="2017",
):
    """
    Creates a plot for the upper limit scan of a *poi* and saves it at *path*. *data* should be a
    mapping to lists of values or a record array with keys "<poi_name>" and "limit", and optionally
    "limit_p1" (plus 1 sigma), "limit_m1" (minus 1 sigma), "limit_p2" and "limit_m2". When the
    variations by 1 or 2 sigma are missing, the plot is created without them. When *injected_values*
    or *theory_values* are given, they should be single lists of values. Therefore, they must have
    the same length as the lists given in *data*. When *log* is *True*, the y axis is plotted with a
    logarithmic scale. *is_xsec* denotes whether the passed values are given as real cross sections
    or, when *False*, as a ratio over the theory prediction. *campaign* should refer to the name of
    a campaign label defined in dhi.config.campaign_labels.

    Examples: http://mrieger.web.cern.ch/mrieger/dhi/examples/?search=limits
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

    # start plotting
    fig, ax = plt.subplots()

    # central limit
    ax.plot(
        poi_values,
        data["limit"],
        label="Expected limit",
        color="black",
        linestyle="dashed",
    )

    # 2 sigma band
    if "limit_p2" in data and "limit_m2" in data:
        ax.fill_between(
            poi_values,
            data["limit_p2"],
            data["limit_m2"],
            color="yellow",
            label=r"$\pm 95\%$ expected",
            interpolate=True,
        )

    # 1 sigma band
    if "limit_p1" in data and "limit_m1" in data:
        ax.fill_between(
            poi_values,
            data["limit_p1"],
            data["limit_m1"],
            color="limegreen",
            label=r"$\pm 68\%$ expected",
            interpolate=True,
        )

    # injected limits
    if injected_values:
        ax.plot(
            poi_values,
            injected_values,
            label=r"Injected limits",
            color="black",
            linestyle="-",
        )

    # theory prediction
    if theory_values:
        ax.plot(
            poi_values,
            theory_values,
            label=r"Theory prediction",
            color="red",
            linestyle="-",
        )

    # legend, labels, titles, etc
    ax.set_xlabel(poi_labels_math.get(poi, poi))
    y_unit = "fb" if is_xsec else r"$\sigma_{SM}$"
    ax.set_ylabel(r"Upper $95\%$ CLs limit on $\sigma$ / " + y_unit)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    ax.legend(loc="best")
    ax.set_title(r"\textbf{CMS} \textit{preliminary}", loc="left")
    ax.set_title(campaign_labels.get(campaign, campaign), loc="right")
    if log:
        ax.set_yscale("log")

    # save
    fig.tight_layout()
    fig.savefig(path)
