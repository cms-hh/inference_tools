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
    expected_values,
    observed_values=None,
    theory_values=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    xsec_unit=None,
    pp_process="pp",
    hh_process="HH",
    campaign="2017",
):
    """
    Creates a plot for the upper limit scan of a *poi* and saves it at *path*. *expected_values*
    should be a mapping to lists of values or a record array with keys "<poi_name>" and "limit", and
    optionally "limit_p1" (plus 1 sigma), "limit_m1" (minus 1 sigma), "limit_p2" and "limit_m2".
    When the variations by 1 or 2 sigma are missing, the plot is created without them. When
    *observed_values* or *theory_values* are given, they should be single lists of values.
    Therefore, they must have the same length as the lists given in *expected_values*.

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *x_min*, *x_max*,
    *y_min* and *y_max* define the axis ranges and default to the range of the given values.
    *xsec_unit* denotes whether the passed values are given as real cross sections in this unit or,
    when *None*, as a ratio over the theory prediction. The *pp_process* label is shown in the
    x-axis title to denote the physics process the computed values are corresponding to.
    *hh_process* is inserted to the process name in the title of the y-axis and indicates that the
    plotted cross section data was (e.g.) scaled by a branching ratio. *campaign* should refer to
    the name of a campaign label defined in dhi.config.campaign_labels.

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/plotting.html#upper-limits
    """
    # convert record array to dict
    if isinstance(expected_values, np.ndarray):
        expected_values = {key: expected_values[key] for key in expected_values.dtype.names}

    # input checks
    assert poi in expected_values
    poi_values = expected_values[poi]
    n_points = len(poi_values)
    assert "limit" in expected_values
    assert all(len(d) == n_points for d in expected_values.values())
    if observed_values is not None:
        assert len(observed_values) == n_points
        observed_values = np.array(observed_values)
    if theory_values is not None:
        assert len(theory_values) == n_points
        theory_values = np.array(theory_values)

    # set default ranges
    if x_min is None:
        x_min = min(poi_values)
    if x_max is None:
        x_max = max(poi_values)

    # start plotting
    fig, ax = plt.subplots()
    legend_handles = []

    # central limit
    p = ax.plot(
        poi_values,
        expected_values["limit"],
        label="Median expected",
        color="black",
        linestyle="dashed",
    )
    legend_handles.append(p[0])

    # 2 sigma band
    if "limit_p2" in expected_values and "limit_m2" in expected_values:
        p = ax.fill_between(
            poi_values,
            expected_values["limit_p2"],
            expected_values["limit_m2"],
            color="yellow",
            label=r"$\pm$ 95\% expected",
            interpolate=True,
        )
        legend_handles.append(p)

    # 1 sigma band
    if "limit_p1" in expected_values and "limit_m1" in expected_values:
        p = ax.fill_between(
            poi_values,
            expected_values["limit_p1"],
            expected_values["limit_m1"],
            color="limegreen",
            label=r"$\pm$ 68\% expected",
            interpolate=True,
        )
        legend_handles.insert(1, p)

    # observed limits
    if observed_values is not None:
        p = ax.plot(
            poi_values,
            observed_values,
            label=r"Observed",
            color="black",
            linestyle="-",
        )
        legend_handles.insert(0, p[0])

    # theory prediction
    if theory_values is not None:
        p = ax.plot(
            poi_values,
            theory_values,
            label=r"SM prediction",
            color="red",
            linestyle="-",
        )
        legend_handles.append(p[0])

    # legend, labels, titles, etc
    ax.set_xlabel(poi_data[poi].label_math)
    ax.set_ylabel(r"Upper 95\% CLs limit on $\sigma$ ({} $\rightarrow$ {}) / {}".format(
        pp_process, hh_process, xsec_unit or r"$\sigma_{SM}$"))
    if y_log:
        ax.set_yscale("log")
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.set_xlim(left=x_min, right=x_max)
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction="in")
    ax.legend(legend_handles, [h.get_label() for h in legend_handles], loc="best")
    ax.set_title(r"\textbf{CMS} \textit{Preliminary}", loc="left")
    if campaign:
        ax.set_title(campaign_labels.get(campaign, campaign), loc="right")
    ax.grid()

    # save
    fig.tight_layout()
    fig.savefig(path)
