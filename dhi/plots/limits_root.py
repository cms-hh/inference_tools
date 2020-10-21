# coding: utf-8

"""
Limit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors
from dhi.util import import_ROOT, to_root_latex


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

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/plotting.html#upper-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record array to dict
    if isinstance(data, np.ndarray):
        data = {key: data[key] for key in data.dtype.names}

    # input checks
    assert poi in data
    poi_values = data[poi]
    n_points = len(poi_values)
    assert "limit" in data
    assert all(len(d) == n_points for d in data.values())
    if injected_values is not None:
        assert len(injected_values) == n_points
    if theory_values is not None:
        assert len(theory_values) == n_points

    # set default ranges
    if x_min is None:
        x_min = min(poi_values)
    if x_max is None:
        x_max = max(poi_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi].label)
    y_unit = "fb" if is_xsec else "#sigma_{SM}"
    y_title = "Upper 95% CLs limit on #sigma / " + y_unit
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # helper to read data into graphs
    def create_graph(sigma=None, values=None):
        # repeat the edges by padding to prevent bouncing effects of interpolated lines
        pad = lambda arr: np.pad(np.array(arr, dtype=np.float32), 1, mode="edge")
        arr = lambda key: pad(np.array(data[key], dtype=np.float32))
        zeros = np.zeros(n_points + 2, dtype=np.float32)
        return ROOT.TGraphAsymmErrors(
            n_points + 2,
            pad(poi_values),
            pad(values) if values is not None else arr("limit"),
            zeros,
            zeros,
            (arr("limit") - arr("limit_m{}".format(sigma))) if sigma else zeros,
            (arr("limit_p{}".format(sigma)) - arr("limit")) if sigma else zeros,
        )

    # 2 sigma band
    if "limit_p2" in data and "limit_m2" in data:
        g_2sigma = create_graph(sigma=2)
        r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
            "MarkerSize": 0, "FillColor": colors.root.yellow})
        draw_objs.append((g_2sigma, "SAME,4"))
        legend_entries.insert(0, (g_2sigma, r"#pm 95% expected"))
        y_max_value = max(y_max_value, max(data["limit_p2"]))
        y_min_value = min(y_min_value, min(data["limit_m2"]))

    # 1 sigma band
    if "limit_p1" in data and "limit_m1" in data:
        g_1sigma = create_graph(sigma=1)
        r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
            "MarkerSize": 0, "FillColor": colors.root.green})
        draw_objs.append((g_1sigma, "SAME,4"))
        legend_entries.insert(0, (g_1sigma, r"#pm 68% expected"))
        y_max_value = max(y_max_value, max(data["limit_p1"]))
        y_min_value = min(y_min_value, min(data["limit_m1"]))

    # central values
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
        "MarkerSize": 0})
    draw_objs.append((g_exp, "SAME,LEZ"))
    legend_entries.insert(0, (g_exp, "Expected limit"))
    y_max_value = max(y_max_value, max(data["limit"]))
    y_min_value = min(y_min_value, min(data["limit"]))

    # injected values
    if injected_values is not None:
        g_inj = create_graph(values=injected_values)
        r.setup_graph(g_inj, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0})
        draw_objs.append((g_inj, "SAME,L"))
        legend_entries.append((g_inj, "Injected limit"))

    # theory prediction
    if theory_values is not None:
        g_thy = create_graph(values=theory_values)
        r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0, "LineColor": colors.root.red})
        draw_objs.append((g_thy, "SAME,L"))
        legend_entries.append((g_thy, "Theory prediction"))

    # set limits
    if y_min is None:
        if y_log:
            y_min = 0.75 * y_min_value
        else:
            y_min = 0.
    if y_max is None:
        if y_log:
            y_max = y_min * 10**(1.35 * math.log10(y_max_value / y_min))
        else:
            y_max = 1.35 * (y_max_value - y_min)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # legend
    legend = r.routines.create_legend(pad=pad, width=230, height=len(legend_entries) * 35)
    r.setup_legend(legend)
    for obj, label in legend_entries:
        legend.AddEntry(obj, label)
    draw_objs.append(legend)

    # cms label
    cms_labels = r.routines.create_cms_labels(pad=pad)
    draw_objs.extend(cms_labels)

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    canvas.SaveAs(path)
