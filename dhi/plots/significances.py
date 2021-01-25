# coding: utf-8

"""
Significance plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors, color_sequence
from dhi.util import import_ROOT, to_root_latex, create_tgraph, try_int
from dhi.plots.styles import use_style


colors = colors.root


@use_style("dhi_default")
def plot_significance_scan(
    path,
    poi,
    scan_parameter,
    expected_values,
    observed_values=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot for the significance scan of a *poi* over a *scan_parameter* and saves it at
    *path*. *expected_values* should be a mapping to lists of values or a record array with keys
    "<poi_name>" and "significance". When *observed_values* is given, it should be single lists of
    values with the same length as the lists given in *expected_values*.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/cms-hh/tools/inference/tasks/significances.html#plotsignificancescan
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record array to dict mapping to arrays
    if isinstance(expected_values, np.ndarray):
        expected_values = {key: expected_values[key] for key in expected_values.dtype.names}

    # input checks
    assert(scan_parameter in expected_values)
    assert("significance" in expected_values)
    scan_values = expected_values[scan_parameter]
    n_points = len(scan_values)
    assert(all(len(d) == n_points for d in expected_values.values()))
    if observed_values is not None:
        assert(len(observed_values) == n_points)
        observed_values = np.array(observed_values)

    # set default ranges
    if x_min is None:
        x_min = min(scan_values)
    if x_max is None:
        x_max = max(scan_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas()
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "Significance ({}) over background-only / #sigma".format(poi_data[poi].label)
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # expected values
    g_exp = create_tgraph(n_points, scan_values, expected_values["significance"])
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
        "MarkerSize": 0.7})
    draw_objs.append((g_exp, "SAME,PL"))
    legend_entries.append((g_exp, "Expected"))
    y_max_value = max(y_max_value, max(expected_values["significance"]))
    y_min_value = min(y_min_value, min(expected_values["significance"]))

    # observed values
    if observed_values is not None:
        g_obs = create_tgraph(n_points, scan_values, observed_values)
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0.7}, color=colors.red)
        draw_objs.append((g_obs, "SAME,PL"))
        legend_entries.append((g_obs, "Observed"))
        y_max_value = max(y_max_value, max(observed_values))
        y_min_value = min(y_min_value, min(observed_values))

    # set limits
    if y_min is None:
        y_min = 0.
    if y_max is None:
        y_max = 1.35 * (y_max_value - y_min)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # horizontal lines at full integers up to 5
    for y in range(1, 5 + 1):
        if not (y_min < y < y_max):
            continue
        line = ROOT.TLine(x_min, y, x_max, y)
        r.setup_line(line, props={"LineStyle": 7, "NDC": False}, color=colors.red)
        draw_objs.append(line)

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=160, y2=-20, n=len(legend_entries))
    r.setup_legend(legend)
    r.fill_legend(legend, legend_entries)
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


@use_style("dhi_default")
def plot_significance_scans(
    path,
    poi,
    scan_parameter,
    expected_values,
    names,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing multiple significance scans of a *poi* over a *scan_parameter* and saves
    it at *path*. *expected_values* should be a list of mappings to lists of values or a record
    array with keys "<poi_name>" and "significance". Each mapping in *expected_values* will result
    in a different curve. *names* denote the names of significance curves shown in the legend.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/tasks/significances.html
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record arrays to dicts mapping to arrays
    _expected_values = []
    for _ev in expected_values:
        if isinstance(_ev, np.ndarray):
            _ev = {key: _ev[key] for key in _ev.dtype.names}
        _expected_values.append(_ev)
    expected_values = _expected_values

    # input checks
    n_graphs = len(expected_values)
    assert(n_graphs >= 1)
    assert(len(names) == n_graphs)
    assert(all(scan_parameter in ev for ev in expected_values))
    assert(all("significance" in ev for ev in expected_values))
    scan_values = expected_values[0][scan_parameter]
    n_points = len(scan_values)
    assert(all(len(ev["significance"]) == n_points for ev in expected_values))

    # set default ranges
    if x_min is None:
        x_min = min(scan_values)
    if x_max is None:
        x_max = max(scan_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas()
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "Significance ({}) over background-only / #sigma".format(poi_data[poi].label)
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # expected values
    for i, (ev, col) in enumerate(zip(expected_values[::-1], color_sequence[:n_graphs][::-1])):
        g_exp = create_tgraph(n_points, scan_values, ev["significance"])
        r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0.7}, color=col)
        draw_objs.append((g_exp, "SAME,PL"))
        legend_entries.append((g_exp, names[n_graphs - i - 1]))
        y_max_value = max(y_max_value, max(ev["significance"]))
        y_min_value = min(y_min_value, min(ev["significance"]))

    # set limits
    if y_min is None:
        y_min = 0.
    if y_max is None:
        y_max = 1.35 * (y_max_value - y_min)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend_cols = int(math.ceil(len(legend_entries) / 4.))
    legend_rows = min(len(legend_entries), 4)
    legend = r.routines.create_legend(pad=pad, y2=-20, width=legend_cols * 160, n=legend_rows,
        props={"NColumns": legend_cols})
    r.setup_legend(legend)
    r.fill_legend(legend, legend_entries)
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
