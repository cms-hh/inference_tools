# coding: utf-8

"""
Significance plots using ROOT.
"""

import math

import numpy as np

from dhi.config import (
    poi_data, br_hh_names, campaign_labels, colors, color_sequence, marker_sequence,
)
from dhi.util import (
    import_ROOT, to_root_latex, create_tgraph, make_list, unique_recarray, dict_to_recarray,
    try_int,
)
from dhi.plots.util import (
    use_style, create_model_parameters, get_y_range, infer_binning_from_grid, get_contours,
    fill_hist_from_points, get_text_extent, locate_contour_labels,
)


colors = colors.root


@use_style("dhi_default")
def plot_significance_scan_1d(
    path,
    scan_parameter,
    expected_values=None,
    observed_values=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_points=False,
):
    """
    Creates a plot for the significance scan over a *scan_parameter* and saves it at *path*.
    *expected_values* should be a mapping to lists of values or a record array with keys
    "<scan_parameter>" and "significance". When *observed_values* is given, it should have the same
    structure as *expected_values*.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/significances.html#significance-vs-scan-parameter
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # helper to check and convert record arrays to dict mappings to arrays
    def check_values(values):
        if isinstance(values, np.ndarray):
            values = {key: values[key] for key in values.dtype.names}
        values = {k: np.array(v) for k, v in values.items()}
        # check fields
        assert(scan_parameter in values)
        assert("significance" in values)
        # remove nans
        mask = ~np.isnan(values["significance"])
        values = {k: v[mask] for k, v in values.items()}
        n_nans = (~mask).sum()
        if n_nans:
            print("WARNING: found {} NaN(s) in significance values".format(n_nans))
        return values

    # input checks
    if expected_values is None and observed_values is None:
        raise Exception("either expected_values or observed_values must be set")

    scan_values = None
    if expected_values is not None:
        expected_values = check_values(expected_values)
        scan_values_exp = expected_values[scan_parameter]
        n_points_exp = len(scan_values_exp)
        scan_values = scan_values_exp
    if observed_values is not None:
        observed_values = check_values(observed_values)
        scan_values_obs = observed_values[scan_parameter]
        n_points_obs = len(scan_values_obs)
        scan_values = scan_values_obs if scan_values is None else scan_values

    # set default ranges
    if x_min is None:
        x_min = min(scan_values)
    if x_max is None:
        x_max = max(scan_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_min_value = 1e5
    y_max_value = -1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "Significance / #sigma"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # expected values
    if expected_values is not None:
        g_exp = create_tgraph(n_points_exp, scan_values_exp, expected_values["significance"])
        r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0.7})
        draw_objs.append((g_exp, "SAME,C" + ("P" if show_points else "")))
        legend_entries.append((g_exp, "Expected", "L"))
        y_min_value = min(y_min_value, min(expected_values["significance"]))
        y_max_value = max(y_max_value, max(expected_values["significance"]))

    # observed values
    if observed_values is not None:
        g_obs = create_tgraph(n_points_obs, scan_values_obs, observed_values["significance"])
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0.7}, color=colors.red)
        draw_objs.append((g_obs, "SAME,PL"))
        legend_entries.append((g_obs, "Observed", "PL"))
        y_max_value = max(y_max_value, max(observed_values["significance"]))
        y_min_value = min(y_min_value, min(observed_values["significance"]))

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value if y_log else 0, y_max_value, y_min, y_max, log=y_log)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # horizontal lines at full integers up to 5
    for y in range(1, max(5, int(math.floor(y_max_value))) + 1):
        if not (y_min < y < y_max):
            continue
        line = ROOT.TLine(x_min, y, x_max, y)
        r.setup_line(line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
        draw_objs.insert(1, line)

    # legend
    legend = r.routines.create_legend(pad=pad, width=160, y2=-20, n=len(legend_entries))
    r.setup_legend(legend)
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        draw_objs.extend(create_model_parameters(model_parameters, pad))

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
def plot_significance_scans_1d(
    path,
    scan_parameter,
    values,
    names,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_points=True,
):
    """
    Creates a plot showing multiple significance scans over a *scan_parameter* and saves it at
    *path*. *values* should be a list of mappings to lists of values or a record array with keys
    "<scan_parameter>" and "significance". Each mapping in *values* will result in a different
    curve. *names* denote the names of significance curves shown in the legend.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/significances.html#multiple-significance-scans-vs-scan-parameter
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record arrays to dicts mapping to arrays
    _values = []
    for i, _vals in enumerate(values):
        if isinstance(_vals, np.ndarray):
            _vals = {key: _vals[key] for key in _vals.dtype.names}
        mask = ~np.isnan(_vals["significance"])
        _vals = {k: np.array(v)[mask] for k, v in _vals.items()}
        _values.append(_vals)
        n_nans = (~mask).sum()
        if n_nans:
            print("WARNING: found {} NaN(s) in significance values at index {}".format(n_nans, i))
    values = _values

    # input checks
    n_graphs = len(values)
    assert(n_graphs >= 1)
    assert(len(names) == n_graphs)
    assert(all(scan_parameter in vals for vals in values))
    assert(all("significance" in vals for vals in values))
    scan_values = values[0][scan_parameter]

    # set default ranges
    if x_min is None:
        x_min = min(scan_values)
    if x_max is None:
        x_max = max(scan_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "Significance / #sigma"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # expected values
    for i, (vals, col, ms) in enumerate(zip(values[::-1], color_sequence[:n_graphs][::-1],
            marker_sequence[:n_graphs][::-1])):
        g_exp = create_tgraph(int(len(scan_values)), scan_values, vals["significance"])
        r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": ms,
            "MarkerSize": 1.2}, color=colors[col])
        draw_objs.append((g_exp, "SAME,CP" if show_points else "SAME,C"))
        name = names[n_graphs - i - 1]
        legend_entries.insert(0, (g_exp, to_root_latex(br_hh_names.get(name, name)),
            "LP" if show_points else "L"))
        y_max_value = max(y_max_value, max(vals["significance"]))
        y_min_value = min(y_min_value, min(vals["significance"]))

    # set limits
    y_min, y_max, _ = get_y_range(0 if y_log else y_min_value, y_max_value, y_min, y_max, log=y_log)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # horizontal lines at full integers up to 5
    for y in range(1, max(5, int(math.floor(y_max_value))) + 1):
        if not (y_min < y < y_max):
            continue
        line = ROOT.TLine(x_min, y, x_max, y)
        r.setup_line(line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
        draw_objs.insert(1, line)

    # legend
    legend_cols = int(math.ceil(len(legend_entries) / 4.))
    legend_rows = min(len(legend_entries), 4)
    legend = r.routines.create_legend(pad=pad, y2=-20, width=legend_cols * 160, n=legend_rows,
        props={"NColumns": legend_cols, "TextSize": 18})
    r.setup_legend(legend)
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        draw_objs.extend(create_model_parameters(model_parameters, pad))

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
