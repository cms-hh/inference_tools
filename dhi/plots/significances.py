# coding: utf-8

"""
Significance plots using ROOT.
"""

import math

import numpy as np
import scipy as sp
import scipy.stats

from dhi.config import (
    poi_data, br_hh_names, br_hh_colors, campaign_labels, colors, color_sequence, marker_sequence,
)
from dhi.util import (
    import_ROOT, to_root_latex, create_tgraph, make_list, unique_recarray, dict_to_recarray,
    try_int,
)
from dhi.plots.util import (
    use_style, create_model_parameters, get_y_range, infer_binning_from_grid, get_contours,
    fill_hist_from_points, get_text_extent, locate_contour_labels, Style,
)


colors = colors.root


@use_style("dhi_default")
def plot_significance_scan_1d(
    paths,
    scan_parameter,
    expected_values=None,
    observed_values=None,
    show_p_values=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_points=False,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot for the significance scan over a *scan_parameter* and saves it at *paths*.
    *expected_values* should be a mapping to lists of values or a record array with keys
    "<scan_parameter>" and "significance". When *observed_values* is given, it should have the same
    structure as *expected_values*. When *show_p_values* is *True*, p-values are obtained from
    significances and shown instead.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

    - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/significances.html#significance-vs-scan-parameter  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # helper to check and convert record arrays to dict mappings to arrays
    def check_values(values):
        if isinstance(values, np.ndarray):
            values = {key: values[key] for key in values.dtype.names}
        values = {k: np.array(v) for k, v in values.items()}
        # check fields
        assert scan_parameter in values
        assert "significance" in values
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
    y_title = "p-value" if show_p_values else "Significance / #sigma"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # expected values
    if expected_values is not None:
        y_values = expected_values["significance"]
        if show_p_values:
            y_values = sp.stats.norm.sf(y_values)
        g_exp = create_tgraph(n_points_exp, scan_values_exp, y_values)
        r.setup_graph(
            g_exp,
            props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20, "MarkerSize": 0.7},
        )
        draw_objs.append((g_exp, "SAME,C" + ("P" if show_points else "")))
        legend_entries.append((g_exp, "Expected", "L"))
        y_min_value = min(y_min_value, min(y_values))
        y_max_value = max(y_max_value, max(y_values))

    # observed values
    if observed_values is not None:
        y_values = observed_values["significance"]
        if show_p_values:
            y_values = sp.stats.norm.sf(y_values)
        g_obs = create_tgraph(n_points_obs, scan_values_obs, y_values)
        r.setup_graph(
            g_obs,
            props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20, "MarkerSize": 0.7},
            color=colors.red,
        )
        draw_objs.append((g_obs, "SAME,PL"))
        legend_entries.append((g_obs, "Observed", "PL"))
        y_max_value = max(y_max_value, max(y_values))
        y_min_value = min(y_min_value, min(y_values))

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value if y_log else 0, y_max_value, y_min, y_max, log=y_log)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # horizontal lines at up to 5 integer significances
    sig_line_labels = np.arange(1, 5 + 1)
    sig_line_values = sp.stats.norm.sf(sig_line_labels) if show_p_values else sig_line_labels
    for y, sig in zip(sig_line_values, sig_line_labels):
        if not (y_min < y < y_max):
            continue
        if show_p_values and not y_log:
            continue
        sig_line = ROOT.TLine(x_min, y, x_max, y)
        r.setup_line(sig_line, props={"NDC": False, "LineWidth": 1}, color=colors.grey)
        draw_objs.insert(1, sig_line)

        # extra labels when showing p-values
        if show_p_values:
            # convert y to a value relative to the pad height
            label_y = math.log(y / y_min) / math.log(y_max / y_min)
            label_y *= 1. - pad.GetTopMargin() - pad.GetBottomMargin()
            label_y += pad.GetBottomMargin() + 0.005
            sig_label = r.routines.create_top_left_label(
                "{}#sigma".format(sig),
                pad=pad,
                x_offset=8,
                y=label_y,
                props={"TextSize": 18, "TextColor": colors.grey},
            )
            draw_objs.insert(1, sig_label)

    # legend
    legend = r.routines.create_legend(pad=pad, width=160, y2=-20, n=len(legend_entries))
    r.setup_legend(legend)
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70},
    )
    draw_objs.insert(-1, legend_box)

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(pad=pad, postfix=cms_postfix or "", layout=cms_layout)
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if cms_layout.startswith("inside"):
            y_offset = 100 if cms_layout == "inside_vertical" and cms_postfix else 80
            param_kwargs = {"y_offset": y_offset}
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)


@use_style("dhi_default")
def plot_significance_scans_1d(
    paths,
    scan_parameter,
    values,
    names,
    show_p_values=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_points=True,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing multiple significance scans over a *scan_parameter* and saves it at
    *paths*. *values* should be a list of mappings to lists of values or a record array with keys
    "<scan_parameter>" and "significance". Each mapping in *values* will result in a different
    curve. *names* denote the names of significance curves shown in the legend. When *show_p_values*
    is *True*, p-values are obtained from significances and shown instead.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

    - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/significances.html#multiple-significance-scans-vs-scan-parameter  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

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
    assert n_graphs >= 1
    assert len(names) == n_graphs
    assert all(scan_parameter in vals for vals in values)
    assert all("significance" in vals for vals in values)
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
    y_title = "p-value" if show_p_values else "Significance / #sigma"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(name in br_hh_colors.root for name in names):
        _color_sequence = [br_hh_colors.root[name] for name in names]

    # expected values
    for i, (vals, name, col, ms) in enumerate(zip(
        values, names, _color_sequence[:n_graphs], marker_sequence[:n_graphs]),
    ):
        y_vals = vals["significance"]
        if show_p_values:
            y_vals = sp.stats.norm.sf(y_vals)
        g_exp = create_tgraph(int(len(scan_values)), scan_values, y_vals)
        r.setup_graph(
            g_exp,
            props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": ms, "MarkerSize": 1.2},
            color=colors[col],
        )
        draw_objs.append((g_exp, "SAME,CP" if show_points else "SAME,C"))
        legend_entries.append(
            (g_exp, to_root_latex(br_hh_names.get(name, name)), "LP" if show_points else "L"),
        )
        y_max_value = max(y_max_value, max(y_vals))
        y_min_value = min(y_min_value, min(y_vals))

    # set limits
    y_min, y_max, _ = get_y_range(0 if y_log else y_min_value, y_max_value, y_min, y_max, log=y_log)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # horizontal lines at up to 5 integer significances
    sig_line_labels = np.arange(1, 5 + 1)
    sig_line_values = sp.stats.norm.sf(sig_line_labels) if show_p_values else sig_line_labels
    for y, sig in zip(sig_line_values, sig_line_labels):
        if not (y_min < y < y_max):
            continue
        if show_p_values and not y_log:
            continue
        sig_line = ROOT.TLine(x_min, y, x_max, y)
        r.setup_line(sig_line, props={"NDC": False, "LineWidth": 1}, color=colors.grey)
        draw_objs.insert(1, sig_line)

        # extra labels when showing p-values
        if show_p_values:
            # convert y to a value relative to the pad height
            label_y = math.log(y / y_min) / math.log(y_max / y_min)
            label_y *= 1. - pad.GetTopMargin() - pad.GetBottomMargin()
            label_y += pad.GetBottomMargin() + 0.005
            # from IPython import embed; embed()
            sig_label = r.routines.create_top_left_label(
                "{}#sigma".format(sig),
                pad=pad,
                x_offset=8,
                y=label_y,
                props={"TextSize": 18, "TextColor": colors.grey},
            )
            draw_objs.insert(1, sig_label)

    # legend
    legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
    legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
    legend = r.routines.create_legend(
        pad=pad,
        width=legend_cols * 210,
        n=legend_rows,
        props={"NColumns": legend_cols, "TextSize": 18},
    )
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70},
    )
    draw_objs.insert(-1, legend_box)

    # cms label
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout="outside_horizontal",
    )
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if legend_cols == 3:
            param_kwargs["y_offset"] = 1. - 0.25 * pad.GetTopMargin() - legend.GetY1()
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)


@use_style("dhi_default")
def plot_significance_scan_2d(
    paths,
    scan_parameter1,
    scan_parameter2,
    values,
    show_p_values=False,
    draw_sm_point=True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    z_log=True,
    model_parameters=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a significance plot of the 2D scan of two parameters *scan_parameter1* and
    *scan_parameter2*, and saves it at *paths*. *values* should be a mapping to lists of values or a
    record array with keys "<scan_parameter1_name>", "<scan_parameter2_name>" and "significance".
    When *show_p_values* is *True*, p-values are obtained from significances and shown instead. The
    standard model point at (1, 1) as drawn as well unless *draw_sm_point* is *False*.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *scan_parameter1* and
    *scan_parameter2*, respectively, and default to the ranges of the poi values. *z_min* and
    *z_max* limit the range of the z-axis. When *z_log* is *True*, the z-axis is plotted with a
    logarithmic scale. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

    - "paper"
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check values
    values = make_list(values)
    for i, _values in enumerate(list(values)):
        if isinstance(_values, np.ndarray):
            _values = {key: np.array(_values[key]) for key in _values.dtype.names}
        assert scan_parameter1 in _values
        assert scan_parameter2 in _values
        assert "significance" in _values
        values[i] = _values

    # join values for contour calculation
    joined_values = unique_recarray(dict_to_recarray(values),
        cols=[scan_parameter1, scan_parameter2])

    # determine contours independent of plotting
    contour_levels = [1, 2, 3, 4, 5]  # sigma
    contours = get_contours(
        joined_values[scan_parameter1],
        joined_values[scan_parameter2],
        joined_values["significance"],
        levels=contour_levels,
    )

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"RightMargin": 0.17, "Logz": z_log})
    pad.cd()
    draw_objs = []

    # create a histogram for each scan patch
    hists = []
    for i, _values in enumerate(values):
        _, _, _x_bins, _y_bins, _x_min, _x_max, _y_min, _y_max = infer_binning_from_grid(
            _values[scan_parameter1],
            _values[scan_parameter2],
        )

        # get the z range
        z_vals = np.array(_values["significance"])
        _z_min = 1e-2 if z_log else 0
        _z_max = np.nanmax(z_vals[z_vals < 8.2])
        if show_p_values:
            z_vals = sp.stats.norm.sf(z_vals)
            _z_min, _z_max = sp.stats.norm.sf(_z_max), sp.stats.norm.sf(_z_min)

        # infer axis limits from the first set of values
        if i == 0:
            x_min = _x_min if x_min is None else x_min
            x_max = _x_max if x_max is None else x_max
            y_min = _y_min if y_min is None else y_min
            y_max = _y_max if y_max is None else y_max
            z_min = _z_min if z_min is None else z_min
            z_max = _z_max if z_max is None else z_max

        # fill and store the histogram
        h = ROOT.TH2F("h" + str(i), "", _x_bins, _x_min, _x_max, _y_bins, _y_min, _y_max)
        fill_hist_from_points(
            h,
            _values[scan_parameter1],
            _values[scan_parameter2],
            z_vals,
            z_min=z_min,
            z_max=z_max,
        )
        hists.append(h)

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter1].label)
    y_title = to_root_latex(poi_data[scan_parameter2].label)
    z_title = "p-value" if show_p_values else "Significance / #sigma"
    h_dummy = ROOT.TH2F(
        "h_sig",
        ";{};{};{}".format(x_title, y_title, z_title),
        1,
        x_min,
        x_max,
        1,
        y_min,
        y_max,
    )
    r.setup_hist(h_dummy, pad=pad, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
    draw_objs.append((h_dummy, ""))

    # setup actual histograms
    for i, h in enumerate(hists):
        r.setup_hist(h, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
        if i == 0:
            r.setup_z_axis(
                h.GetZaxis(),
                pad=pad,
                props={"Title": z_title, "TitleSize": 24, "TitleOffset": 1.5},
            )
        draw_objs.append((h, "SAME,COLZ"))
        # for debugging purposes
        # draw_objs.append((h, "SAME,TEXT"))

    # draw contours
    for graphs, level in zip(contours, contour_levels):
        for g in graphs:
            r.setup_graph(g, props={"LineWidth": 1, "LineColor": colors.black})
            draw_objs.append((g, "SAME,C"))

    # draw contour labels at automatic positions
    pad_width = canvas.GetWindowWidth() * (1. - pad.GetLeftMargin() - pad.GetRightMargin())
    pad_height = canvas.GetWindowHeight() * (1. - pad.GetTopMargin() - pad.GetBottomMargin())
    px_to_x = (x_max - x_min) / pad_width
    py_to_y = (y_max - y_min) / pad_height
    all_positions = []
    for graphs, level in zip(contours, contour_levels):
        # get the approximate label width
        text = "{}#sigma".format(try_int(level))
        label_width, label_height = get_text_extent(text, 12, 43)
        label_width *= px_to_x
        label_height *= py_to_y

        # calculate and store the position
        label_positions = locate_contour_labels(
            graphs,
            level,
            label_width,
            label_height,
            pad_width,
            pad_height,
            x_min,
            x_max,
            y_min,
            y_max,
            other_positions=all_positions,
            label_offset=1.3,
        )
        all_positions.extend(label_positions)

        # draw them
        for x, y, rot in label_positions:
            xsec_label = ROOT.TLatex(0., 0., text)
            r.setup_latex(
                xsec_label,
                props={
                    "NDC": False, "TextSize": 16, "TextAlign": 22, "TextColor": colors.black,
                    "TextAngle": rot, "X": x, "Y": y,
                },
            )
            draw_objs.append((xsec_label, "SAME"))

    # SM point
    if draw_sm_point:
        g_sm = create_tgraph(1, 1, 1)
        r.setup_graph(g_sm, props={"MarkerStyle": 33, "MarkerSize": 2.5}, color=colors.red)
        draw_objs.insert(-1, (g_sm, "P"))

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(pad=pad, postfix=cms_postfix or "", layout=cms_layout)
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if cms_layout.startswith("inside"):
            y_offset = 100 if cms_layout == "inside_vertical" and cms_postfix else 80
            param_kwargs = {"y_offset": y_offset}
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    pad.cd()
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)
