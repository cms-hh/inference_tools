# coding: utf-8

"""
Limit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import (
    poi_data, br_hh_names, campaign_labels, colors, color_sequence, marker_sequence,
)
from dhi.util import import_ROOT, to_root_latex, create_tgraph, try_int, colored
from dhi.plots.util import (
    use_style, draw_model_parameters, create_hh_process_label, determine_limit_digits,
    get_graph_points,
)


colors = colors.root


@use_style("dhi_default")
def plot_limit_scan(
    path,
    poi,
    scan_parameter,
    expected_values,
    observed_values=None,
    theory_values=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    xsec_unit=None,
    hh_process=None,
    model_parameters=None,
    campaign=None,
    show_points=False,
):
    """
    Creates a plot for the upper limit scan of a *poi* over a *scan_parameter* and saves it at
    *path*. *expected_values* should be a mapping to lists of values or a record array with keys
    "<scan_parameter>" and "limit", and optionally "limit_p1" (plus 1 sigma), "limit_m1" (minus 1
    sigma), "limit_p2" and "limit_m2". When the variations by 1 or 2 sigma are missing, the plot is
    created without them. When *observed_values* is set, it should have a similar format with keys
    "<scan_parameter>" and "limit". When *theory_values* is set, it should have a similar format
    with keys "<scan_parameter>" and "xsec", and optionally "xsec_p1" and "xsec_m1".

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *x_min*, *x_max*,
    *y_min* and *y_max* define the axes ranges and default to the ranges of the given values.
    *xsec_unit* denotes whether the passed values are given as real cross sections in this unit or,
    when *None*, as a ratio over the theory prediction. *hh_process* can be the name of a HH
    subprocess configured in *dhi.config.br_hh_names* and is inserted to the process name
    in the title of the y-axis and indicates that the plotted cross section data was (e.g.) scaled
    by a branching ratio. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*. When *show_points* is *True*, the central scan points are drawn
    on top of the interpolated curve.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#limit-on-poi-vs-scan-parameter
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # input checks
    def check_values(values, keys=None):
        # convert record array to dict mapping to arrays
        if isinstance(values, np.ndarray):
            values = {key: values[key] for key in values.dtype.names}
        assert(scan_parameter in values)
        if keys:
            assert(all(key in values for key in keys))
        return values

    expected_values = check_values(expected_values, ["limit"])
    if observed_values is not None:
        observed_values = check_values(observed_values, ["limit"])
    has_thy = theory_values is not None
    has_thy_err = False
    if theory_values is not None:
        theory_values = check_values(theory_values, ["xsec"])
        has_thy_err = "xsec_p1" in theory_values and "xsec_m1" in theory_values

    # set default ranges
    if x_min is None:
        x_min = min(expected_values[scan_parameter])
    if x_max is None:
        x_max = max(expected_values[scan_parameter])

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "Upper 95% CLs limit on #sigma({}) / {}".format(
        create_hh_process_label(poi, hh_process), to_root_latex(xsec_unit or "#sigma_{SM}"))
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # setup up to 6 legend entries that are inserted by index downstream
    legend_entries = 6 * [(h_dummy, " ", "L")]

    # helper to read values into graphs
    def create_graph(values=expected_values, key="limit", sigma=None, pad=True, insert=None):
        return create_tgraph(
            len(values[key]),
            values[scan_parameter],
            values[key],
            0,
            0,
            (values[key] - values["{}_m{}".format(key, sigma)]) if sigma else 0,
            (values["{}_p{}".format(key, sigma)] - values[key]) if sigma else 0,
            pad=pad,
            insert=insert,
        )

    # 2 sigma band
    g_2sigma = None
    if "limit_p2" in expected_values and "limit_m2" in expected_values:
        g_2sigma = create_graph(sigma=2)
        r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.yellow})
        draw_objs.append((g_2sigma, "SAME,4"))  # option 4 might fallback to 3, see below
        legend_entries[5] = (g_2sigma, "#pm 2 #sigma expected", "LF")
        y_max_value = max(y_max_value, max(expected_values["limit_p2"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m2"]))

    # 1 sigma band
    g_1sigma = None
    if "limit_p1" in expected_values and "limit_m1" in expected_values:
        g_1sigma = create_graph(sigma=1)
        r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.green})
        draw_objs.append((g_1sigma, "SAME,4"))  # option 4 might fallback to 3, see below
        legend_entries[4] = (g_1sigma, "#pm 1 #sigma expected", "LF")
        y_max_value = max(y_max_value, max(expected_values["limit_p1"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m1"]))

    # central values
    g_exp = create_graph()
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 2})
    draw_objs.append((g_exp, "SAME,CP" if show_points else "SAME,C"))
    legend_entries[3] = (g_exp, "Median expected", "L")
    y_max_value = max(y_max_value, max(expected_values["limit"]))
    y_min_value = min(y_min_value, min(expected_values["limit"]))

    # observed values
    if observed_values is not None:
        g_inj = create_graph(values=observed_values)
        r.setup_graph(g_inj, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_inj, "SAME,C"))
        legend_entries[0] = (g_inj, "Observed", "L")
        y_max_value = max(y_max_value, max(observed_values["limit"]))
        y_min_value = min(y_min_value, min(observed_values["limit"]))

    # get theory prediction limits
    if has_thy:
        y_min_value = min(y_min_value, min(theory_values["xsec_m1" if has_thy_err else "xsec"]))

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

    # draw option 4 of error graphs is buggy when the error point or bar is above the visible,
    # vertical range of the pad, so check these cases and fallback to option 3
    fallback_graph_option = False
    for g in filter(bool, [g_2sigma, g_1sigma]):
        _, y_values, _, _, _, y_errors_up = get_graph_points(g, errors=True)
        if any((y + y_err_up) >= y_max for y, y_err_up in zip(y_values, y_errors_up)):
            fallback_graph_option = True
            break
    if fallback_graph_option:
        for i, objs in enumerate(list(draw_objs)):
            if isinstance(objs, tuple) and len(objs) >= 2 and objs[0] in [g_2sigma, g_1sigma]:
                draw_objs[i] = (objs[0], "SAME,3") + objs[2:]
                print("{}: changed draw option of graph {} to '3' as it exceeds the vertical pad "
                    "range which is not supported for option '4'".format(
                        colored("WARNING", "yellow"), objs[0]))

    # theory prediction
    if has_thy:
        if has_thy_err:
            # when the maximum value is far above the maximum y range, ROOT will fail drawing the
            # first point correctly, so insert two values that are so off that it does not matter
            insert = [(0, -1e7, 0, 0, 0, 0, 0)] if max(theory_values["xsec"]) > y_max else None
            g_thy = create_graph(values=theory_values, key="xsec", sigma=1, insert=insert)
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red,
                "FillStyle": 1001, "FillColor": colors.red_trans_50})
            draw_objs.append((g_thy, "SAME,C3"))
            legend_entries[0 if observed_values is None else 1] = (g_thy, "Theory prediction", "LF")
        else:
            g_thy = create_graph(values=theory_values, key="xsec")
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red})
            draw_objs.append((g_thy, "SAME,C"))
            legend_entries[0 if observed_values is None else 1] = (g_thy, "Theory prediction", "L")

    # legend
    legend = r.routines.create_legend(pad=pad, width=440, n=3, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "trl",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        draw_objs.extend(draw_model_parameters(model_parameters, pad))

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
def plot_limit_scans(
    path,
    poi,
    scan_parameter,
    names,
    expected_values,
    theory_values=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    xsec_unit=None,
    hh_process=None,
    model_parameters=None,
    campaign=None,
    show_points=True,
):
    """
    Creates a plot showing multiple upper limit scans of a *poi* over a *scan_parameter* and saves
    it at *path*. *expected_values* should be a list of mappings to lists of values or a record
    array with keys "<scan_parameter>" and "limit". Each mapping in *expected_values* will result in
    a different curve. When *theory_values* is set, it should have a similar format with keys
    "<scan_parameter>" and "xsec", and optionally "xsec_p1" and "xsec_m1". *names* denote the names
    of limit curves shown in the legend. When a name is found to be in dhi.config.br_hh_names, its
    value is used as a label instead.

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *x_min*, *x_max*,
    *y_min* and *y_max* define the axis ranges and default to the range of the given values.
    *xsec_unit* denotes whether the passed values are given as real cross sections in this unit or,
    when *None*, as a ratio over the theory prediction. *hh_process* can be the name of a HH
    subprocess configured in *dhi.config.br_hh_names* and is inserted to the process name in the
    title of the y-axis and indicates that the plotted cross section data was (e.g.) scaled by a
    branching ratio. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    When *show_points* is *True*, the central scan points are drawn on top of the interpolated
    curve.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#multiple-limits-on-poi-vs-scan-parameter
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
    assert(all("limit" in ev for ev in expected_values))
    scan_values = expected_values[0][scan_parameter]
    has_thy = theory_values is not None
    has_thy_err = False
    if theory_values is not None:
        # convert record array to dicts mapping to arrays
        if isinstance(theory_values, np.ndarray):
            theory_values = {key: theory_values[key] for key in theory_values.dtype.names}
        assert(scan_parameter in theory_values)
        assert("xsec" in theory_values)
        has_thy_err = "xsec_p1" in theory_values and "xsec_m1" in theory_values

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
    y_title = "Upper 95% CLs limit on #sigma({}) / {}".format(
        create_hh_process_label(poi, hh_process), to_root_latex(xsec_unit or "#sigma_{SM}"))
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # central values
    for i, (ev, col, ms) in enumerate(zip(expected_values[::-1], color_sequence[:n_graphs][::-1],
            marker_sequence[:n_graphs][::-1])):
        mask = ~np.isnan(ev["limit"])
        g_exp = create_tgraph(mask.sum(), scan_values[mask], ev["limit"][mask])
        r.setup_graph(g_exp, props={"LineWidth": 2, "MarkerStyle": ms, "MarkerSize": 1.2},
            color=colors[col])
        draw_objs.append((g_exp, "SAME,CP" if show_points else "SAME,C"))
        name = names[n_graphs - i - 1]
        legend_entries.append((g_exp, to_root_latex(br_hh_names.get(name, name)),
            "LP" if show_points else "L"))
        y_max_value = max(y_max_value, max(ev["limit"]))
        y_min_value = min(y_min_value, min(ev["limit"]))

    # get theory prediction limits
    if has_thy:
        y_min_value = min(y_min_value, min(theory_values["xsec_m1" if has_thy_err else "xsec"]))

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

    # draw the theory prediction
    if has_thy:
        scan_values_thy = theory_values[scan_parameter]
        if has_thy_err:
            # when the maximum value is far above the maximum y range, ROOT will fail drawing the
            # first point correctly, so insert two values that are so off that it does not matter
            insert = [(0, -1e7, 0, 0, 0, 0, 0)] if max(theory_values["xsec"]) > y_max else None
            g_thy = create_tgraph(len(scan_values_thy), scan_values_thy, theory_values["xsec"],
                0, 0, theory_values["xsec"] - theory_values["xsec_m1"],
                theory_values["xsec_p1"] - theory_values["xsec"], pad=True, insert=insert)
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red,
                "FillStyle": 1001, "FillColor": colors.red_trans_50})
            draw_objs.insert(1, (g_thy, "SAME,C3"))
            legend_entries.append((g_thy, "Theory prediction", "LF"))
        else:
            g_thy = create_tgraph(len(scan_values_thy), scan_values_thy, theory_values["xsec"])
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1}, color=colors.red)
            draw_objs.insert(1, (g_thy, "SAME,C"))
            legend_entries.append((g_thy, "Theory prediction", "L"))

    # legend
    legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
    legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
    legend = r.routines.create_legend(pad=pad, width=legend_cols * 210, n=legend_rows,
        props={"NColumns": legend_cols, "TextSize": 18})
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "trl",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

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
def plot_limit_points(
    path,
    poi,
    data,
    x_log=False,
    x_min=None,
    x_max=None,
    xsec_unit=None,
    hh_process=None,
    left_margin=None,
    model_parameters=None,
    h_lines=None,
    campaign=None,
    digits=None,
):
    """
    Creates a plot showing a comparison of limits of multiple analysis (or channels) on a *poi* and
    saves it at *path*. *data* should be a list of dictionaries with fields

    - "expected", a sequence of five values, i.e., central limit, and +1 sigma, -1 sigma, +2 sigma,
      and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a single value,
    - "theory" (optional), a single value or a sequence of three values, i.e., nominal value, and
      +1 sigma and -1 sigma variations (absolute values, not errors!),
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead.

    Example:

    .. code-block:: python

        plot_limit_points(
            path="plot.pdf",
            poi="r",
            data=[{
                "expected": (40., 50., 28., 58., 18.),
                "observed": 45.,
                "theory": (38., 40., 36.),
                "name": "bbXX",
            }, {
                ...
            }],
        )

    When *x_log* is *True*, the x-axis is scaled logarithmically. *x_min* and *x_max* define the
    range of the x-axis and default to the maximum range of values passed in data, including
    uncertainties. *xsec_unit* denotes whether the passed values are given as real cross sections in
    this unit or, when *None*, as a ratio over the theory prediction. *hh_process* can be the name
    of a HH subprocess configured in *dhi.config.br_hh_names* and is inserted to the process name in
    the title of the x-axis and indicates that the plotted cross section data was (e.g.) scaled by a
    branching ratio. *left_margin* controls the left margin of the pad in pixels. *model_parameters*
    can be a dictionary of key-value pairs of model parameters. *h_lines* can be a list of integers
    denoting positions where additional horizontal lines are drawn for visual guidance. *campaign*
    should refer to the name of a campaign label defined in dhi.config.campaign_labels. *digits*
    controls the number of digits of the limit values shown for each entry. When *None*, a number
    based on the lowest limit values is determined automatically.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#multiple-limits-at-a-certain-point-of-parameters
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check inputs and get extrema
    n = len(data)
    has_obs = False
    has_thy = False
    has_thy_err = False
    x_min_value = 1e5
    x_max_value = -1e5
    for d in data:
        assert("name" in d)
        assert("expected" in d)
        x_min_value = min(x_min_value, min(d["expected"]))
        x_max_value = max(x_max_value, max(d["expected"]))
        if "observed" in d:
            assert(isinstance(d["observed"], (float, int)))
            has_obs = True
            x_min_value = min(x_min_value, d["observed"])
            x_max_value = max(x_max_value, d["observed"])
            d["observed"] = [d["observed"]]
        if "theory" in d:
            if isinstance(d["theory"], (tuple, list)):
                if len(d["theory"]) == 3:
                    has_thy_err = True
                else:
                    assert(len(d["theory"]) == 1)
            else:
                d["theory"] = 3 * (d["theory"],)
            has_thy = True
            x_min_value = min(x_min_value, min(d["theory"]))
            x_max_value = max(x_max_value, max(d["theory"]))

    # set default ranges
    if x_min is None:
        if not xsec_unit:
            x_min = 0.75 if x_log else 0.
        else:
            x_min = 0.75 * x_min_value
    if x_max is None:
        x_max = x_max_value * 1.33

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = left_margin or 150  # pixels
    entry_height = 90  # pixels
    head_space = 130  # pixels

    # get the canvas height
    canvas_height = n * entry_height + head_space + top_margin + bottom_margin

    # get relative pad margins and fill into props
    pad_margins = {
        "TopMargin": float(top_margin) / canvas_height,
        "BottomMargin": float(bottom_margin) / canvas_height,
        "LeftMargin": float(left_margin) / canvas_width,
        "Logx": x_log,
    }

    # get the y maximum
    y_max = (canvas_height - top_margin - bottom_margin) / float(entry_height)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(width=canvas_width, height=canvas_height,
        pad_props=pad_margins)
    pad.cd()
    draw_objs = []

    # dummy histogram to control axes
    x_title = "Upper 95% CLs limit on #sigma({}) / {}".format(
        create_hh_process_label(poi, hh_process), to_root_latex(xsec_unit or "#sigma_{SM}"))
    h_dummy = ROOT.TH1F("dummy", ";{};".format(x_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    r.setup_x_axis(h_dummy.GetXaxis(), pad=pad, props={"TitleOffset": 1.2,
        "LabelOffset": r.pixel_to_coord(canvas, y=4)})
    draw_objs.append((h_dummy, "HIST"))

    # setup up to 6 legend entries that are inserted by index downstream
    legend_entries = 6 * [(h_dummy, " ", "L")]

    # helper to read values into graphs
    def create_graph(key="expected", sigma=None):
        # repeat the edges by padding to prevent bouncing effects of interpolated lines
        _data = [d for d in data if key in d]
        n = len(_data)
        zeros = np.zeros(n, dtype=np.float32)
        y = (np.arange(n, dtype=np.float32))[::-1]
        x_err_u, x_err_d = zeros, zeros
        y_err_u, y_err_d = zeros + 1, zeros
        limits = np.array([d[key][0] for d in _data], dtype=np.float32)
        if sigma:
            x_err_d = np.array([d[key][sigma * 2] for d in _data], dtype=np.float32)
            x_err_d = limits - x_err_d
            x_err_u = np.array([d[key][sigma * 2 - 1] for d in _data], dtype=np.float32)
            x_err_u = x_err_u - limits
        return create_tgraph(n, limits, y, x_err_d, x_err_u, y_err_d, y_err_u)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.yellow})
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries[5] = (g_2sigma, r"#pm 2 #sigma expected", "LF")

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.green})
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries[4] = (g_1sigma, r"#pm 1 #sigma expected", "LF")

    # central values
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 2})
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries[3] = (g_exp, "Median expected", "L")

    # observed values
    if has_obs:
        g_obs = create_graph(key="observed")
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,EZ"))
        legend_entries[0] = (g_obs, "Observed", "L")

    # vertical line for theory prediction, represented by a graph in case of uncertainties
    if has_thy and any((d["theory"][0] >= x_min) for d in data):
        # uncertainty line
        g_thy_line = create_graph(key="theory")
        r.setup_graph(g_thy_line, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red})
        draw_objs.append((g_thy_line, "SAME,LZ"))
        legend_entry = (g_thy_line, "Theory prediction", "L")
        # uncertainty area
        if has_thy_err:
            g_thy_area = create_graph(key="theory", sigma=1)
            r.setup_graph(g_thy_area, props={"LineWidth": 2, "LineStyle": 1,
                "LineColor": colors.red, "FillStyle": 1001, "FillColor": colors.red_trans_50})
            draw_objs.append((g_thy_area, "SAME,2"))
            legend_entry = (g_thy_area, "Theory prediction", "LF")
        legend_entries[1 if has_obs else 0] = legend_entry

    # horizontal guidance lines
    if h_lines:
        for i in h_lines:
            line_obs = ROOT.TLine(x_min, float(i), x_max, float(i))
            r.setup_line(line_obs, props={"NDC": False}, color=12)
            draw_objs.append(line_obs)

    # y axis labels and ticks
    y_label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{Expected %s}}"
    y_label_tmpl_obs = "#splitline{#bf{%s}}{#scale[0.75]{#splitline{Expected %s}{Observed %s}}}"
    if digits is None:
        min_limit = min(sum((([d["expected"][0]] + [d.get("observed", 1e7)]) for d in data), []))
        digits = determine_limit_digits(min_limit, is_xsec=bool(xsec_unit))

    def make_y_label(name, exp, obs=None):
        if xsec_unit:
            fmt = lambda v: "{{:.{}f}} {{}}".format(digits).format(v, xsec_unit)
        else:
            fmt = lambda v: "{{:.{}f}}".format(digits).format(v)
        if obs is None:
            return y_label_tmpl % (label, fmt(exp))
        else:
            return y_label_tmpl_obs % (label, fmt(exp), fmt(obs[0]))

    h_dummy.GetYaxis().SetBinLabel(1, "")
    for i, d in enumerate(data):
        # name labels
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        label = make_y_label(label, d["expected"][0], d.get("observed"))
        label_x = r.get_x(10, canvas)
        label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        label = ROOT.TLatex(label_x, label_y, label)
        r.setup_latex(label, props={"NDC": True, "TextAlign": 12, "TextSize": 22})
        draw_objs.append(label)

        # left and right ticks
        tick_length = 0.03
        if x_log:
            tick_length_l = x_min * ((x_max / float(x_min))**tick_length - 1)
            tick_length_r = x_max * (1 - (x_min / float(x_max))**tick_length)
        else:
            tick_length_l = tick_length_r = tick_length * (x_max - x_min)
        tl = ROOT.TLine(x_min, i + 1, x_min + tick_length_l, i + 1)
        tr = ROOT.TLine(x_max - tick_length_r, i + 1, x_max, i + 1)
        r.setup_line(tl, props={"NDC": False, "LineWidth": 1})
        r.setup_line(tr, props={"NDC": False, "LineWidth": 1})
        draw_objs.extend([tl, tr])

    # model parameter labels
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=440, n=3, props={"NColumns": 2})
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
def plot_benchmark_limits(
    path,
    data,
    poi="r_gghh",
    y_min=None,
    y_max=None,
    y_log=False,
    xsec_unit="fb",
    hh_process=None,
    campaign=None,
    bar_width=0.66,
):
    """
    Creates a plot showing a the limits of BSM benchmarks for a *poi* and saves it at *path*. *data*
    should be a list of dictionaries with fields

    - "expected", a sequence of five values, i.e., central limit, and +1 sigma, -1 sigma, +2 sigma,
      and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a single value,
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead.

    Example:

    .. code-block:: python

        plot_benchmark_limits(
            path="plot.pdf",
            data=[{
                "expected": (40., 50., 28., 58., 18.),
                "observed": 45.,
                "name": "1",
            }, {
                ...
            }],
        )

    *y_min* and *y_max* define the y axis range and default to the range of the given values. When
    *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *xsec_unit* denotes the unit
    of the passed values and defaults to fb. *hh_process* can be the name of a HH subprocess
    configured in *dhi.config.br_hh_names* and is inserted to the process name in the title of the
    y-axis and indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    The *bar_width* should be a value between 0 and 1 and controls the fraction of the limit bar
    width relative to the bin width.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/eft.html#benchmark-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check inputs and get extrema
    n = len(data)
    has_obs = False
    y_min_value = 1e5
    y_max_value = -1e5
    for d in data:
        assert("name" in d)
        assert("expected" in d)
        y_min_value = min(y_min_value, min(d["expected"]))
        y_max_value = max(y_max_value, max(d["expected"]))
        if "observed" in d:
            assert(isinstance(d["observed"], (float, int)))
            has_obs = True
            y_min_value = min(y_min_value, d["observed"])
            y_max_value = max(y_max_value, d["observed"])
            d["observed"] = [d["observed"]]

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

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    x_title = "Shape benchmark"
    y_title = "Upper 95% CLs limit on #sigma({}) / {}".format(
        create_hh_process_label(poi, hh_process), to_root_latex(xsec_unit))
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), n, -0.5, n - 0.5)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(h_dummy.GetXaxis(), pad=pad, props={"Ndivisions": n})
    draw_objs.append((h_dummy, "HIST"))

    # benchmark labels
    for i, d in enumerate(data):
        h_dummy.GetXaxis().SetBinLabel(i + 1, d["name"])

    # helper to read values into graphs
    def create_graph(key="expected", sigma=None):
        args = x, y, x_err_d, x_err_u, y_err_d, y_err_u = [n * [0.] for _ in range(6)]
        for i, d in enumerate(data):
            if key not in d:
                y[i] = -1.e5
                continue
            x[i] = i - 0.5 * bar_width
            x_err_u[i] = bar_width
            y[i] = d[key][0]
            if sigma:
                y_err_d[i] = y[i] - d[key][sigma * 2]
                y_err_u[i] = d[key][sigma * 2 - 1] - y[i]
        return create_tgraph(n, *args)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.yellow})
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries.append((g_2sigma, r"#pm 2 #sigma expected", "LF"))

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.green})
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries.insert(0, (g_1sigma, r"#pm 1 #sigma expected", "LF"))

    # prepare graphs
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 2})
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries.insert(0, (g_exp, "Median expected", "L"))

    # observed values
    if has_obs:
        g_obs = create_graph(key="observed")
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,EZ"))
        legend_entries.insert(0, (g_obs, "Observed", "L"))

    # legend
    n_cols = 2 if has_obs else 1
    legend = r.routines.create_legend(pad=pad, width=220 * n_cols, n=3, props={"NColumns": n_cols})
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
