# coding: utf-8

"""
Limit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors as _colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, create_tgraph


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
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record array to dict mapping to arrays
    if isinstance(expected_values, np.ndarray):
        expected_values = {key: expected_values[key] for key in expected_values.dtype.names}

    # input checks
    assert poi in expected_values
    assert "limit" in expected_values
    poi_values = expected_values[poi]
    n_points = len(poi_values)
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
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi].label)
    y_title = "Upper 95% CLs limit on #sigma({} #rightarrow {}) / {}".format(
        to_root_latex(pp_process), to_root_latex(hh_process),
        to_root_latex(xsec_unit or "#sigma_{SM}"))
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # helper to read values into graphs
    def create_graph(sigma=None, values=None):
        # repeat the edges by padding to prevent bouncing effects of interpolated lines
        pad = lambda arr: np.pad(np.array(arr, dtype=np.float32), 1, mode="edge")
        arr = lambda key: pad(np.array(expected_values[key], dtype=np.float32))
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
    if "limit_p2" in expected_values and "limit_m2" in expected_values:
        g_2sigma = create_graph(sigma=2)
        r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
            "MarkerSize": 0, "FillColor": _colors.root.yellow})
        draw_objs.append((g_2sigma, "SAME,4"))
        legend_entries.insert(0, (g_2sigma, r"#pm 95% expected"))
        y_max_value = max(y_max_value, max(expected_values["limit_p2"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m2"]))

    # 1 sigma band
    if "limit_p1" in expected_values and "limit_m1" in expected_values:
        g_1sigma = create_graph(sigma=1)
        r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
            "MarkerSize": 0, "FillColor": _colors.root.green})
        draw_objs.append((g_1sigma, "SAME,4"))
        legend_entries.insert(0, (g_1sigma, r"#pm 68% expected"))
        y_max_value = max(y_max_value, max(expected_values["limit_p1"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m1"]))

    # central values
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
        "MarkerSize": 0})
    draw_objs.append((g_exp, "SAME,LEZ"))
    legend_entries.insert(0, (g_exp, "Median expected"))
    y_max_value = max(y_max_value, max(expected_values["limit"]))
    y_min_value = min(y_min_value, min(expected_values["limit"]))

    # observed values
    if observed_values is not None:
        g_inj = create_graph(values=observed_values)
        r.setup_graph(g_inj, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0})
        draw_objs.append((g_inj, "SAME,L"))
        legend_entries.insert(0, (g_inj, "Observed"))
        y_max_value = max(y_max_value, max(observed_values))
        y_min_value = min(y_min_value, min(observed_values))

    # theory prediction
    if theory_values is not None:
        g_thy = create_graph(values=theory_values)
        r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0, "LineColor": _colors.root.red})
        draw_objs.append((g_thy, "SAME,L"))
        legend_entries.append((g_thy, "SM prediction"))

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


def plot_limit_scans(
    path,
    poi,
    expected_values,
    names,
    theory_values=None,
    colors=None,
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
    Creates a plot showing multiple upper limit scans of a *poi* and saves it at *path*.
    *expected_values* should be a list of mappings to lists of values or a record array with keys
    "<poi_name>" and "limit". Each mapping in *expected_values* will result in a different curve.
    When *theory_values* is given, it should be a single lists of values. Therefore, it must have
    the same length as the lists given in *expected_values*. *names* denote the names of limit
    curves shown in the legend. When a name is found to be in dhi.config.br_hh_names, its value is
    used as a label instead. Likewise, *colors* can be a sequence of color numbers or names to be
    used per curve.

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
    assert n_graphs >= 1
    assert len(names) == n_graphs
    assert not colors or len(colors) == n_graphs
    assert all(poi in ev for ev in expected_values)
    assert all("limit" in ev for ev in expected_values)
    poi_values = expected_values[0][poi]
    n_points = len(poi_values)
    assert all(len(ev["limit"]) == n_points for ev in expected_values)
    if theory_values is not None:
        assert len(theory_values) == n_points
        theory_values = np.array(theory_values)

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
    y_title = "Upper 95% CLs limit on #sigma({} #rightarrow {}) / {}".format(
        to_root_latex(pp_process), to_root_latex(hh_process),
        to_root_latex(xsec_unit or "#sigma_{SM}"))
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # theory prediction
    if theory_values is not None:
        g_thy = create_tgraph(n_points, poi_values, theory_values)
        r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSize": 0, "LineColor": _colors.root.red})
        draw_objs.append((g_thy, "SAME,L"))
        legend_entries.append((g_thy, "SM prediction"))

    # central values
    for i, ev in enumerate(expected_values[::-1]):
        g_exp = create_tgraph(n_points, poi_values, ev["limit"])
        r.setup_graph(g_exp, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 0.7})
        if colors:
            color = colors[n_graphs - i - 1]
            color = _colors.root.get(color, color)
            r.set_color(g_exp, color)
            draw_objs.append((g_exp, "SAME,PL"))
        else:
            draw_objs.append((g_exp, "SAME,PL,PLC,PMC"))
        legend_entries.insert(0, (g_exp, names[n_graphs - i - 1]))
        y_max_value = max(y_max_value, max(ev["limit"]))
        y_min_value = min(y_min_value, min(ev["limit"]))

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
    legend_cols = int(math.ceil(len(legend_entries) / 4.))
    legend_rows = min(len(legend_entries), 4)
    legend = r.routines.create_legend(pad=pad, width=legend_cols * 220, height=legend_rows * 30,
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


def plot_limit_points(
    path,
    data,
    x_log=False,
    x_min=None,
    x_max=None,
    pp_process="pp",
    campaign="2017",
):
    """
    Creates a plot showing a comparison of limits of multiple analysis (or channels) and saves it at
    *path*. *data* should be a list of dictionaries with fields "name" (shown on the y axis),
    "expected" (a sequence of five values, i.e., central limit, +1 sigma, -1 sigma, +2 sigma, and
    -2 sigma), and "observed" (optional). When the name is a key of dhi.config.br_hh_names, its
    value is used as a label instead.

    When *x_log* is *True*, the x-axis is scaled logarithmically. *x_min* and *x_max* define the
    range of the x-axis and default to the maximum range of values passed in data, including
    uncertainties. The *pp_process* label is shown in the x-axis title to denote the physics process
    the computed values are corresponding to. *campaign* should refer to the name of a campaign
    label defined in dhi.config.campaign_labels.
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check minimal fields per data entry
    assert(all("name" in d for d in data))
    assert(all("expected" in d for d in data))
    n = len(data)
    has_obs = any("observed" in d for d in data)

    # set default ranges
    if x_min is None:
        x_min = 0.8 if x_log else 0.
    if x_max is None:
        x_max_value = max(sum(([e["expected"][3], e.get("observed", 0)] for e in data), []))
        x_max = x_max_value * 1.33

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = 130  # pixels
    entry_height = 90  # pixels
    head_space = 100  # pixels

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
    legend_entries = []

    # dummy histogram to control axes
    x_title = "Upper 95% CLs limit on #sigma({} #rightarrow HH) / #sigma_{{SM}}".format(
        to_root_latex(pp_process))
    h_dummy = ROOT.TH1F("dummy", ";{};".format(x_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    r.setup_x_axis(h_dummy.GetXaxis(), pad, props={"TitleOffset": 1.2})
    draw_objs.append((h_dummy, "HIST"))

    # helper to read values into graphs
    def create_graph(sigma=None, obs=False):
        # repeat the edges by padding to prevent bouncing effects of interpolated lines
        zeros = np.zeros(n, dtype=np.float32)
        y = (np.arange(n, dtype=np.float32))[::-1]
        x_err_u, x_err_d = zeros, zeros
        y_err_u, y_err_d = zeros + 1, zeros
        if obs:
            limits = np.array([d.get("observed", 1e5) for d in data], dtype=np.float32)
            y, y_err_u, y_err_d = y + 0.5, zeros + 0.5, zeros + 0.5
        else:
            limits = np.array([d["expected"][0] for d in data], dtype=np.float32)
            if sigma:
                x_err_d = np.array([d["expected"][sigma * 2] for d in data], dtype=np.float32)
                x_err_d = limits - x_err_d
                x_err_u = np.array([d["expected"][sigma * 2 - 1] for d in data], dtype=np.float32)
                x_err_u = x_err_u - limits
        return create_tgraph(n, limits, y, x_err_d, x_err_u, y_err_d, y_err_u)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(g_2sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
        "MarkerSize": 0, "FillColor": _colors.root.yellow})
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries.insert(0, (g_2sigma, r"#pm 95% expected"))

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(g_1sigma, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
        "MarkerSize": 0, "FillColor": _colors.root.green})
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries.insert(0, (g_1sigma, r"#pm 68% expected"))

    # central values
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
        "MarkerSize": 0})
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries.insert(0, (g_exp, "Median expected"))

    # observed values
    if has_obs:
        g_inj = create_graph(obs=True)
        r.setup_graph(g_inj, props={"LineWidth": 2, "LineStyle": 1, "MarkerStyle": 20,
            "MarkerSiz2e": 2})
        draw_objs.append((g_inj, "SAME,PEZ"))
        legend_entries.insert(0, (g_inj, "Observed limit"))
    else:
        # add a dummy to the legend entries
        legend_entries.insert(1, (h_dummy, " ", ""))

    # vertical line at 1
    # if x_min < 1:
    #     line_one = ROOT.TLine(1., 0., 1., n)
    #     r.setup_line(line_one, props={"NDC": False, "LineColor": _colors.root.red})
    #     draw_objs.append(line_one)

    # line to separate combined result
    if data[-1]["name"].lower() == "combined":
        line_obs = ROOT.TLine(x_min, 1., x_max, 1)
        r.setup_line(line_obs, props={"NDC": False, "LineStyle": 3})
        draw_objs.append(line_obs)

    # y axis labels and ticks
    h_dummy.GetYaxis().SetBinLabel(1, "")
    label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{Expected %.1f}}"
    label_tmpl_obs = "#splitline{#bf{%s}}{#scale[0.75]{#splitline{Expected %.1f}{Observed %.1f}}}"
    for i, d in enumerate(data):
        # name labels
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        if "observed" in d:
            label = label_tmpl_obs % (label, d["expected"][0], d["observed"])
        else:
            label = label_tmpl % (label, d["expected"][0])
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

    # legend
    legend = r.routines.create_legend(pad=pad, width=440, height=70,
        x2=r.get_x(0, pad, anchor="right"), y2=r.get_y(16, pad, anchor="top"))
    r.setup_legend(legend, props={"NColumns": 2})
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
