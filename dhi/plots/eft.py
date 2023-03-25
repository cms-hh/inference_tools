# coding: utf-8

"""
Limit plots using ROOT.
"""

import math
from collections import OrderedDict

from dhi.config import (
    br_hh_names, br_hh_colors, campaign_labels, colors, color_sequence, bm_labels,
)
from dhi.util import import_ROOT, to_root_latex, create_tgraph, make_list, make_tuple
from dhi.plots.util import use_style, create_hh_xsbr_label, get_y_range, Style


colors = colors.root


@use_style("dhi_default")
def plot_benchmark_limits(
    paths,
    data,
    poi,
    y_min=None,
    y_max=None,
    y_log=False,
    xsec_unit="fb",
    hh_process=None,
    campaign=None,
    bar_width=0.6,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing the limits of BSM benchmarks for a *poi* and saves it at *paths*. *data*
    should be a list of dictionaries with fields

    - "expected", a sequence of five values, i.e., central limit, and +1 sigma, -1 sigma, +2 sigma,
      and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a single value,
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead. It can also be a 2-tuple (name, group name) and if so,
      lines are draw between groups and a group name label is shown.

    Example:

    .. code-block:: python

        plot_benchmark_limits(
            paths=["plot.pdf", "plot.png"],
            data=[
                {
                    "expected": (40., 50., 28., 58., 18.),
                    "observed": 45.,
                    "name": "1",
                }, {
                    ...
                },
            ],
        )

    *y_min* and *y_max* define the y axis range and default to the range of the given values. When
    *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *xsec_unit* denotes the unit
    of the passed values and defaults to fb. *hh_process* can be the name of a HH subprocess
    configured in *dhi.config.br_hh_names* and is inserted to the process name in the title of the
    y-axis and indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    The *bar_width* should be a value between 0 and 1 and controls the fraction of the limit bar
    width relative to the bin width. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"
        - "multilepton"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/eft.html#benchmark-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    style.sep_line_style = 1
    style.sep_line_color = colors.black
    style.sep_line_width = 2
    style.exp_line_color = colors.black
    style.exp_line_style = 2
    style.exp_line_width = 2
    style.bar_width = bar_width
    if style.matches("paper"):
        cms_postfix = None
    if style.matches("multilepton"):
        style.sep_line_color = colors.red
        style.sep_line_style = 9
        style.sep_line_width = 5
        style.exp_line_color = colors.blue
        style.exp_line_style = 2
        style.exp_line_width = 5
        style.bar_width = 1

    # check inputs and get extrema
    n = len(data)
    has_obs = False
    y_min_value = 1e5
    y_max_value = -1e5
    for d in data:
        assert "name" in d
        assert "expected" in d
        y_min_value = min(y_min_value, min(d["expected"]))
        y_max_value = max(y_max_value, max(d["expected"]))
        if "observed" in d:
            assert isinstance(d["observed"], (float, int))
            has_obs = True
            y_min_value = min(y_min_value, d["observed"])
            y_max_value = max(y_max_value, d["observed"])
            d["observed"] = [d["observed"]]

    # when certain names are detected (keys of bm_labels), group and reoder the cards
    groups = OrderedDict()
    if any(d["name"] in bm_labels for d in data):
        # sort
        bm_keys = list(bm_labels.keys())
        data.sort(key=lambda d: bm_keys.index(d["name"]) if d["name"] in bm_keys else 1e9)

        # fill the group dict
        for d in data:
            bm_name, group_name = d["name"], ""
            if bm_name in bm_labels:
                bm_name, group_name = bm_labels[bm_name]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(bm_name)
            d["name"] = bm_name

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(
        width=1200,
        height=600,
        pad_props={"Logy": y_log, "LeftMargin": 0.09, "BottomMargin": 0.15 if groups else 0.1},
    )
    pad.cd()
    draw_objs = []

    # dummy histogram to control axes
    x_title = "Benchmark scenario"
    y_title = "95% CL limit on {} ({})".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex(xsec_unit),
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), n, -0.5, n - 0.5)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"Ndivisions": n, "TitleOffset": 1.7 if groups else 1.1},
    )
    draw_objs.append((h_dummy, "HIST"))

    # reserve legend entries
    legend_entries = 4 * [(h_dummy, " ", "L")]

    # benchmark labels
    for i, d in enumerate(data):
        h_dummy.GetXaxis().SetBinLabel(i + 1, to_root_latex(make_tuple(d["name"])[0]))

    # helper to read values into graphs
    def create_graph(key="expected", sigma=None):
        args = x, y, x_err_d, x_err_u, y_err_d, y_err_u = [n * [0.] for _ in range(6)]
        for i, d in enumerate(data):
            if key not in d:
                y[i] = -1.e5
                continue
            x[i] = i - 0.5 * style.bar_width
            x_err_u[i] = style.bar_width
            y[i] = d[key][0]
            if sigma:
                y_err_d[i] = y[i] - d[key][sigma * 2]
                y_err_u[i] = d[key][sigma * 2 - 1] - y[i]
        return create_tgraph(n, *args)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(
        g_2sigma,
        color=colors.brazil_yellow,
        color_flags="lf",
        props={"LineWidth": 2, "LineStyle": 1},
    )
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries[3] = (g_2sigma, r"95% expected", "LF")

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(
        g_1sigma,
        color=colors.brazil_green,
        color_flags="lf",
        props={"LineWidth": 2, "LineStyle": 1},
    )
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries[2] = (g_1sigma, r"68% expected", "LF")

    # prepare graphs
    g_exp = create_graph(sigma=0)
    r.setup_graph(
        g_exp,
        props={
            "LineWidth": style.exp_line_width, "LineStyle": style.exp_line_style,
            "LineColor": style.exp_line_color,
        },
    )
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries[0] = (g_exp, "Median expected", "L")

    # observed values
    if has_obs:
        g_obs = create_graph(key="observed")
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,EZ"))
        legend_entries[1] = (g_obs, "Observed", "L")

    # texts over and lines between groups
    group_names = list(groups)
    for i, group_name in enumerate(group_names):
        # label
        x = sum(len(groups[group_names[j]]) for j in range(i + 1))
        y_offset = 0.1
        label = ROOT.TLatex(
            x - 0.5 * (1 + len(groups[group_name])),
            (y_min * (y_min / y_max)**y_offset) if y_log else (y_min - y_offset * (y_max - y_min)),
            to_root_latex(group_name),
        )
        r.setup_latex(label, props={"NDC": False, "TextAlign": 21, "TextSize": 20})
        draw_objs.append(label)

        # separating line
        if i < len(groups) - 1:
            line = ROOT.TLine(x - 0.5, y_min, x - 0.5, y_max_value)
            r.setup_line(
                line,
                props={
                    "LineColor": style.sep_line_color, "LineStyle": style.sep_line_style,
                    "NDC": False,
                },
            )
            draw_objs.append(line)

    # legend
    legend = r.routines.create_legend(
        pad=pad,
        width=440,
        y2=-20,
        n=2,
        props={"NColumns": 2},
    )
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout=cms_layout,
    )
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
    for path in make_list(paths):
        canvas.SaveAs(path)


@use_style("dhi_default")
def plot_multi_benchmark_limits(
    paths,
    data,
    names,
    poi,
    y_min=None,
    y_max=None,
    y_log=False,
    xsec_unit="fb",
    hh_process=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing multiple limits for various BSM benchmarks for a *poi* and saves it at
    *paths*. *data* should be a list of dictionaries with fields

    - "expected", a list of sequences of five values, i.e., central limit, and +1 sigma, -1 sigma,
      +2 sigma, and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a sequence of single float values and when set, it should have the same
      length as "expected",
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead. It can also be a 2-tuple (name, group name) and if so,
      lines are draw between groups and a group name label is shown.

    Example:

    .. code-block:: python

        plot_benchmark_limits(
            paths=["plot.pdf", "plot.png"],
            data=[
                {
                    "expected": [
                        (40.0, 50.0, 28.0, 58.0, 18.0),
                        (38.0, 47.0, 27.0, 61.0, 19.0),
                    ],
                    "observed": [
                        45.0,
                        42.0,
                    ],
                    "name": "1",
                }, {
                    ...
                },
            ],
            names=["Analysis A", "Aenalysis B"],
        )

    *names* should be a list of names that are shown in the legend and refer to each element in the
    *exected* and *observed* lists in *data*.

    *y_min* and *y_max* define the y axis range and default to the range of the given values. When
    *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *xsec_unit* denotes the unit
    of the passed values and defaults to fb. *hh_process* can be the name of a HH subprocess
    configured in *dhi.config.br_hh_names* and is inserted to the process name in the title of the
    y-axis and indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/eft.html#multiple-benchmark-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check inputs and get extrema
    n = len(data)
    n_graphs = len(names)
    has_obs = False
    y_min_value = 1e5
    y_max_value = -1e5
    for d in data:
        assert "name" in d
        assert "expected" in d
        assert isinstance(d["expected"], (list, tuple))
        assert all(isinstance(e, (list, tuple)) for e in d["expected"])
        assert len(d["expected"]) == n_graphs
        y_min_value = min(y_min_value, min(e[0] for e in d["expected"]))
        y_max_value = max(y_max_value, max(e[0] for e in d["expected"]))
        if "observed" in d:
            assert isinstance(d["observed"], (list, tuple))
            assert all(isinstance(o, (int, float)) for o in d["observed"])
            assert len(d["observed"]) == n_graphs
            has_obs = True
            y_min_value = min(y_min_value, min(d["observed"]))
            y_max_value = max(y_max_value, max(d["observed"]))

    # when certain names are detected (keys of bm_labels), group and reoder the cards
    groups = OrderedDict()
    if any(d["name"] in bm_labels for d in data):
        # sort
        bm_keys = list(bm_labels.keys())
        data.sort(key=lambda d: bm_keys.index(d["name"]) if d["name"] in bm_keys else 1e9)

        # fill the group dict
        for d in data:
            bm_name, group_name = d["name"], ""
            if bm_name in bm_labels:
                bm_name, group_name = bm_labels[bm_name]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(bm_name)
            d["name"] = bm_name

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(
        width=1200,
        height=600,
        pad_props={"Logy": y_log, "LeftMargin": 0.09, "BottomMargin": 0.15 if groups else 0.1},
    )
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    x_title = "Benchmark scenario"
    y_title = "95% CL limit on {} ({})".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex(xsec_unit),
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), n, -0.5, n - 0.5)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"Ndivisions": n, "TitleOffset": 1.7 if groups else 1.1},
    )
    draw_objs.append((h_dummy, "HIST"))

    # benchmark labels
    for i, d in enumerate(data):
        h_dummy.GetXaxis().SetBinLabel(i + 1, to_root_latex(make_tuple(d["name"])[0]))

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(name in br_hh_colors.root for name in names):
        _color_sequence = [br_hh_colors.root[name] for name in names]

    # helper to read values into graphs
    def create_graph(index, key):
        args = x, y, x_err_d, x_err_u, y_err_d, y_err_u = [n * [0.] for _ in range(6)]
        for i, d in enumerate(data):
            if key not in d:
                y[i] = -1.e5
                continue
            x[i] = i - 0.5
            x_err_u[i] = 1.0
            y[i] = d[key][index] if key == "observed" else d[key][index][0]
        return create_tgraph(n, *args)

    # central values
    for i, (name, col) in enumerate(zip(names, _color_sequence[:n_graphs])):
        # prepare graphs
        g_exp = create_graph(i, "expected")
        r.setup_graph(
            g_exp,
            color=colors[col],
            props={"LineWidth": 2, "LineStyle": 2},
        )
        draw_objs.append((g_exp, "SAME,EZ"))
        legend_entries.append((g_exp, to_root_latex(br_hh_names.get(name, name)), "L"))

        # observed values
        if has_obs:
            g_obs = create_graph(i, "observed")
            r.setup_graph(
                g_obs,
                color=colors[col],
                props={"LineWidth": 2, "LineStyle": 1},
            )
            draw_objs.append((g_obs, "SAME,EZ"))

    # add additional legend entries to distinguish expected and observed lines
    for _ in range(3 - len(legend_entries) % 3):
        legend_entries.append((h_dummy, " ", "L"))
    g_exp_dummy = g_exp.Clone()
    r.apply_properties(g_exp_dummy, {"LineColor": colors.black})
    legend_entries.append((g_exp_dummy, "Median expected", "L"))
    if has_obs:
        g_obs_dummy = g_obs.Clone()
        r.apply_properties(g_obs_dummy, {"LineColor": colors.black})
        legend_entries.append((g_obs_dummy, "Observed", "L"))
    else:
        legend_entries.append((h_dummy, " ", "L"))

    # texts over and lines between groups
    group_names = list(groups)
    for i, group_name in enumerate(group_names):
        # label
        x = sum(len(groups[group_names[j]]) for j in range(i + 1))
        y_offset = 0.1
        label = ROOT.TLatex(
            x - 0.5 * (1 + len(groups[group_name])),
            (y_min * (y_min / y_max)**y_offset) if y_log else (y_min - y_offset * (y_max - y_min)),
            to_root_latex(group_name),
        )
        r.setup_latex(label, props={"NDC": False, "TextAlign": 21, "TextSize": 20})
        draw_objs.append(label)

        # separating line
        if i < len(groups) - 1:
            line = ROOT.TLine(x - 0.5, y_min, x - 0.5, y_max_value)
            r.setup_line(
                line,
                props={"LineColor": colors.black, "LineStyle": 1, "NDC": False},
            )
            draw_objs.append(line)

    # legend
    legend_cols = int(math.ceil(len(legend_entries) / 3.0))
    legend = r.routines.create_legend(
        pad=pad,
        width=legend_cols * 180,
        y2=-20,
        n=3,
        props={"NColumns": legend_cols, "TextSize": 18},
    )
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "trl",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70},
    )
    draw_objs.insert(-1, legend_box)

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout=cms_layout,
    )
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
    for path in make_list(paths):
        canvas.SaveAs(path)
