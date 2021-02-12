# coding: utf-8

"""
Goodness-of-fit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, try_int, linspace, create_tgraph
from dhi.plots.styles import use_style


colors = colors.root


@use_style("dhi_default")
def plot_gof_distribution(
    path,
    data,
    toys,
    algorithm,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing the goodness-of-fit value *data* between simulated events and real data
    alognside those values computed for *toys* and saves it at *path*. The name of the *algorithm*
    used for the test is shown in the legend.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/gof.html#testing-a-datacard
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check values
    toys = list(toys)

    # set default ranges
    if x_min is None:
        x_min = min([data] + toys) / 1.2
    if x_max is None:
        x_max = max([data] + toys) * 1.2

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas()
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    h_dummy = ROOT.TH1F("dummy", ";Test statistic;Entries", 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # create the toy histogram
    h_toys = ROOT.TH1F("h_toys", "", 32, x_min, x_max)
    r.setup_hist(h_toys, props={"LineWidth": 2})
    for v in toys:
        h_toys.Fill(v)
    y_max_value = h_toys.GetMaximum()
    draw_objs.append((h_toys, "SAME,HIST"))
    legend_entries.append((h_toys, "{} toys ({})".format(len(toys), algorithm)))

    # make a simple gaus fit and determine how many stddevs the data point is off
    fit_res = h_toys.Fit("gaus", "QEMNS").Get()
    toy_ampl = fit_res.Parameter(0)
    toy_mean = fit_res.Parameter(1)
    toy_stddev = fit_res.Parameter(2)
    data_pull = abs(data - toy_mean) / toy_stddev

    # draw the fit
    f_fit = ROOT.TF1("fit", "{}*exp(-0.5*((x-{})/{})^2)".format(toy_ampl, toy_mean, toy_stddev),
        x_min, x_max)
    r.setup_func(f_fit, props={})
    draw_objs.append((f_fit, "SAME"))

    # set limits
    if y_min is None:
        y_min = 0.
    if y_max is None:
        y_max = 1.35 * (y_max_value - y_min)
    y_max_line = y_max / 1.35 + y_min
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # vertical data line
    line_data = ROOT.TLine(data, y_min, data, y_max_line)
    r.setup_line(line_data, props={"NDC": False, "LineWidth": 2, "LineStyle": 7},
        color=colors.blue_signal)
    draw_objs.append(line_data)
    legend_entries.append((line_data, "Data (#Delta = {:.1f} #sigma)".format(data_pull), "L"))

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=250, n=2)
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

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
def plot_gofs(
    path,
    data,
    algorithm,
    show_histograms=True,
    x_min=None,
    x_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing the results of multiple goodness-of-fit tests and saves it at *path*.
    Each entry in *data* should contain the fields "name", "data" (the test statistic value
    corresponding to real data) and "toys" (a list of test statistic values for toys). The name of
    the *algorithm* used for the test is shown in the legend. Per entry, a gaussion fit is drawn
    to visualize the distribution of toys. When *show_histograms*, a small histogram is shown in
    addition.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis ranges and default to the range of the
    given values. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/gof.html#testing-multiple-datacards
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check inputs
    n = len(data)
    for d in data:
        assert("name" in d)
        assert("data" in d)
        assert("toys" in d)

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = 150  # pixels
    entry_height = 90  # pixels
    head_space = 130  # pixels

    # get the canvas height
    canvas_height = n * entry_height + head_space + top_margin + bottom_margin

    # get relative pad margins and fill into props
    pad_margins = {
        "TopMargin": float(top_margin) / canvas_height,
        "BottomMargin": float(bottom_margin) / canvas_height,
        "LeftMargin": float(left_margin) / canvas_width,
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

    # make gaussian fits and store results
    fit_data = []
    for i, d in enumerate(data):
        # make a simple gaus fit and determine how many stddevs the data point is off
        h_toys = ROOT.TH1F("h_toys_{}".format(i), "", 32, min(d["toys"]), max(d["toys"]))
        for v in d["toys"]:
            h_toys.Fill(v)
        fit_res = h_toys.Fit("gaus", "QEMNS").Get()
        fd = {}
        fd["amplitude"] = fit_res.Parameter(0)
        fd["mean"] = fit_res.Parameter(1)
        fd["stddev"] = fit_res.Parameter(2)
        fd["data_diff"] = d["data"] - fd["mean"]
        fd["data_pull"] = fd["data_diff"] / fd["stddev"]
        fit_data.append(fd)

    # default x range
    max_stddev = max(fd["stddev"] for fd in fit_data)
    min_data_diff = min(fd["data_diff"] for fd in fit_data)
    max_data_diff = max(fd["data_diff"] for fd in fit_data)
    if x_min is None:
        x_min = min(-3 * max_stddev, min_data_diff * 1.25)
    if x_max is None:
        x_max = max(3 * max_stddev, max_data_diff * 1.25)

    # dummy histogram to control axes
    h_dummy = ROOT.TH1F("dummy", ";#Delta Test statistic (t - #tilde{t}_{toys});", 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    r.setup_x_axis(h_dummy.GetXaxis(), pad=pad, props={"TitleOffset": 1.2,
        "LabelOffset": r.pixel_to_coord(canvas, y=4)})
    h_dummy.GetYaxis().SetBinLabel(1, "")
    draw_objs.append((h_dummy, "HIST"))
    y_label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{#Delta = %.1f#sigma}}"

    # draw curves and data lines
    for i, (d, fd) in enumerate(zip(data, fit_data)):
        y_offset = n - 1 - i

        # underlying horizontal line
        if i < n:
            h_line = ROOT.TLine(x_min, i + 1, x_max, i + 1)
            r.setup_line(h_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
            draw_objs.append(h_line)

        # stylized histograms
        if show_histograms:
            h_toys = ROOT.TH1F("h_toys_{}".format(i), "", 32, x_min, x_max)
            r.setup_hist(h_toys)
            for v in d["toys"]:
                h_toys.Fill(v - fd["mean"])
            h_toys.Scale(0.75 / h_toys.GetMaximum())
            for b in range(1, h_toys.GetNbinsX() + 1):
                h_toys.SetBinContent(b, h_toys.GetBinContent(b) + y_offset)
            draw_objs.append((h_toys, "SAME,HIST"))

        # create a vertically shifted, centrally aligned gaus curve
        f_fit = ROOT.TF1("fit", "{} + 0.75*exp(-0.5*((x)/{})^2)".format(y_offset, fd["stddev"]),
            x_min, x_max)
        r.setup_func(f_fit, props={})
        draw_objs.append((f_fit, "SAME"))
        if i == 0:
            legend_entries.append((f_fit, "{} toys ({})".format(len(d["toys"]), algorithm), "L"))

        # data lines as graphs
        g_data = create_tgraph(1, d["data"] - fd["mean"], y_offset, 0, 0, 0, 1)
        r.setup_graph(g_data, props={"LineWidth": 2, "LineStyle": 7, "MarkerStyle": 20,
            "MarkerSize": 0}, color=colors.blue_signal)
        draw_objs.append((g_data, "SAME,EZ"))
        if i == 0:
            legend_entries.append((g_data, "Data"))

        # name labels on the y-axis
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        label = y_label_tmpl % (d["name"], abs(fd["data_pull"]))
        label_x = r.get_x(10, canvas)
        label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        label = ROOT.TLatex(label_x, label_y, label)
        r.setup_latex(label, props={"NDC": True, "TextAlign": 12, "TextSize": 22})
        draw_objs.append(label)

        # left and right ticks
        tick_length = 0.03 * (x_max - x_min)
        tl = ROOT.TLine(x_min, i + 1, x_min + tick_length, i + 1)
        tr = ROOT.TLine(x_max - tick_length, i + 1, x_max, i + 1)
        r.setup_line(tl, props={"NDC": False, "LineWidth": 1})
        r.setup_line(tr, props={"NDC": False, "LineWidth": 1})
        draw_objs.extend([tl, tr])

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=250, n=len(legend_entries))
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tlr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

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
