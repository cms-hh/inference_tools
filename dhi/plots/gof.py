# coding: utf-8

"""
Goodness-of-fit plots using ROOT.
"""

import math

import numpy as np
import scipy.stats

from dhi.config import campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, create_tgraph
from dhi.plots.util import use_style, draw_model_parameters, get_y_range


colors = colors.root


@use_style("dhi_default")
def plot_gof_distribution(
    path,
    data,
    toys,
    algorithm,
    n_bins=30,
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

    The toy histogram is drawn with *n_bins* bins. *x_min*, *x_max*, *y_min* and *y_max* define the
    axis ranges and default to the range of the given values. *model_parameters* can be a dictionary
    of key-value pairs of model parameters. *campaign* should refer to the name of a campaign label
    defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/gof.html#testing-a-datacard
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # remove nans and outliers
    toys = remove_nans(list(toys))
    central_toys = remove_outliers(list(toys))

    # set default ranges
    x_min_value = min([data] + central_toys)
    x_max_value = max([data] + central_toys)
    if x_min is None:
        x_min = x_min_value - 0.2 * (x_max_value - x_min_value)
    if x_max is None:
        x_max = x_max_value + 0.2 * (x_max_value - x_min_value)

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
    h_toys = ROOT.TH1F("h_toys", "", n_bins, x_min, x_max)
    r.setup_hist(h_toys, props={"LineWidth": 2})
    for v in toys:
        h_toys.Fill(v)
    y_max_value = h_toys.GetMaximum()
    draw_objs.append((h_toys, "SAME,HIST"))
    legend_entries.append((h_toys, "{} toys ({})".format(len(toys), algorithm), "L"))

    # set limits
    y_min, y_max, y_max_line = get_y_range(0., y_max_value, y_min, y_max)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # vertical data line
    line_data = ROOT.TLine(data, y_min, data, y_max_line)
    r.setup_line(line_data, props={"NDC": False, "LineWidth": 3, "LineStyle": 2},
        color=colors.blue_signal)
    draw_objs.append(line_data)
    legend_entries.append((line_data, "Data", "L"))

    # integration graph
    g_int = create_integration_graph(h_toys, data)
    prob = (np.array(toys) >= data).mean()
    r.setup_graph(g_int, props={"FillStyle": 3345, "FillColor": colors.blue_signal, "LineWidth": 0})
    draw_objs.insert(-1, (g_int, "SAME,02"))
    legend_entries.append((g_int, "p = {:.1f} %".format(prob * 100), "F"))

    # legend
    legend = r.routines.create_legend(pad=pad, width=250, n=len(legend_entries))
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tr",
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
def plot_gofs(
    path,
    data,
    algorithm,
    n_bins=30,
    x_min=-3.,
    x_max=3.,
    left_margin=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing the results of multiple goodness-of-fit tests and saves it at *path*.
    Each entry in *data* should contain the fields "name", "data" (the test statistic value
    corresponding to real data) and "toys" (a list of test statistic values for toys). The name of
    the *algorithm* used for the test is shown in the legend.

    The toy histograms are drawn with *n_bins* bins. *x_min*, *x_max*, *y_min* and *y_max* define
    the axis ranges and default to the range of the given values. *left_margin* controls the left
    margin of the pad in pixels. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

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
        d["toys"] = remove_nans(list(d["toys"]))
        d["central_toys"] = remove_outliers(list(d["toys"]))

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = left_margin or 150  # pixels
    entry_height = 90  # pixels
    head_space = 100  # pixels

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

    # default x range
    if x_min is None:
        x_min = -3.
    if x_max is None:
        x_max = 3.

    # dummy histogram to control axes
    h_dummy = ROOT.TH1F("dummy", ";Normalized test statistic (t - #mu_{toys}) / #sigma_{toys};",
        1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    r.setup_x_axis(h_dummy.GetXaxis(), pad=pad, props={"TitleOffset": 1.2,
        "LabelOffset": r.pixel_to_coord(canvas, y=4)})
    h_dummy.GetYaxis().SetBinLabel(1, "")
    draw_objs.append((h_dummy, "HIST"))
    y_label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{p = %.1f %%}}"
    stats_label_tmpl = "#splitline{#splitline{N = %d}{#mu = %.1f}}{#sigma = %.1f}"

    # vertical line at 1
    v_line = ROOT.TLine(0, 0, 0, n)
    r.setup_line(v_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
    draw_objs.append(v_line)

    # horizontal lines
    for i in range(n):
        h_line = ROOT.TLine(x_min, i + 1, x_max, i + 1)
        r.setup_line(h_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
        draw_objs.append(h_line)

    # draw histograms and data lines
    for i, d in enumerate(data):
        y_offset = n - 1 - i

        # stats label
        mean, stddev = scipy.stats.norm.fit(d["central_toys"])
        stats_label_x = r.get_x(84, pad, anchor="right")
        stats_label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        stats_label = stats_label_tmpl % (len(d["toys"]), mean, stddev)
        stats_label = ROOT.TLatex(stats_label_x, stats_label_y, stats_label)
        r.setup_latex(stats_label, props={"NDC": True, "TextAlign": 12, "TextSize": 16})
        draw_objs.append(stats_label)

        # stylized histograms
        h_toys = ROOT.TH1F("h_toys_{}".format(i), "", n_bins, x_min, x_max)
        r.setup_hist(h_toys)
        for v in d["toys"]:
            h_toys.Fill((v - mean) / stddev)
        h_toys.Scale(0.85 / h_toys.GetMaximum())
        for b in range(1, h_toys.GetNbinsX() + 1):
            h_toys.SetBinContent(b, h_toys.GetBinContent(b) + y_offset)
        draw_objs.append((h_toys, "SAME,HIST"))
        if i == 0:
            legend_entries.append((h_toys, "Toys ({})".format(algorithm), "L"))

        # integration graph
        g_int = create_integration_graph(h_toys, (d["data"] - mean) / stddev, y_offset=y_offset)
        r.setup_graph(g_int, props={"FillStyle": 3345, "FillColor": colors.blue_signal,
            "LineWidth": 0})
        draw_objs.insert(-1, (g_int, "SAME,02"))

        # data lines as graphs
        g_data = create_tgraph(1, (d["data"] - mean) / stddev, y_offset, 0, 0, 0, 1)
        r.setup_graph(g_data, props={"LineWidth": 3, "LineStyle": 2}, color=colors.blue_signal)
        draw_objs.append((g_data, "SAME,EZ"))
        if i == 0:
            legend_entries.append((g_data, "Data", "L"))

        # name labels on the y-axis
        prob = (np.array(d["toys"]) > d["data"]).mean()
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        label = y_label_tmpl % (label, prob * 100.)
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

    # legend
    legend = r.routines.create_legend(pad=pad, width=250, n=len(legend_entries))
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "tlr",
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


def remove_nans(toys):
    # remove nans
    valid_toys = [t for t in toys if not math.isnan(t)]
    if len(valid_toys) != len(toys):
        print("\nWARNING: found {} nan's in toy values\n".format(len(toys) - len(valid_toys)))

    return valid_toys


def remove_outliers(toys, sigma_outlier=10.):
    # remove outliers using distances from normalized medians
    median = np.median(toys)
    diffs = [abs(t - median) for t in toys]
    norm = np.median(diffs)
    central_toys = [t for t, diff in zip(toys, diffs) if diff / norm < sigma_outlier]
    if len(central_toys) != len(toys):
        print("\nWARNING: found {} outliers in toy values\n".format(len(toys) - len(central_toys)))
        toys = central_toys

    return central_toys


def create_integration_graph(hist, data, y_offset=0):
    # get the index of the bin in hist that contains the data point
    x_axis = hist.GetXaxis()
    for b_data in range(1, x_axis.GetNbins() + 1):
        if x_axis.GetBinLowEdge(b_data) <= data < x_axis.GetBinLowEdge(b_data + 1):
            break
    else:
        return create_tgraph(1, -1e6, 0)

    # create a graph with errors that mimics the area under the histogram
    x_values, x_widths, y_heights = [], [], []
    # add the bin that was hit by the data point
    x_values.append(data)
    x_widths.append(x_axis.GetBinLowEdge(b_data + 1) - data)
    y_heights.append(hist.GetBinContent(b_data) - y_offset)
    # fill the remaining bins
    for b in range(b_data + 1, x_axis.GetNbins() + 1):
        x_values.append(x_axis.GetBinLowEdge(b))
        x_widths.append(x_axis.GetBinWidth(b))
        y_heights.append(hist.GetBinContent(b) - y_offset)

    # create the actual graph
    g = create_tgraph(len(x_values), x_values, y_offset, 0, x_widths, 0, y_heights)

    return g
