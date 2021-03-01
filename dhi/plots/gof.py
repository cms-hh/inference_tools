# coding: utf-8

"""
Goodness-of-fit plots using ROOT.
"""

import math

import numpy as np
import scipy.stats

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, try_int, create_tgraph
from dhi.plots.util import use_style, draw_model_parameters


colors = colors.root


@use_style("dhi_default")
def plot_gof_distribution(
    path,
    data,
    toys,
    algorithm,
    n_bins=32,
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

    # check values
    toys, n_valid_toys = remove_nans_and_outliers(list(toys))

    # set default ranges
    x_min_value = min([data] + toys)
    x_max_value = max([data] + toys)
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
    legend_entries.append((h_toys, "{} toys ({})".format(n_valid_toys, algorithm), "L"))

    # make a simple gaus fit
    toy_mean, toy_stddev = scipy.stats.norm.fit(toys)
    toy_ampl = fit_amplitude_gaus(h_toys, toy_mean, toy_stddev)
    data_pull = abs(data - toy_mean) / toy_stddev

    # draw the fit
    formula = "{} * exp(-0.5 * ((x - {}) / {})^2)".format(toy_ampl, toy_mean, toy_stddev)
    f_fit = ROOT.TF1("fit", formula, x_min, x_max)
    r.setup_func(f_fit, props={})
    draw_objs.append((f_fit, "SAME"))

    # set limits
    if y_min is None:
        y_min = 0.
    if y_max is None:
        y_max = 1.35 * (y_max_value - y_min)
    y_max_line = y_max / 1.4 + y_min
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # vertical data line
    line_data = ROOT.TLine(data, y_min, data, y_max_line)
    r.setup_line(line_data, props={"NDC": False, "LineWidth": 2, "LineStyle": 2},
        color=colors.blue_signal)
    draw_objs.append(line_data)
    delta_label = "< 0.01" if data_pull < 0.005 else "= {:.2f}".format(data_pull)
    legend_entries.append((line_data, "Data (#Delta {} #sigma)".format(delta_label), "L"))

    # legend
    legend = r.routines.create_legend(pad=pad, width=250, n=2)
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
    n_bins=32,
    x_min=-3.,
    x_max=3.,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing the results of multiple goodness-of-fit tests and saves it at *path*.
    Each entry in *data* should contain the fields "name", "data" (the test statistic value
    corresponding to real data) and "toys" (a list of test statistic values for toys). The name of
    the *algorithm* used for the test is shown in the legend. Per entry, a gaussion fit is drawn
    to visualize the distribution of toys.

    The toy histograms are drawn with *n_bins* bins. *x_min*, *x_max*, *y_min* and *y_max* define
    the axis ranges and default to the range of the given values. *model_parameters* can be a
    dictionary of key-value pairs of model parameters. *campaign* should refer to the name of a
    campaign label defined in *dhi.config.campaign_labels*.

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
        d["toys"], d["n_valid_toys"] = remove_nans_and_outliers(list(d["toys"]))

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = 150  # pixels
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

    # make gaussian fits and store results
    fit_data = []
    for i, d in enumerate(data):
        # make a simple gaus fit and determine how many stddevs the data point is off
        mean, stddev = scipy.stats.norm.fit(d["toys"])
        fd = {}
        fd["mean"] = mean
        fd["stddev"] = stddev
        fd["data_diff"] = d["data"] - mean
        fd["data_pull"] = fd["data_diff"] / stddev
        fit_data.append(fd)

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
    y_label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{#Delta = %.2f #sigma}}"
    y_label_tmpl_zero = "#splitline{#bf{%s}}{#scale[0.75]{#Delta < 0.01 #sigma}}"
    fit_label_tmpl = "#splitline{#splitline{N = %d}{#mu = %.1f}}{#sigma = %.1f}"

    # vertical line at 1
    v_line = ROOT.TLine(0, 0, 0, n)
    r.setup_line(v_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
    draw_objs.append(v_line)

    # horizontal lines
    for i in range(n):
        h_line = ROOT.TLine(x_min, i + 1, x_max, i + 1)
        r.setup_line(h_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
        draw_objs.append(h_line)

    # draw curves and data lines
    for i, (d, fd) in enumerate(zip(data, fit_data)):
        y_offset = n - 1 - i

        # fit stats label
        fit_label_x = r.get_x(84, pad, anchor="right")
        fit_label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        fit_label = fit_label_tmpl % (d["n_valid_toys"], fd["mean"], fd["stddev"])
        fit_label = ROOT.TLatex(fit_label_x, fit_label_y, fit_label)
        r.setup_latex(fit_label, props={"NDC": True, "TextAlign": 12, "TextSize": 16})
        draw_objs.append(fit_label)

        # stylized histograms
        h_toys = ROOT.TH1F("h_toys_{}".format(i), "", n_bins, x_min, x_max)
        r.setup_hist(h_toys)
        for v in d["toys"]:
            h_toys.Fill((v - fd["mean"]) / fd["stddev"])
        h_scale = 0.75 / h_toys.GetMaximum()
        h_toys.Scale(h_scale)
        ampl = fit_amplitude_gaus(h_toys, 0, 1)
        for b in range(1, h_toys.GetNbinsX() + 1):
            h_toys.SetBinContent(b, h_toys.GetBinContent(b) + y_offset)
        draw_objs.append((h_toys, "SAME,HIST"))

        # create a common gaus curve to visualize the fit quality
        formula = "{} + {} * exp(-0.5 * x^2)".format(y_offset, ampl)
        f_fit = ROOT.TF1("fit", formula, x_min, x_max)
        r.setup_func(f_fit)
        draw_objs.append((f_fit, "SAME"))
        if i == 0:
            legend_entries.append((f_fit, "Toys ({})".format(algorithm), "L"))

        # data lines as graphs
        g_data = create_tgraph(1, (d["data"] - fd["mean"]) / fd["stddev"], y_offset, 0, 0, 0, 1)
        r.setup_graph(g_data, props={"LineWidth": 2, "LineStyle": 2}, color=colors.blue_signal)
        draw_objs.append((g_data, "SAME,EZ"))
        if i == 0:
            legend_entries.append((g_data, "Data", "L"))

        # name labels on the y-axis
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        if abs(fd["data_pull"]) < 0.005:
            label = y_label_tmpl_zero % (label,)
        else:
            label = y_label_tmpl % (label, abs(fd["data_pull"]))
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


def remove_nans_and_outliers(toys, sigma_outlier=10.):
    # remove nans
    valid_toys = [t for t in toys if not math.isnan(t)]
    if len(valid_toys) != len(toys):
        print("\nWARNING: found {} nan's in toy values\n".format(len(toys) - len(valid_toys)))
        toys = valid_toys

    # store the number of valid toys
    n_valid = len(toys)

    # remove outliers using distances from normalized medians
    median = np.median(toys)
    diffs = [abs(t - median) for t in toys]
    norm = np.median(diffs)
    central_toys = [t for t, diff in zip(toys, diffs) if diff / norm < sigma_outlier]
    if len(central_toys) != len(toys):
        print("\nWARNING: found {} outliers in toy values\n".format(len(toys) - len(central_toys)))
        toys = central_toys

    return toys, n_valid


def fit_amplitude_gaus(hist, mean, stddev):
    from plotlib.util import create_random_name
    ROOT = import_ROOT()

    # create the function and set an initial guess
    f = ROOT.TF1(create_random_name(), "[0] * exp(-0.5 * ((x - {}) / {})^2)".format(mean, stddev),
        hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax())
    f.SetParLimits(0, 0., 10. * hist.GetMaximum())

    # perform the fit
    fit = hist.Fit(f.GetName(), "QEMNS").Get()
    if not fit:
        raise Exception("amplitude fit failed")

    return fit.Parameter(0)
