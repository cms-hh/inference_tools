# coding: utf-8

"""
Goodness-of-fit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, try_int, create_tgraph
from dhi.plots.styles import use_style


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
    toys = remove_nans_and_outliers(list(toys))

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
    legend_entries.append((h_toys, "{} toys ({})".format(len(toys), algorithm)))

    # make a simple gaus fit
    toy_mean, toy_stddev = fit_toys_gaus(toys, x_min=x_min, x_max=x_max, n_bins=n_bins)
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
    y_max_line = y_max / 1.35 + y_min
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # vertical data line
    line_data = ROOT.TLine(data, y_min, data, y_max_line)
    r.setup_line(line_data, props={"NDC": False, "LineWidth": 2, "LineStyle": 2},
        color=colors.blue_signal)
    draw_objs.append(line_data)
    delta_label = "< 0.01" if data_pull < 0.005 else "= {:.2f}".format(data_pull)
    legend_entries.append((line_data, "Data (#Delta {} #sigma)".format(delta_label), "L"))

    # model parameter labels
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
        d["toys"] = remove_nans_and_outliers(list(d["toys"]))

    # get the number of toys per entry
    n_toys = [len(d["toys"]) for d in data]
    n_toys_even = len(set(n_toys)) == 1

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
        mean, stddev = fit_toys_gaus(d["toys"], n_bins=n_bins)
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
    fit_label_tmpl = "#splitline{#mu = %.1f}{#sigma = %.1f}"
    if not n_toys_even:
        fit_label_tmpl = "#splitline{{N = %d}}{{{}}}".format(fit_label_tmpl)

    # vertical line at 1
    v_line = ROOT.TLine(0, 0, 0, n)
    r.setup_line(v_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
    draw_objs.append(v_line)

    # draw curves and data lines
    for i, (d, fd) in enumerate(zip(data, fit_data)):
        y_offset = n - 1 - i

        # underlying horizontal line
        if i < n:
            h_line = ROOT.TLine(x_min, i + 1, x_max, i + 1)
            r.setup_line(h_line, props={"NDC": False, "LineWidth": 1}, color=colors.light_grey)
            draw_objs.append(h_line)

        # fit stats label
        fit_label_x = r.get_x(84, pad, anchor="right")
        fit_label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        fit_label_data = (fd["mean"], fd["stddev"])
        if not n_toys_even:
            fit_label_data = (len(d["toys"]),) + fit_label_data
        fit_label = fit_label_tmpl % fit_label_data
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
            toy_label = "{} toys".format(len(d["toys"])) if n_toys_even else "Toys"
            legend_entries.append((f_fit, "{} ({})".format(toy_label, algorithm), "L"))

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

    # model parameter labels
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


def remove_nans_and_outliers(toys, sigma_outlier=10.):
    # remove nans
    valid_toys = [t for t in toys if not math.isnan(t)]
    if len(valid_toys) != len(toys):
        print("\nWARNING: found {} nan's in toy values\n".format(len(toys) - len(valid_toys)))
        toys = valid_toys

    # remove outliers using distances from normalized medians
    median = np.median(toys)
    diffs = [abs(t - median) for t in toys]
    norm = np.median(diffs)
    valid_toys = [t for t, diff in zip(toys, diffs) if diff / norm < sigma_outlier]
    if len(valid_toys) != len(toys):
        print("\nWARNING: found {} outliers in toy values\n".format(len(toys) - len(valid_toys)))
        toys = valid_toys

    return toys


def fit_toys_gaus(toys, x_min=None, x_max=None, n_bins=32):
    from plotlib.util import create_random_name
    ROOT = import_ROOT()

    # compute the initial range
    x_min_value = min(toys)
    x_max_value = max(toys)
    if x_min is None:
        x_min = x_min_value - 0.2 * (x_max_value - x_min_value)
    if x_max is None:
        x_max = x_max_value + 0.2 * (x_max_value - x_min_value)

    # create a first histogram and perform a fit
    h1 = ROOT.TH1F(create_random_name(), "", n_bins, x_min, x_max)
    for t in toys:
        h1.Fill(t)
    fit1 = h1.Fit("gaus", "QEMNS").Get()
    if not fit1:
        raise Exception("initial toy fit failed")

    # compute the range again to cover 3 sigma of the initial fit and repeat
    mean = fit1.Parameter(1)
    stddev = fit1.Parameter(2)
    x_min2 = mean - 3 * stddev
    x_max2 = mean + 3 * stddev
    h2 = ROOT.TH1F(create_random_name(), "", n_bins, x_min2, x_max2)
    for t in toys:
        h2.Fill(t)
    fit2 = h2.Fit("gaus", "QEMNS").Get()

    # return mean and stddev
    return fit2.Parameter(1), fit2.Parameter(2)


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
