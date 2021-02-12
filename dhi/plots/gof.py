# coding: utf-8

"""
Goodness-of-fit plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors
from dhi.util import import_ROOT, to_root_latex, try_int
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
    r.setup_line(line_data, props={"NDC": False, "LineWidth": 2, "LineStyle": 7}, color=colors.red)
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
