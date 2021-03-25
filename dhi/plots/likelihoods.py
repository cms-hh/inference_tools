# coding: utf-8

"""
Likelihood plots using ROOT.
"""

import math
from collections import OrderedDict

import numpy as np
import scipy.interpolate
import scipy.optimize
from scinum import Number

from dhi.config import (
    poi_data, br_hh_names, campaign_labels, chi2_levels, colors, color_sequence, marker_sequence,
)
from dhi.util import import_ROOT, to_root_latex, create_tgraph, DotDict, minimize_1d, multi_match
from dhi.plots.util import (
    use_style, draw_model_parameters, fill_hist_from_points, create_random_name, get_contours,
)


colors = colors.root


@use_style("dhi_default")
def plot_likelihood_scan_1d(
    path,
    poi,
    values,
    theory_value=None,
    poi_min=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a likelihood plot of the 1D scan of a *poi* and saves it at *path*. *values* should be a
    mapping to lists of values or a record array with keys "<poi_name>" and "dnll2". *theory_value*
    can be a 3-tuple denoting the nominal theory prediction of the POI and its up and down
    uncertainties which is drawn as a vertical bar. When *poi_min* is set, it should be the value of
    the poi that leads to the best likelihood. Otherwise, it is estimated from the interpolated
    curve.

    *x_min* and *x_max* define the x-axis range of POI, and *y_min* and *y_max* control the range of
    the y-axis. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#1d
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get valid poi and delta nll values
    poi_values = np.array(values[poi], dtype=np.float32)
    dnll2_values = np.array(values["dnll2"], dtype=np.float32)

    # set x range
    if x_min is None:
        x_min = min(poi_values)
    if x_max is None:
        x_max = max(poi_values)

    # select valid points
    mask = ~np.isnan(dnll2_values)
    poi_values = poi_values[mask]
    dnll2_values = dnll2_values[mask]

    # set y range
    y_max_value = max(dnll2_values[(poi_values >= x_min) & (poi_values <= x_max)])
    if y_log:
        if y_min is None:
            y_min = 1e-3
        if y_max is None:
            y_max = y_min * 10**(math.log10(y_max_value / y_min) * 1.35)
        y_max_line = y_min * 10**(math.log10(y_max / y_min) / 1.4)
    else:
        if y_min is None:
            y_min = 0.
        if y_max is None:
            y_max = 1.35 * (y_max_value - y_min)
        y_max_line = y_max / 1.4 + y_min

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_1d(poi_values, dnll2_values, poi_min=poi_min)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi].label)
    y_title = "-2 #Delta log(L)"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    draw_objs.append((h_dummy, "HIST"))

    # 1 and 2 sigma indicators
    for value in [scan.poi_p1, scan.poi_m1, scan.poi_p2, scan.poi_m2]:
        if value is not None:
            line = ROOT.TLine(value, y_min, value, scan.interp(value))
            r.setup_line(line, props={"LineColor": colors.black, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # lines at chi2_1 intervals
    for n in [chi2_levels[1][1], chi2_levels[1][2]]:
        if n < y_max_line:
            line = ROOT.TLine(x_min, n, x_max, n)
            r.setup_line(line, props={"LineColor": colors.black, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # theory prediction with uncertainties
    if theory_value:
        has_thy_err = len(theory_value) == 3
        if has_thy_err:
            # theory graph
            g_thy = create_tgraph(1, theory_value[0], y_min, theory_value[2], theory_value[1],
                0, y_max_line)
            r.setup_graph(g_thy, props={"LineColor": colors.red, "FillStyle": 1001,
                "FillColor": colors.red_trans_50})
            draw_objs.append((g_thy, "SAME,02"))
            legend_entries.append((g_thy, "Theory prediction", "LF"))
        # theory line
        line_thy = ROOT.TLine(theory_value[0], y_min, theory_value[0], y_max_line)
        r.setup_line(line_thy, props={"NDC": False}, color=colors.red)
        draw_objs.append(line_thy)
        if not has_thy_err:
            legend_entries.append((line_thy, "Theory prediction", "L"))

    # line for best fit value
    line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_line)
    r.setup_line(line_fit, props={"LineWidth": 2, "NDC": False}, color=colors.black)
    fit_label = "{} = {}".format(to_root_latex(poi_data[poi].label),
        scan.num_min.str(format="%.2f", style="root"))
    draw_objs.append(line_fit)
    legend_entries.insert(0, (line_fit, fit_label, "L"))

    # nll curve
    g_nll = create_tgraph(len(poi_values), poi_values, dnll2_values)
    r.setup_graph(g_nll, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 0.75})
    draw_objs.append((g_nll, "SAME,CP"))

    # legend
    legend = r.routines.create_legend(pad=pad, width=230, n=len(legend_entries))
    r.setup_legend(legend)
    for tpl in legend_entries:
        legend.AddEntry(*tpl)
    draw_objs.append(legend)

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
def plot_likelihood_scans_1d(
    path,
    poi,
    data,
    theory_value=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
):
    """
    Plots multiple curves of 1D likelihood scans of a POI *poi1* and *poi2*, and saves it at *path*.
    All information should be passed as a list *data*. Entries must be dictionaries with the
    following content:

        - "values": A mapping to lists of values or a record array with keys "<poi1_name>",
          "<poi2_name>" and "dnll2".
        - "poi_min": A float describing the best fit value of the POI. When not set, the minimum is
          estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    *theory_value* can be a 3-tuple denoting the nominal theory prediction of the POI and its up and
    down uncertainties which is drawn as a vertical bar.

    *x_min* and *x_max* define the x-axis range of POI, and *y_min* and *y_max* control the range of
    the y-axis. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. When
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#1d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # validate data entries
    for i, d in enumerate(data):
        # convert likelihood values to arrays
        assert("values" in d)
        values = d["values"]
        if isinstance(values, np.ndarray):
            values = {k: values[k] for k in values.dtype.names}
        assert(poi in values)
        assert("dnll2" in values)
        # keep only valid points
        values = {k: np.array(v, dtype=np.float32) for k, v in values.items()}
        mask = ~np.isnan(values["dnll2"])
        values[poi] = values[poi][mask]
        values["dnll2"] = values["dnll2"][mask]
        d["values"] = values
        # check poi minimum
        d.setdefault("poi_min", None)
        # default name
        d.setdefault("name", str(i + 1))

    # set x range
    if x_min is None:
        x_min = min([min(d["values"][poi]) for d in data])
    if x_max is None:
        x_max = max([max(d["values"][poi]) for d in data])

    # set y range
    y_max_value = max([
        d["values"]["dnll2"][
            (d["values"][poi] >= x_min) & (d["values"][poi] <= x_max)
        ]
        for d in data
    ])
    if y_log:
        if y_min is None:
            y_min = 1e-3
        if y_max is None:
            y_max = y_min * 10**(math.log10(y_max_value / y_min) * 1.35)
        y_max_line = y_min * 10**(math.log10(y_max / y_min) / 1.4)
    else:
        if y_min is None:
            y_min = 0.
        if y_max is None:
            y_max = 1.35 * (y_max_value - y_min)
        y_max_line = y_max / 1.4 + y_min

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi].label)
    y_title = "-2 #Delta log(L)"
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    draw_objs.append((h_dummy, "HIST"))

    # lines at chi2_1 intervals
    for n in [chi2_levels[1][1], chi2_levels[1][2]]:
        if n < y_max:
            line = ROOT.TLine(x_min, n, x_max, n)
            r.setup_line(line, props={"LineColor": colors.black, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # theory prediction with uncertainties
    if theory_value:
        has_thy_err = len(theory_value) == 3
        if has_thy_err:
            # theory graph
            g_thy = create_tgraph(1, theory_value[0], y_min, theory_value[2], theory_value[1],
                0, y_max_line)
            r.setup_graph(g_thy, props={"LineColor": colors.red, "FillStyle": 1001,
                "FillColor": colors.red_trans_50})
            draw_objs.append((g_thy, "SAME,02"))
            legend_entries.append((g_thy, "Theory prediction", "LF"))
        # theory line
        line_thy = ROOT.TLine(theory_value[0], y_min, theory_value[0], y_max_line)
        r.setup_line(line_thy, props={"NDC": False}, color=colors.red)
        draw_objs.append(line_thy)
        if not has_thy_err:
            legend_entries.append((line_thy, "Theory prediction", "L"))

    # perform scans and draw nll curves
    for d, col, ms in zip(data, color_sequence[:len(data)], marker_sequence[:len(data)]):
        # evaluate the scan, run interpolation and error estimation
        scan = evaluate_likelihood_scan_1d(d["values"][poi], d["values"]["dnll2"],
            poi_min=d["poi_min"])

        # draw the curve
        g_nll = create_tgraph(len(d["values"][poi]), d["values"][poi],
            d["values"]["dnll2"])
        r.setup_graph(g_nll, props={"LineWidth": 2, "MarkerStyle": ms, "MarkerSize": 1.2},
            color=colors[col])
        draw_objs.append((g_nll, "SAME,CP"))
        legend_entries.append((g_nll, to_root_latex(br_hh_names.get(d["name"], d["name"])), "LP"))

        # line for best fit value
        line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_line)
        r.setup_line(line_fit, props={"LineWidth": 2, "NDC": False}, color=colors[col])
        draw_objs.append(line_fit)

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
def plot_likelihood_scan_2d(
    path,
    poi1,
    poi2,
    values,
    poi1_min=None,
    poi2_min=None,
    draw_sm_point=True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a likelihood plot of the 2D scan of two POIs *poi1* and *poi2*, and saves it at *path*.
    *values* should be a mapping to lists of values or a record array with keys "<poi1_name>",
    "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be the values of
    the POIs that lead to the best likelihood. Otherwise, they are  estimated from the interpolated
    curve. The standard model point at (1, 1) as drawn as well unless *draw_sm_point* is *False*.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. *z_min* and *z_max* limit the range of the z-axis.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get poi and delta nll values
    poi1_values = np.array(values[poi1], dtype=np.float32)
    poi2_values = np.array(values[poi2], dtype=np.float32)
    dnll2_values = np.array(values["dnll2"], dtype=np.float32)

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_2d(
        poi1_values, poi2_values, dnll2_values, poi1_min=poi1_min, poi2_min=poi2_min,
    )

    # determine contours independent of plotting
    contours = get_contours(
        poi1_values,
        poi2_values,
        dnll2_values,
        levels=[chi2_levels[2][1], chi2_levels[2][2]],
        frame_kwargs=[{"mode": "edge"}],
    )

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"RightMargin": 0.17, "Logz": True})
    pad.cd()
    draw_objs = []

    # create the 2D histogram from values
    h_nll = create_dnll2_hist(poi1_values, poi2_values, dnll2_values, x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
    x_min = h_nll.GetXaxis().GetXmin()
    x_max = h_nll.GetXaxis().GetXmax()
    y_min = h_nll.GetYaxis().GetXmin()
    y_max = h_nll.GetYaxis().GetXmax()
    z_min = h_nll.GetMinimum() or 1e-3
    z_max = h_nll.GetMaximum()

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    z_title = "-2 #Delta log(L)"
    h_dummy = ROOT.TH2F("h_nll", ";{};{};{}".format(x_title, y_title, z_title),
        1, x_min, x_max, 1, y_min, y_max)
    r.setup_hist(h_dummy, pad=pad, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
    draw_objs.append((h_dummy, ""))

    # setup the nll hist
    r.setup_hist(h_nll, props={"ContourXX": 100, "Minimum": z_min, "Maximum": z_max})
    r.setup_z_axis(h_nll.GetZaxis(), pad=pad, props={"Title": z_title, "TitleOffset": 1.3})
    draw_objs.append((h_nll, "SAME,COLZ"))

    # 1 and 2 sigma contours
    for g in contours[0]:
        r.setup_graph(g, props={"LineWidth": 2, "LineColor": colors.green})
        draw_objs.append((g, "SAME,C"))
    for g in contours[1]:
        r.setup_graph(g, props={"LineWidth": 2, "LineColor": colors.yellow})
        draw_objs.append((g, "SAME,C"))

    # SM point
    if draw_sm_point:
        g_sm = create_tgraph(1, 1, 1)
        r.setup_graph(g_sm, props={"MarkerStyle": 33, "MarkerSize": 2.5}, color=colors.red)
        draw_objs.insert(-1, (g_sm, "P"))

    # best fit point
    g_fit = ROOT.TGraphAsymmErrors(1)
    g_fit.SetPoint(0, scan.num1_min(), scan.num2_min())
    if scan.num1_min.uncertainties:
        g_fit.SetPointEXhigh(0, scan.num1_min.u(direction="up"))
        g_fit.SetPointEXlow(0, scan.num1_min.u(direction="down"))
    if scan.num2_min.uncertainties:
        g_fit.SetPointEYhigh(0, scan.num2_min.u(direction="up"))
        g_fit.SetPointEYlow(0, scan.num2_min.u(direction="down"))
    r.setup_graph(g_fit, color=colors.black)
    draw_objs.append((g_fit, "PEZ"))

    # measurement and best fit value labels
    fit_label1 = "{} = {}".format(to_root_latex(poi_data[poi1].label),
        scan.num1_min.str(format="%.2f", style="root"))
    fit_label2 = "{} = {}".format(to_root_latex(poi_data[poi2].label),
        scan.num2_min.str(format="%.2f", style="root"))
    labels = [fit_label1, fit_label2]
    for i, l in enumerate(labels):
        l = r.routines.create_top_right_label(l, pad=pad, x_offset=160, y_offset=30 + i * 34,
            props={"TextAlign": 13})
        draw_objs.append(l)

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
def plot_likelihood_scans_2d(
    path,
    poi1,
    poi2,
    data,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    fill_nans=True,
    model_parameters=None,
    campaign=None,
):
    """
    Creates the likelihood contour plots of multiple 2D scans of two POIs *poi1* and *poi2*, and
    saves it at *path*. All information should be passed as a list *data*. Entries must be
    dictionaries with the following content:

        - "values": A mapping to lists of values or a record array with keys "<poi1_name>",
          "<poi2_name>" and "dnll2".
        - "poi_mins": A list of two floats describing the best fit value of the two POIs. When not
          set, the minima are estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. When *fill_nans* is *True*, points with failed
    fits, denoted by nan values, are filled with the averages of neighboring fits. When
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # validate data entries
    for i, d in enumerate(data):
        # convert likelihood values to arrays
        assert("values" in d)
        values = d["values"]
        if isinstance(values, np.ndarray):
            values = {k: values[k] for k in values.dtype.names}
        assert(poi1 in values)
        assert(poi2 in values)
        assert("dnll2" in values)
        values = {k: np.array(v, dtype=np.float32) for k, v in values.items()}
        d["values"] = values
        # check poi minima
        d["poi_mins"] = d.get("poi_mins") or [None, None]
        assert(len(d["poi_mins"]) == 2)
        # default name
        d.setdefault("name", str(i + 1))

    # determine contours independent of plotting
    contours = [
        get_contours(
            d["values"][poi1],
            d["values"][poi2],
            d["values"]["dnll2"],
            levels=[chi2_levels[2][1], chi2_levels[2][2]],
            frame_kwargs=[{"mode": "edge"}],
        )
        for d in data
    ]

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logz": True})
    pad.cd()
    legend_entries = []
    draw_objs = []

    # set ranges
    if x_min is None:
        x_min = min([min(d["values"][poi1]) for d in data])
    if x_max is None:
        x_max = max([max(d["values"][poi1]) for d in data])
    if y_min is None:
        y_min = min([min(d["values"][poi2]) for d in data])
    if y_max is None:
        y_max = max([max(d["values"][poi2]) for d in data])

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    h_dummy = ROOT.TH2F("h", ";{};{};".format(x_title, y_title), 1, x_min, x_max, 1, y_min, y_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # loop through data entries
    for d, (cont1, cont2), col in zip(data, contours, color_sequence[:len(data)]):
        # evaluate the scan
        scan = evaluate_likelihood_scan_2d(
            d["values"][poi1], d["values"][poi2], d["values"]["dnll2"],
            poi1_min=d["poi_mins"][0], poi2_min=d["poi_mins"][1],
        )

        # plot 1 and 2 sigma contours
        for g1 in cont1:
            r.setup_graph(g1, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors[col]})
            draw_objs.append((g1, "SAME,C"))
        for g2 in cont2:
            r.setup_graph(g2, props={"LineWidth": 2, "LineStyle": 2, "LineColor": colors[col]})
            draw_objs.append((g2, "SAME,C"))
        name = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        legend_entries.append((g1, name, "L"))

        # best fit point
        g_fit = create_tgraph(1, scan.num1_min(), scan.num2_min())
        r.setup_graph(g_fit, props={"MarkerStyle": 33, "MarkerSize": 2}, color=colors[col])
        draw_objs.append((g_fit, "SAME,PEZ"))

    # append legend entries to show styles
    g_fit_style = g_fit.Clone()
    g1_style = g1.Clone()
    g2_style = g2.Clone()
    r.apply_properties(g_fit_style, {"MarkerColor": colors.black})
    r.apply_properties(g1_style, {"LineColor": colors.black})
    r.apply_properties(g2_style, {"LineColor": colors.black})
    legend_entries.extend([
        (g_fit_style, "Best fit value", "P"),
        (g1_style, "#pm 1 #sigma", "L"),
        (g2_style, "#pm 2 #sigma", "L"),
    ])

    # prepend empty values
    n_empty = 3 - (len(legend_entries) % 3)
    if n_empty not in (0, 3):
        for _ in range(n_empty):
            legend_entries.insert(3 - n_empty, (h_dummy, " ", "L"))

    # legend with actual entries in different colors
    legend_cols = int(math.ceil(len(legend_entries) / 3.))
    legend_rows = min(len(legend_entries), 3)
    legend = r.routines.create_legend(pad=pad, width=legend_cols * 150, height=legend_rows * 30,
        props={"NColumns": legend_cols})
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


def plot_nuisance_likelihood_scans(
    path,
    poi,
    workspace,
    dataset,
    fit_diagnostics_path,
    fit_name="fit_s",
    skip_parameters=None,
    only_parameters=None,
    parameters_per_page=1,
    scan_points=201,
    x_min=-2.,
    x_max=2,
    y_log=False,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing the change of the negative log-likelihood, obtained *poi*, when varying
    values of nuisance paramaters and saves it at *path*. The calculation of the likelihood change
    requires the RooFit *workspace* to read the model config, a RooDataSet *dataset* to construct
    the functional likelihood, and the output file *fit_diagnostics_path* of the combine fit
    diagnostics for reading pre- and post-fit parameters for the fit named *fit_name*, defaulting
    to ``"fit_s"``.

    Nuisances to skip, or to show exclusively can be configured via *skip_parameters* and
    *only_parameters*, respectively, which can be lists of patterns. *parameters_per_page* defines
    the number of parameter curves that are drawn in the same canvas page. The scan range and
    granularity is set via *scan_points*, *x_min* and *x_max*. When *y_log* is *True*, the y-axis is
    plotted with a logarithmic scale. *model_parameters* can be a dictionary of key-value pairs of
    model parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/postfit.html#nuisance-parameter-influence-on-likelihood
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # helper to convert a RooArgSet  into a dictionary mapping names to value-errors pairs
    def convert_argset(argset):
        data = OrderedDict()
        it = argset.createIterator()
        while True:
            param = it.Next()
            if not param:
                break
            data[param.GetName()] = (param.getVal(), param.getErrorHi(), param.getErrorLo())
        return data

    # get the best fit value and prefit data from the diagnostics file
    f = ROOT.TFile(fit_diagnostics_path, "READ")
    best_fit = f.Get(fit_name)
    fit_args = best_fit.floatParsFinal()
    prefit_params = convert_argset(f.Get("nuisances_prefit"))

    # get the model config from the workspace
    model_config = workspace.genobj("ModelConfig")

    # build the nll object
    nll_args = ROOT.RooLinkedList()
    nll_args.Add(ROOT.RooFit.Constrain(model_config.GetNuisanceParameters()))
    nll_args.Add(ROOT.RooFit.Extended(model_config.GetPdf().canBeExtended()))
    nll = model_config.GetPdf().createNLL(dataset, nll_args)

    # save the best fit in a snap shot
    snapshot_name = "best_fit_parameters"
    workspace.saveSnapshot(snapshot_name, ROOT.RooArgSet(fit_args), True)

    # prepare parameters to plot, stored in groups
    param_names = [[]]
    for param_name in prefit_params:
        if only_parameters and not multi_match(param_name, only_parameters):
            continue
        if skip_parameters and multi_match(param_name, skip_parameters):
            continue
        if parameters_per_page < 1 or len(param_names[-1]) < parameters_per_page:
            param_names[-1].append(param_name)
        else:
            param_names.append([param_name])

    # prepare the scan values, ensure that 0 is contained
    scan_values = np.linspace(x_min, x_max, scan_points).tolist()
    if 0 not in scan_values:
        scan_values = sorted(scan_values + [0.])

    # go through nuisances
    canvas = None
    for _param_names in param_names:
        # setup the default style and create canvas and pad
        first_canvas = canvas is None
        r.setup_style()
        canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
        pad.cd()

        # start the multi pdf file
        if first_canvas:
            canvas.Print(path + "[")

        # get nll curves for all parameters on this page
        curve_data = []
        for param_name in _param_names:
            pre_u, pre_d = prefit_params[param_name][1:3]
            workspace.loadSnapshot(snapshot_name)
            param = workspace.var(param_name)
            if not param:
                raise Exception("parameter {} not found in workspace".format(param_name))
            param_bf = param.getVal()
            nll_base = nll.getVal()
            x_values, y_values = [], []
            for x in scan_values:
                param.setVal(param_bf + (pre_u if x >= 0 else -pre_d) * x)
                x_values.append(param.getVal())
                y_values.append(2 * (nll.getVal() - nll_base))
            curve_data.append((param_name, x_values, y_values))

        # get y range
        y_min_value = min(min(y_values) for _, _, y_values in curve_data)
        y_max_value = max(max(y_values) for _, _, y_values in curve_data)
        if y_log:
            y_min = 1.e-3
            y_max = y_min * 10**(1.35 * math.log10(y_max_value / y_min))
        else:
            y_min = y_min_value
            y_max = 1.35 * (y_max_value - y_min)

        # dummy histogram to control axes
        x_title = "(#theta - #theta_{best}) / #Delta#theta_{pre}"
        y_title = "Change in -2 log(L)"
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
        draw_objs = [(h_dummy, "HIST")]
        legend_entries = []

        # nll graphs
        for (param_name, x, y), col in zip(curve_data, color_sequence[:len(curve_data)]):
            g_nll = create_tgraph(len(x), x, y)
            r.setup_graph(g_nll, props={"LineWidth": 2, "LineStyle": 1}, color=colors[col])
            draw_objs.append((g_nll, "SAME,C"))
            legend_entries.append((g_nll, to_root_latex(param_name), "L"))

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
            draw_objs.extend(draw_model_parameters(model_parameters, pad))

        # cms label
        cms_labels = r.routines.create_cms_labels(pad=pad)
        draw_objs.extend(cms_labels)

        # campaign label
        if campaign:
            campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
            campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
            draw_objs.append(campaign_label)

        # draw objects, update and save
        r.routines.draw_objects(draw_objs)
        r.update_canvas(canvas)
        canvas.Print(path)

    # finish the pdf
    if canvas:
        canvas.Print(path + "]")


def evaluate_likelihood_scan_1d(poi_values, dnll2_values, poi_min=None):
    """
    Takes the results of a 1D likelihood profiling scan given by the *poi_values* and the
    corresponding *delta_2nll* values, performs an interpolation and returns certain results of the
    scan in a dict. When *poi_min* is *None*, it is estimated from the interpolated curve.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``poi_min``: The poi value corresponding to the minimum delta nll value.
    - ``poi_p1``: The poi value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi_m1``: The poi value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi_p2``: The poi value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi_m2``: The poi value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``num_min``: A Number instance representing the best fit value and its 1 sigma uncertainty.
    """
    # ensure we are dealing with arrays
    poi_values = np.array(poi_values)
    dnll2_values = np.array(dnll2_values)

    # store ranges
    poi_values_min = poi_values.min()
    poi_values_max = poi_values.max()

    # remove values where dnll2 is nan
    mask = ~np.isnan(dnll2_values)
    poi_values = poi_values[mask]
    dnll2_values = dnll2_values[mask]

    # first, obtain an interpolation function
    # interp = scipy.interpolate.interp1d(poi_values, dnll2_values, kind="cubic")
    interp = scipy.interpolate.interp1d(poi_values, dnll2_values, kind="linear")

    # get the minimum when not set
    if poi_min is None:
        objective = lambda x: abs(interp(x))
        bounds = (poi_values_min + 1e-4, poi_values_max - 1e-4)
        res = minimize_1d(objective, bounds)
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi_min = res.x[0]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v):
        def minimize(bounds):
            objective = lambda x: (interp(x) - v) ** 2.0
            res = minimize_1d(objective, bounds)
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        return (
            minimize((poi_min, poi_values_max - 1e-4)),
            minimize((poi_values_min + 1e-4, poi_min)),
        )

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi_p1, poi_m1 = get_intersections(chi2_levels[1][1])
    poi_p2, poi_m2 = get_intersections(chi2_levels[1][2])

    # create a Number object wrapping the best fit value and its 1 sigma error when given
    unc = None
    if poi_p1 is not None and poi_m1 is not None:
        unc = (poi_p1 - poi_min, poi_min - poi_m1)
    num_min = Number(poi_min, unc)

    return DotDict(
        interp=interp,
        poi_min=poi_min,
        poi_p1=poi_p1,
        poi_m1=poi_m1,
        poi_p2=poi_p2,
        poi_m2=poi_m2,
        num_min=num_min,
    )


def evaluate_likelihood_scan_2d(
    poi1_values, poi2_values, dnll2_values, poi1_min=None, poi2_min=None
):
    """
    Takes the results of a 2D likelihood profiling scan given by *poi1_values*, *poi2_values* and
    the corresponding *dnll2_values* values, performs an interpolation and returns certain results
    of the scan in a dict. The two lists of poi values should represent an expanded grid, so that
    *poi1_values*, *poi2_values* and *dnll2_values* should all be 1D with the same length. When
    *poi1_min* and *poi2_min* are *None*, they are estimated from the interpolated curve.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``poi1_min``: The poi1 value corresponding to the minimum delta nll value.
    - ``poi2_min``: The poi2 value corresponding to the minimum delta nll value.
    - ``poi1_p1``: The poi1 value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_m1``: The poi1 value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_p2``: The poi1 value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi1_m2``: The poi1 value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_p1``: The poi2 value corresponding to the +1 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_m1``: The poi2 value corresponding to the -1 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_p2``: The poi2 value corresponding to the +2 sigma variation, or *None* when the
      calculation failed.
    - ``poi2_m2``: The poi2 value corresponding to the -2 sigma variation, or *None* when the
      calculation failed.
    - ``num1_min``: A Number instance representing the poi1 minimum and its 1 sigma uncertainty.
    - ``num2_min``: A Number instance representing the poi2 minimum and its 1 sigma uncertainty.
    """
    # ensure we are dealing with arrays
    poi1_values = np.array(poi1_values)
    poi2_values = np.array(poi2_values)
    dnll2_values = np.array(dnll2_values)

    # store ranges
    poi1_values_min = poi1_values.min()
    poi1_values_max = poi1_values.max()
    poi2_values_min = poi2_values.min()
    poi2_values_max = poi2_values.max()

    # remove values where dnll2 is nan
    mask = ~np.isnan(dnll2_values)
    poi1_values = poi1_values[mask]
    poi2_values = poi2_values[mask]
    dnll2_values = dnll2_values[mask]

    # obtain an interpolation function
    # interp = scipy.interpolate.interp2d(poi1_values, poi2_values, dnll2_values)
    # interp = scipy.interpolate.SmoothBivariateSpline(poi1_values, poi2_values, dnll2_values,
    #     kx=2, ky=2)
    coords = np.stack([poi1_values, poi2_values], axis=1)
    interp = scipy.interpolate.CloughTocher2DInterpolator(coords, dnll2_values)

    # get the minima
    if poi1_min is None or poi2_min is None:
        objective = lambda x: interp(*x) ** 2.0
        bounds1 = (poi1_values_min + 1e-4, poi1_values_max - 1e-4)
        bounds2 = (poi2_values_min + 1e-4, poi2_values_max - 1e-4)
        res = scipy.optimize.minimize(objective, [1.0, 1.0], tol=1e-7, bounds=[bounds1, bounds2])
        if res.status != 0:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
        poi1_min = res.x[0]
        poi2_min = res.x[1]

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v, n_poi):
        def minimize(bounds):
            res = minimize_1d(objective, bounds)
            return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

        if n_poi == 1:
            poi_values_min, poi_values_max = poi1_values_min, poi1_values_max
            poi_min = poi1_min
            objective = lambda x: (interp(x, poi2_min) - v) ** 2.0
        else:
            poi_values_min, poi_values_max = poi2_values_min, poi2_values_max
            poi_min = poi2_min
            objective = lambda x: (interp(poi1_min, x) - v) ** 2.0

        return (
            minimize((poi_min, poi_values_max - 1e-4)),
            minimize((poi_values_min + 1e-4, poi_min)),
        )

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi1_p1, poi1_m1 = get_intersections(chi2_levels[2][1], 1)
    poi2_p1, poi2_m1 = get_intersections(chi2_levels[2][1], 2)
    poi1_p2, poi1_m2 = get_intersections(chi2_levels[2][2], 1)
    poi2_p2, poi2_m2 = get_intersections(chi2_levels[2][2], 2)

    # create Number objects wrapping the best fit values and their 1 sigma error when given
    unc1 = None
    unc2 = None
    if poi1_p1 is not None and poi1_m1 is not None:
        unc1 = (poi1_p1 - poi1_min, poi1_min - poi1_m1)
    if poi2_p1 is not None and poi2_m1 is not None:
        unc2 = (poi2_p1 - poi2_min, poi2_min - poi2_m1)
    num1_min = Number(poi1_min, unc1)
    num2_min = Number(poi2_min, unc2)

    return DotDict(
        interp=interp,
        poi1_min=poi1_min,
        poi2_min=poi2_min,
        poi1_p1=poi1_p1,
        poi1_m1=poi1_m1,
        poi1_p2=poi1_p2,
        poi1_m2=poi1_m2,
        poi2_p1=poi2_p1,
        poi2_m1=poi2_m1,
        poi2_p2=poi2_p2,
        poi2_m2=poi2_m2,
        num1_min=num1_min,
        num2_min=num2_min,
    )


def create_dnll2_hist(poi1_values, poi2_values, dnll2_values, x_min=None, x_max=None, y_min=None,
        y_max=None, z_min=None, z_max=None):
    ROOT = import_ROOT()

    # make sure to work with numpy arrays
    poi1_values = np.array(poi1_values, dtype=np.float32)
    poi2_values = np.array(poi2_values, dtype=np.float32)
    dnll2_values = np.array(dnll2_values, dtype=np.float32)

    # set negative numbers to 0
    neg_mask = dnll2_values < 0
    dnll2_values[neg_mask] = 0

    # get the smallest difference between two points in each direction and call it bin width
    ex = np.unique(poi1_values)
    ey = np.unique(poi2_values)
    x_width = min(ex[1:] - ex[:-1])
    y_width = min(ey[1:] - ey[:-1])

    # get axis limits
    if x_min is None:
        x_min = min(poi1_values) - 0.5 * x_width
    if x_max is None:
        x_max = max(poi1_values) + 0.5 * x_width
    if y_min is None:
        y_min = min(poi2_values) - 0.5 * y_width
    if y_max is None:
        y_max = max(poi2_values) + 0.5 * y_width
    if z_min is None:
        z_min = dnll2_values[(~np.isnan(dnll2_values)) & (dnll2_values > 0)].min()
    if z_max is None:
        z_max = dnll2_values[~np.isnan(dnll2_values)].max()

    # infer number of bins
    x_bins = (x_max - x_min) / x_width
    y_bins = (y_max - y_min) / y_width
    if round(x_bins, 3) != int(x_bins):
        raise Exception("x axis range [{:3f},{:3f}) cannot be evenly split by bin width {}".format(
            x_min, x_max, x_width))
    if round(y_bins, 3) != int(y_bins):
        raise Exception("y axis range [{:3f},{:3f}) cannot be evenly split by bin width {}".format(
            y_min, y_max, y_width))
    x_bins = int(x_bins)
    y_bins = int(y_bins)

    # 2D histogram
    h_nll = ROOT.TH2F(create_random_name(), "", x_bins, x_min, x_max, y_bins, y_min, y_max)
    h_nll.SetMinimum(z_min)
    h_nll.SetMaximum(z_max)

    # fill it and lift values to the z_min to avoid drawing unfilled white bins
    fill_hist_from_points(h_nll, poi1_values, poi2_values, dnll2_values)
    for bx in range(1, h_nll.GetNbinsX() + 1):
        for by in range(1, h_nll.GetNbinsY() + 1):
            if h_nll.GetBinContent(bx, by) < z_min:
                h_nll.SetBinContent(bx, by, z_min)

    return h_nll
