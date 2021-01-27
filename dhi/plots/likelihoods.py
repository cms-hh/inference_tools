# coding: utf-8

"""
Likelihood plots using ROOT.
"""

import math
import array

import numpy as np
import scipy.interpolate
import scipy.optimize
from scinum import Number

from dhi.config import poi_data, campaign_labels, chi2_levels, colors, color_sequence
from dhi.util import (
    import_ROOT, to_root_latex, get_neighbor_coordinates, create_tgraph, DotDict, minimize_1d,
    try_int,
)
from dhi.plots.styles import use_style


colors = colors.root


@use_style("dhi_default")
def plot_likelihood_scan_1d(
    path,
    poi,
    expected_values,
    theory_value=None,
    poi_min=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a likelihood plot of the 1D scan of a *poi* and saves it at *path*. *expected_values*
    should be a mapping to lists of values or a record array with keys "<poi_name>" and "dnll2".
    *theory_value* can be a 3-tuple denoting the nominal theory prediction of the POI and its up and
    down uncertainties which is drawn as a vertical bar. When *poi_min* is set, it should be the
    value of the poi that leads to the best likelihood. Otherwise, it is estimated from the
    interpolated curve.

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *y_min* sets the lower
    limit of the y-axis and defaults to 0, or 0.01 when *y_log* is *True*. *x_min* and *x_max*
    define the x-axis range and default to the range of poi values. *model_parameters* can be a
    dictionary of key-value pairs of model parameters. *campaign* should refer to the name of a
    campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#1d
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get valid poi and delta nll values
    poi_values = np.array(expected_values[poi], dtype=np.float32)
    dnll2_values = np.array(expected_values["dnll2"], dtype=np.float32)

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
    y_max_value = max(dnll2_values[(poi_values >= x_min) & (poi_values <= x_min)])
    if y_log:
        if y_min is None:
            y_min = 1e-2
        y_max = y_min * 10**(1.35 * math.log10(y_max_value / y_min))
    else:
        if y_min is None:
            y_min = 0.
        y_max = 1.35 * (y_max_value - y_min)

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
            r.setup_line(line, props={"LineColor": colors.red, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # lines at chi2_1 intervals
    for n in [chi2_levels[1][1], chi2_levels[1][2]]:
        if n < y_max_value:
            line = ROOT.TLine(x_min, n, x_max, n)
            r.setup_line(line, props={"LineColor": colors.red, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # theory prediction with uncertainties
    if theory_value:
        # theory graph and line
        g_thy = create_tgraph(1, theory_value[0], 0, theory_value[2], theory_value[1], 0,
            y_max_value)
        r.setup_graph(g_thy, props={"FillStyle": 3244, "MarkerStyle": 20, "MarkerSize": 0},
            color=colors.red, color_flags="lfm")
        line_thy = ROOT.TLine(theory_value[0], 0., theory_value[0], y_max_value)
        r.setup_line(line_thy, props={"NDC": False}, color=colors.red)
        draw_objs.append((g_thy, "SAME,2"))
        draw_objs.append(line_thy)
        legend_entries.append((g_thy, "Theory prediction"))

    # line for best fit value
    line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_value)
    r.setup_line(line_fit, props={"LineWidth": 2, "NDC": False}, color=colors.black)
    fit_label = "{} = {}".format(to_root_latex(poi_data[poi].label),
        scan.num_min.str(format="%.2f", style="root"))
    draw_objs.append(line_fit)
    legend_entries.insert(0, (line_fit, fit_label, "l"))

    # nll curve
    g_nll = create_tgraph(len(poi_values), poi_values, dnll2_values)
    r.setup_graph(g_nll, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 0.75})
    draw_objs.append((g_nll, "SAME,CP"))

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=230, n=len(legend_entries))
    r.setup_legend(legend)
    for tpl in legend_entries:
        legend.AddEntry(*tpl)
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
def plot_likelihood_scans_1d(
    path,
    poi,
    data,
    theory_value=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
):
    """
    Plots multiple curves of 1D likelihood scans of a POI *poi1* and *poi2*, and saves it at *path*.
    All information should be passed as a list *data*. Entries must be dictionaries with the
    following content:

        - "expected_values": A mapping to lists of values or a record array with keys "<poi1_name>",
          "<poi2_name>" and "dnll2".
        - "poi_min": A float describing the best fit value of the POI. When not set, the minimum is
          estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    *theory_value* can be a 3-tuple denoting the nominal theory prediction of the POI and its up and
    down uncertainties which is drawn as a vertical bar.

    *x_min* and *x_max* define the x-axis range of POI, and *y_min* controls the minimum y-axis
    value. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#1d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # validate data entries
    for i, d in enumerate(data):
        # convert expected values to arrays
        assert("expected_values" in d)
        exp_values = d["expected_values"]
        if isinstance(exp_values, np.ndarray):
            exp_values = {k: exp_values[k] for k in exp_values.dtype.names}
        assert(poi in exp_values)
        assert("dnll2" in exp_values)
        # keep only valid points
        exp_values = {k: np.array(v, dtype=np.float32) for k, v in exp_values.items()}
        mask = ~np.isnan(exp_values["dnll2"])
        exp_values[poi] = exp_values[poi][mask]
        exp_values["dnll2"] = exp_values["dnll2"][mask]
        d["expected_values"] = exp_values
        # check poi minimum
        d.setdefault("poi_min", None)
        # default name
        d.setdefault("name", str(i + 1))

    # set x range
    if x_min is None:
        x_min = min([min(d["expected_values"][poi]) for d in data])
    if x_max is None:
        x_max = max([max(d["expected_values"][poi]) for d in data])

    # set y range
    y_max_value = max([
        d["expected_values"]["dnll2"][
            (d["expected_values"][poi] >= x_min) & (d["expected_values"][poi] <= x_min)
        ]
        for d in data
    ])
    if y_log:
        if y_min is None:
            y_min = 1e-2
        y_max = y_min * 10**(1.35 * math.log10(y_max_value / y_min))
    else:
        if y_min is None:
            y_min = 0.
        y_max = 1.35 * (y_max_value - y_min)

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
        if n < y_max_value:
            line = ROOT.TLine(x_min, n, x_max, n)
            r.setup_line(line, props={"LineColor": colors.black, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # theory prediction with uncertainties
    if theory_value:
        # theory graph and line
        g_thy = create_tgraph(1, theory_value[0], 0, theory_value[2], theory_value[1], 0,
            y_max_value)
        r.setup_graph(g_thy, props={"FillStyle": 3244, "MarkerStyle": 20, "MarkerSize": 0},
            color=colors.red, color_flags="lfm")
        line_thy = ROOT.TLine(theory_value[0], 0., theory_value[0], y_max_value)
        r.setup_line(line_thy, props={"NDC": False}, color=colors.red)
        draw_objs.append((g_thy, "SAME,2"))
        draw_objs.append(line_thy)
        legend_entries.append((g_thy, "Theory prediction"))

    # perform scans and draw nll curves
    for d, col in zip(data, color_sequence[:len(data)]):
        # evaluate the scan, run interpolation and error estimation
        scan = evaluate_likelihood_scan_1d(d["expected_values"][poi], d["expected_values"]["dnll2"],
            poi_min=d["poi_min"])

        # draw the curve
        g_nll = create_tgraph(len(d["expected_values"][poi]), d["expected_values"][poi],
            d["expected_values"]["dnll2"])
        r.setup_graph(g_nll, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 0.75},
            color=col)
        draw_objs.append((g_nll, "SAME,CP"))
        legend_entries.append((g_nll, d["name"]))

        # line for best fit value
        line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_value)
        r.setup_line(line_fit, props={"LineWidth": 2, "NDC": False}, color=col)
        draw_objs.append(line_fit)

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
    legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
    legend = r.routines.create_legend(pad=pad, width=legend_cols * 210, n=legend_rows,
        props={"NColumns": legend_cols})
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
def plot_likelihood_scan_2d(
    path,
    poi1,
    poi2,
    expected_values,
    poi1_min=None,
    poi2_min=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    fill_nans=True,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a likelihood plot of the 2D scan of two POIs *poi1* and *poi2*, and saves it at *path*.
    *expected_values* should be a mapping to lists of values or a record array with keys
    "<poi1_name>", "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be
    the values of the POIs that lead to the best likelihood. Otherwise, they are  estimated from the
    interpolated curve.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. *z_min* and *z_max* limit the range of the z-axis.
    When *fill_nans* is *True*, points with failed fits, denoted by nan values, are filled with the
    averages of neighboring fits. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get poi and delta nll values
    poi1_values = np.array(expected_values[poi1], dtype=np.float32)
    poi2_values = np.array(expected_values[poi2], dtype=np.float32)
    dnll2_values = np.array(expected_values["dnll2"], dtype=np.float32)

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_2d(
        poi1_values, poi2_values, dnll2_values, poi1_min=poi1_min, poi2_min=poi2_min,
    )

    # transform the poi coordinates and dnll2 values into a 2d array
    # where the inner (outer) dimension refers to poi1 (poi2)
    e1 = np.unique(poi1_values)
    e2 = np.unique(poi2_values)
    i1 = np.searchsorted(e1, poi1_values, side="right") - 1
    i2 = np.searchsorted(e2, poi2_values, side="right") - 1
    data = np.zeros((e2.size, e1.size), dtype=np.float32)
    data[i2, i1] = dnll2_values

    # optionally fill nans with averages over neighboring points
    if fill_nans:
        nans = np.argwhere(np.isnan(data))
        npoints = {tuple(p): get_neighbor_coordinates(data.shape, *p) for p in nans}
        nvals = {p: data[[c[0] for c in cs], [c[1] for c in cs]] for p, cs in npoints.items()}
        nmeans = {p: vals[~np.isnan(vals)].mean() for p, vals in nvals.items()}
        data[[p[0] for p in nmeans], [p[1] for p in nmeans]] = nmeans.values()

    # change non-positive numbers to the next smallest number below a threshold
    mask = data <= 0
    pos_min = min(data[~mask].min(), z_min or 1e-3)
    data[mask] = pos_min

    # set ranges
    bin_width1 = (max(poi1_values) - min(poi1_values)) / (len(e1) - 1)
    bin_width2 = (max(poi2_values) - min(poi2_values)) / (len(e2) - 1)
    x_min_default = min(poi1_values) - 0.5 * bin_width1
    x_max_default = max(poi1_values) + 0.5 * bin_width1
    y_min_default = min(poi2_values) - 0.5 * bin_width2
    y_max_default = max(poi2_values) + 0.5 * bin_width2
    if x_min is None:
        x_min = x_min_default
    if x_max is None:
        x_max = x_max_default
    if y_min is None:
        y_min = y_min_default
    if y_max is None:
        y_max = y_max_default
    if z_min is None:
        z_min = data.min()
    if z_max is None:
        z_max = data.max()

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"RightMargin": 0.17, "Logz": True})
    pad.cd()
    draw_objs = []

    # 2d histogram
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    z_title = "-2 #Delta log(L)"
    h_nll = ROOT.TH2F("h_nll", ";{};{};{}".format(x_title, y_title, z_title),
        data.shape[1], x_min_default, x_max_default, data.shape[0], y_min_default, y_max_default)
    r.setup_hist(h_nll, pad=pad, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
    r.setup_z_axis(h_nll.GetZaxis(), pad=pad, props={"TitleOffset": 1.3})
    h_nll.GetXaxis().SetRangeUser(x_min, x_max)
    h_nll.GetYaxis().SetRangeUser(y_min, y_max)
    draw_objs.append((h_nll, "COLZ"))
    for i, j in np.ndindex(data.shape):
        h_nll.SetBinContent(j + 1, i + 1, data[i, j])

    # best fit point
    g_fit = ROOT.TGraphAsymmErrors(1)
    g_fit.SetPoint(0, scan.num1_min(), scan.num2_min())
    if scan.num1_min.uncertainties:
        g_fit.SetPointEXhigh(0, scan.num1_min.u(direction="up"))
        g_fit.SetPointEXlow(0, scan.num1_min.u(direction="down"))
    if scan.num2_min.uncertainties:
        g_fit.SetPointEYhigh(0, scan.num2_min.u(direction="up"))
        g_fit.SetPointEYlow(0, scan.num2_min.u(direction="down"))
    r.setup_graph(g_fit, color=colors.red)
    draw_objs.append((g_fit, "PEZ"))

    # 1 and 2 sigma contours
    h1 = ROOT.TH2F(h_nll)
    h2 = ROOT.TH2F(h_nll)
    r.setup_hist(h1, props={"LineWidth": 2, "LineColor": colors.green,
        "Contour": (1, array.array("d", [chi2_levels[2][1]]))})
    r.setup_hist(h2, props={"LineWidth": 2, "LineColor": colors.yellow,
        "Contour": (1, array.array("d", [chi2_levels[2][2]]))})
    draw_objs.append((h1, "SAME,CONT3"))
    draw_objs.append((h2, "SAME,CONT3"))

    # best fit value labels
    fit_label1 = "{} = {}".format(to_root_latex(poi_data[poi1].label),
        scan.num1_min.str(format="%.2f", style="root"))
    fit_label2 = "{} = {}".format(to_root_latex(poi_data[poi2].label),
        scan.num2_min.str(format="%.2f", style="root"))
    fit_label1 = r.routines.create_top_right_label(fit_label1, pad=pad, x_offset=150, y_offset=30,
        props={"TextAlign": 13})
    fit_label2 = r.routines.create_top_right_label(fit_label2, pad=pad, x_offset=150, y_offset=68,
        props={"TextAlign": 13})
    draw_objs.append(fit_label1)
    draw_objs.append(fit_label2)

    # model parameter label
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

        - "expected_values": A mapping to lists of values or a record array with keys "<poi1_name>",
          "<poi2_name>" and "dnll2".
        - "poi_mins": A list of two floats describing the best fit value of the two POIs. When not
          set, the minima are estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. When *fill_nans* is *True*, points with failed
    fits, denoted by nan values, are filled with the averages of neighboring fits.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*.

    Example: Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # validate data entries
    for i, d in enumerate(data):
        # convert expected values to arrays
        assert("expected_values" in d)
        exp_values = d["expected_values"]
        if isinstance(exp_values, np.ndarray):
            exp_values = {k: exp_values[k] for k in exp_values.dtype.names}
        assert(poi1 in exp_values)
        assert(poi2 in exp_values)
        assert("dnll2" in exp_values)
        exp_values = {k: np.array(v, dtype=np.float32) for k, v in exp_values.items()}
        d["expected_values"] = exp_values
        # check poi minima
        d["poi_mins"] = d.get("poi_mins") or [None, None]
        assert(len(d["poi_mins"]) == 2)
        # default name
        d.setdefault("name", str(i + 1))

    # set ranges
    if x_min is None:
        x_min = min([min(d["expected_values"][poi1]) for d in data])
    if x_max is None:
        x_max = max([max(d["expected_values"][poi1]) for d in data])
    if y_min is None:
        y_min = min([min(d["expected_values"][poi2]) for d in data])
    if y_max is None:
        y_max = max([max(d["expected_values"][poi2]) for d in data])

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logz": True})
    pad.cd()
    legend_entries = []
    draw_objs = []

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    h_dummy = ROOT.TH2F("h", ";{};{};".format(x_title, y_title), 1, x_min, x_max, 1, y_min, y_max)
    r.setup_hist(h_dummy, pad=pad)
    draw_objs.append((h_dummy, "HIST"))

    # loop through data entries
    for d, col in zip(data, color_sequence[:len(data)]):
        # evaluate the scan
        scan = evaluate_likelihood_scan_2d(
            d["expected_values"][poi1], d["expected_values"][poi2], d["expected_values"]["dnll2"],
            poi1_min=d["poi_mins"][0], poi2_min=d["poi_mins"][1],
        )

        # for each set of expected values, transform the poi coordinates and dnll2 values into a 2d
        # array where the inner (outer) dimension refers to poi1 (poi2)
        e1 = np.unique(d["expected_values"][poi1])
        e2 = np.unique(d["expected_values"][poi2])
        i1 = np.searchsorted(e1, d["expected_values"][poi1], side="right") - 1
        i2 = np.searchsorted(e2, d["expected_values"][poi2], side="right") - 1
        exp = np.zeros((e2.size, e1.size), dtype=np.float32)
        exp[i2, i1] = d["expected_values"]["dnll2"]

        # optionally fill nans with averages over neighboring points
        if fill_nans:
            nans = np.argwhere(np.isnan(exp))
            npoints = {tuple(p): get_neighbor_coordinates(exp.shape, *p) for p in nans}
            nvals = {p: exp[[c[0] for c in cs], [c[1] for c in cs]] for p, cs in npoints.items()}
            nmeans = {p: vals[~np.isnan(vals)].mean() for p, vals in nvals.items()}
            exp[[p[0] for p in nmeans], [p[1] for p in nmeans]] = nmeans.values()

        # infer the binning
        _x_min = min(d["expected_values"][poi1])
        _x_max = max(d["expected_values"][poi1])
        _y_min = min(d["expected_values"][poi2])
        _y_max = max(d["expected_values"][poi2])
        bin_width1 = (_x_max - _x_min) / (len(e1) - 1)
        bin_width2 = (_y_max - _y_min) / (len(e2) - 1)
        binning = (
            exp.shape[1], _x_min - 0.5 * bin_width1, _x_max + 0.5 * bin_width1,
            exp.shape[0], _y_min - 0.5 * bin_width2, _y_max + 0.5 * bin_width2,
        )

        # create two histograms for plotting 1 and 2 sigma contours
        h1 = ROOT.TH2F("h1_{}".format(i), "", *binning)
        h2 = ROOT.TH2F("h2_{}".format(i), "", *binning)
        r.setup_hist(h1, props={"LineWidth": 2, "LineColor": colors[col], "LineStyle": 1,
           "Contour": (1, array.array("d", [chi2_levels[2][1]]))})
        r.setup_hist(h2, props={"LineWidth": 2, "LineColor": colors[col], "LineStyle": 2,
           "Contour": (1, array.array("d", [chi2_levels[2][2]]))})

        # fill them
        for x, y in np.ndindex(exp.shape):
            h1.SetBinContent(y + 1, x + 1, exp[x, y])
            h2.SetBinContent(y + 1, x + 1, exp[x, y])

        # best fit point
        g_fit = create_tgraph(1, scan.num1_min(), scan.num2_min())
        r.setup_graph(g_fit, props={"MarkerStyle": 33, "MarkerSize": 2}, color=colors[col])

        # add to draw objects and legend
        draw_objs.append((g_fit, "PEZ"))
        draw_objs.append((h1, "SAME,CONT3"))
        draw_objs.append((h2, "SAME,CONT3"))
        legend_entries.append((g_fit, "{} (best fit)".format(d["name"]), "P"))
        legend_entries.append((h1, "#pm1#sigma"))
        legend_entries.append((h2, "#pm2#sigma"))

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=40 + i * 24, props={"TextSize": 20}))

    # legend
    legend_cols = len(data)
    legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
    legend = r.routines.create_legend(pad=pad, width=legend_cols * 210, height=legend_rows * 30,
        props={"NColumns": legend_cols})
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

    # remove values where dnnl2 is nan
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

    # remove values where dnnl2 is nan
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
