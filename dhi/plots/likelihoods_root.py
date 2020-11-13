# coding: utf-8

"""
Likelihood plots using ROOT.
"""

import math
import array

import numpy as np

from dhi.config import poi_data, campaign_labels, chi2_levels, colors as _colors
from dhi.util import import_ROOT, to_root_latex, get_neighbor_coordinates, create_tgraph
from dhi.plots.likelihoods_mpl import evaluate_likelihood_scan_1d, evaluate_likelihood_scan_2d


def plot_likelihood_scan_1d(
    path,
    poi,
    expected_values,
    theory_value=None,
    poi_min=None,
    campaign="2017",
    x_min=None,
    x_max=None,
    y_min=None,
    y_log=False,
):
    """
    Creates a likelihood plot of the 1D scan of a *poi* and saves it at *path*. *expected_values*
    should be a mapping to lists of values or a record array with keys "<poi_name>" and "dnll2".
    *theory_value* can be a 3-tuple denoting the nominal theory prediction and its up and down
    uncertainties which is drawn as a vertical bar. When *poi_min* is set, it should be the value of
    the poi that leads to the best likelihood. Otherwise, it is estimated from the interpolated
    curve. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels. When *y_log* is *True*, the y-axis is plotted with a logarithmic
    scale. *y_min* sets the lower limit of the y-axis and defaults to 0, or 0.01 when *y_log* is
    *True*. *x_min* and *x_max* define the x-axis range and default to the range of poi values.

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/plotting.html#1d-likelihood-scans
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get valid poi and delta nll values
    poi_values = np.array(expected_values[poi], dtype=np.float32)
    dnll2_values = np.array(expected_values["dnll2"], dtype=np.float32)

    # set default range
    if x_min is None:
        x_min = min(poi_values)
    if x_max is None:
        x_max = max(poi_values)

    # select valid points
    mask = ~np.isnan(dnll2_values)
    poi_values = poi_values[mask]
    dnll2_values = dnll2_values[mask]

    # define limits
    y_max_value = max(dnll2_values)
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
            r.setup_line(line, props={"LineColor": _colors.root.red, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # lines at chi2_1 intervals
    for n in [chi2_levels[1][1], chi2_levels[1][2]]:
        if n < max(dnll2_values):
            line = ROOT.TLine(x_min, n, x_max, n)
            r.setup_line(line, props={"LineColor": _colors.root.red, "LineStyle": 2, "NDC": False})
            draw_objs.append(line)

    # theory prediction with uncertainties
    if theory_value:
        # theory graph and line
        g_thy = create_tgraph(0, theory_value[0], 0, theory_value[2], theory_value[1], 0,
            y_max_value)
        r.setup_graph(g_thy, props={"FillStyle": 3345, "MarkerStyle": 20, "MarkerSize": 0},
            color=_colors.root.red, color_flags="lfm")
        line_thy = ROOT.TLine(theory_value[0], 0., theory_value[0], y_max_value)
        r.setup_line(line_thy, props={"NDC": False}, color=_colors.root.red)
        draw_objs.append((g_thy, "SAME,2"))
        draw_objs.append(line_thy)
        legend_entries.append((g_thy, "SM prediction"))

    # line for best fit value
    line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_value)
    r.setup_line(line_fit, props={"LineWidth": 2, "NDC": False}, color=_colors.root.black)
    fit_label = "{} = {}".format(to_root_latex(poi_data[poi].label),
        scan.num_min.str(format="%.2f", style="root"))
    draw_objs.append(line_fit)
    legend_entries.insert(0, (line_fit, fit_label, "l"))

    # nll curve
    g_nll = create_tgraph(len(poi_values), poi_values, dnll2_values)
    r.setup_graph(g_nll, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 0.75})
    draw_objs.append((g_nll, "SAME,CP"))

    # legend
    legend = r.routines.create_legend(pad=pad, width=230, height=len(legend_entries) * 35)
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


def plot_likelihood_scan_2d(
    path,
    poi1,
    poi2,
    expected_values,
    poi1_min=None,
    poi2_min=None,
    campaign="2017",
    z_log=True,
    x1_min=None,
    x1_max=None,
    x2_min=None,
    x2_max=None,
    fill_nans=True,
):
    """
    Creates a likelihood plot of the 2D scan of two pois *poi1* and *poi2*, and saves it at *path*.
    *expected_values* should be a mapping to lists of values or a record array with keys
    "<poi1_name>", "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be
    the values of the pois that lead to the best likelihood. Otherwise, they are  estimated from the
    interpolated curve. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels. When *z_log* is *True*, the z-axis is plotted with a logarithmic
    scale. *x1_min*, *x1_max*, *x2_min* and *x2_max* define the axis range of poi1 and poi2,
    respectively, and default to the ranges of the poi values. When *fill_nans* is *True*, points
    with failed fits, denoted by nan values, are filled with the averages of neighboring fits.

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/plotting.html#2d-likelihood-scans
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # get poi and delta nll values
    poi1_values = np.array(expected_values[poi1], dtype=np.float32)
    poi2_values = np.array(expected_values[poi2], dtype=np.float32)
    dnll2_values = np.array(expected_values["dnll2"], dtype=np.float32)

    # set default ranges
    if x1_min is None:
        x1_min = min(poi1_values)
    if x1_max is None:
        x1_max = max(poi1_values)
    if x2_min is None:
        x2_min = min(poi2_values)
    if x2_max is None:
        x2_max = max(poi2_values)

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_2d(
        poi1_values, poi2_values, dnll2_values, poi1_min=poi1_min, poi2_min=poi2_min
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

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"RightMargin": 0.17, "Logz": z_log})
    pad.cd()
    draw_objs = []

    # 2d histogram
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    z_title = "-2 #Delta log(L)"
    bin_width1 = (x1_max - x1_min) / (len(e1) - 1)
    bin_width2 = (x2_max - x2_min) / (len(e2) - 1)
    h_nll = ROOT.TH2F("h", ";{};{};{}".format(x_title, y_title, z_title),
        data.shape[1], x1_min - 0.5 * bin_width1, x1_max + 0.5 * bin_width1,
        data.shape[0], x2_min - 0.5 * bin_width2, x2_max + 0.5 * bin_width2,
    )
    r.setup_hist(h_nll, pad=pad, props={"Contour": 100})
    r.setup_z_axis(h_nll.GetZaxis(), pad=pad, props={"TitleOffset": 1.3})
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
    r.setup_graph(g_fit, color=_colors.root.red)
    draw_objs.append((g_fit, "PEZ"))

    # contours
    h_contours68 = ROOT.TH2F(h_nll)
    h_contours95 = ROOT.TH2F(h_nll)
    r.setup_hist(h_contours68, props={"LineWidth": 2, "LineColor": _colors.root.green})
    r.setup_hist(h_contours95, props={"LineWidth": 2, "LineColor": _colors.root.yellow})
    h_contours68.SetContour(1, array.array("d", [chi2_levels[2][1]]))
    h_contours95.SetContour(1, array.array("d", [chi2_levels[2][2]]))
    draw_objs.append((h_contours68, "SAME,CONT3"))
    draw_objs.append((h_contours95, "SAME,CONT3"))

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
