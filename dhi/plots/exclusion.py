# coding: utf-8

"""
Exclusion result plots using ROOT.
"""

import math
import array
import uuid

import numpy as np
import scipy.interpolate

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, DotDict, to_root_latex, create_tgraph, minimize_1d, try_int
from dhi.plots.likelihoods import evaluate_likelihood_scan_1d, evaluate_likelihood_scan_2d
from dhi.plots.styles import use_style


colors = colors.root


@use_style("dhi_default")
def plot_exclusion_and_bestfit_1d(
    path,
    data,
    poi,
    scan_parameter,
    x_min=None,
    x_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a plot showing exluded regions of a *poi* over a *scan_parameter* for multiple analysis
    (or channels) as well as best fit values and saves it at *path*. *data* should be a list of
    dictionaries with fields "expected_limits" and "expected_nll", and optionally *observed_limits*,
    *observed_nll* and "name" (shown on the y-axis). The former four should be given as either
    dictionaries or numpy record arrays containing fields *poi* and "limit", or *poi* and "dnll2".
    When the name is a key of dhi.config.br_hh_names, its value is used as a label instead. Example:

    .. code-block:: python

        plot_exclusion_and_bestfit_1d(
            path="plot.pdf",
            poi="r",
            scan_parameter="kl",
            data=[{
                "expected_limits": {"kl": [...], "limit": [...]},
                "expected_nll": {"kl": [...], "dnll2": [...]},
                "expected_scan_min": 1.0,
            }, {
                ...
            }],
        )

    *x_min* and *x_max* define the range of the x-axis and default to the maximum range of poi
    values passed in data. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/exclusion.html#comparison-of-exclusion-performance
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check minimal fields per data entry
    assert(all("name" in d for d in data))
    assert(all("expected_limits" in d for d in data))
    assert(all("expected_nll" in d for d in data))
    n = len(data)
    has_obs = any("observed_limits" in d for d in data)
    scan_values = np.array(data[0]["expected_limits"][scan_parameter])

    # set default ranges
    if x_min is None:
        x_min = min(scan_values)
    if x_max is None:
        x_max = max(scan_values)

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

    # dummy histogram to control axes
    scan_label = to_root_latex(poi_data[scan_parameter].label)
    h_dummy = ROOT.TH1F("dummy", ";{};".format(scan_label), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    draw_objs.append((h_dummy, "HIST"))

    # expected exclusion area from intersections of limit with 1
    def create_exclusion_graph(data_key):
        excl_x, excl_y, excl_d, excl_u = [], [], [], []
        for i, d in enumerate(data):
            if data_key not in d:
                continue
            ranges = evaluate_limit_scan_1d(scan_values, d[data_key]["limit"]).excluded_ranges
            for start, stop in ranges:
                is_left = start < 1 and stop < 1
                excl_x.append(stop if is_left else start)
                excl_y.append(n - i - 0.5)
                excl_d.append((stop - start) if is_left else 0)
                excl_u.append(0 if is_left else (stop - start))
        return create_tgraph(len(excl_x), excl_x, excl_y, excl_d, excl_u, 0.5, 0.5)

    # expected
    g_excl_exp = create_exclusion_graph("expected_limits")
    r.setup_graph(g_excl_exp, color=colors.black, color_flags="f",
        props={"FillStyle": 3345, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0})
    draw_objs.append((g_excl_exp, "SAME,2"))
    legend_entries.append((g_excl_exp, "Excluded (expected)"))

    # observed
    if has_obs:
        g_excl_obs = create_exclusion_graph("observed_limits")
        r.setup_graph(g_excl_obs, color=colors.red, color_flags="f",
            props={"FillStyle": 3354, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0})
        draw_objs.append((g_excl_obs, "SAME,2"))
        legend_entries.append((g_excl_obs, "Excluded (observed)"))
    else:
        # dummy legend entry
        legend_entries.append((h_dummy, " ", ""))

    # best fit values
    scans = [
        evaluate_likelihood_scan_1d(d["expected_nll"][scan_parameter], d["expected_nll"]["dnll2"],
            poi_min=d.get("expected_scan_min"))
        for d in data
    ]
    g_bestfit = create_tgraph(n,
        [scan.num_min() for scan in scans],
        [n - i - 0.5 for i in range(n)],
        [scan.num_min.u(direction="down", default=0.) for scan in scans],
        [scan.num_min.u(direction="up", default=0.,) for scan in scans],
        0,
        0,
    )
    r.setup_graph(g_bestfit, props={"MarkerStyle": 20, "MarkerSize": 1.2, "LineWidth": 1})
    draw_objs.append((g_bestfit, "PEZ"))
    legend_entries.append((g_bestfit, "Best fit value"))

    # vertical line at 1
    if x_min < 1:
        line_one = ROOT.TLine(1., 0., 1., n)
        r.setup_line(line_one, props={"NDC": False, "LineStyle": 7}, color=colors.black)
        draw_objs.insert(-1, line_one)
        legend_entries.append((line_one, "Theory prediction", "l"))

    # line to separate combined result
    if len(data) > 1 and data[-1]["name"].lower() == "combined":
        line_obs = ROOT.TLine(x_min, 1., x_max, 1)
        r.setup_line(line_obs, props={"NDC": False}, color=12)
        draw_objs.append(line_obs)

    # y axis labels and ticks
    h_dummy.GetYaxis().SetBinLabel(1, "")
    label_tmpl = "#splitline{#bf{%s}}{#scale[0.75]{%s = %s}}"
    for i, (d, scan) in enumerate(zip(data, scans)):
        # name labels
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        label = label_tmpl % (label, scan_label, scan.num_min.str("%.1f", style="root"))
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
    legend = r.routines.create_legend(pad=pad, width=480, n=2)
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


@use_style("dhi_default")
def plot_exclusion_and_bestfit_2d(
    path,
    poi,
    scan_parameter1,
    scan_parameter2,
    expected_limits,
    observed_limits=None,
    xsec_values=None,
    xsec_levels=None,
    xsec_label_positions=None,
    xsec_unit=None,
    expected_likelihoods=None,
    expected_scan_minima=None,
    observed_likelihoods=None,
    observed_scan_minima=None,
    draw_sm_point=True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    model_parameters=None,
    campaign=None,
):
    """
    Creates a 2D plot showing excluded regions of two paramters *scan_parameter1* and
    *scan_parameter2* extracted from limits on a *poi* and saves it at *path*. The limit values must
    be passed via *expected_values* which should be a mapping to lists of values or a record array
    with keys "<scan_parameter1>", "<scan_parameter2>", "limit", and optionally "limit_p1",
    "limit_m1", "limit_p2" and "limit_m2" to denote uncertainties at 1 and 2 sigma. When
    *observed_limits* it set, it should have the same format (except for the uncertainties) to draw
    a colored, observed exclusion area.

    For visual guidance, contours can be drawn to certain cross section values which depend on the
    two scan parameters. To do so, *xsec_values* should be a map to lists of values or a record
    array with keys "<scan_parameter1>", "<scan_parameter2>" and "xsec". Based on these values,
    contours are derived at levels defined by *xsec_levels*, which are automatically inferred when
    not set explicitely. The position and units of labels for these contours are to be defined in
    *xsec_label_positions* and *xsec_unit* as there is not mechanism in ROOT to automatically draw
    them. Therefore, *xsec_label_positions* should be a list with the same length as *xsec_levels*.
    Each item can be a list of 3-tuples, each one referring to the x-value, the y-value (in units of
    *scan_parameter1* and *scan_parameter2*) and the rotation of a label for that contour level.
    *xsec_unit* can be a string that is appended to every label.

    When *expected_likelihoods* (*observed_likelihoods*) is set, it is used to extract expected
    (observed) best fit values and their uncertainties which are drawn as well. When set, it should
    be a mapping to lists of values or a record array with keys "<scan_parameter1>",
    "<scan_parameter2>" and "dnll2". By default, the position of the best value is directly
    extracted from the likelihood values. However, when *expected_scan_minima*
    (*observed_scan_minima*) is a 2-tuple of positions per scan parameter, this best fit value is
    used instead, e.g. to use combine's internally interpolated value. It should be noted that the
    expected best fit value is not drawn in case the *observed_likelihoods* is defined. In any case,
    the standard model point at (1, 1) as drawn as well unless *draw_sm_point* is *False*.

    *x_min*, *x_max*, *y_min* and *y_max* define the range of the x- and y-axis, respectively, and
    default to the scan parameter ranges found in *expected_limits*. *model_parameters* can be a
    dictionary of key-value pairs of model parameters. *campaign* should refer to the name of a
    campaign label defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/exclusion.html#2d-parameter-exclusion
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # convert record arrays to dicts
    def rec2dict(arr):
        if isinstance(arr, np.ndarray):
            arr = {key: arr[key] for key in arr.dtype.names}
        return arr

    expected_limits = rec2dict(expected_limits)
    expected_likelihoods = rec2dict(expected_likelihoods)
    xsec_values = rec2dict(xsec_values)

    # input checks
    assert(scan_parameter1 in expected_limits)
    assert(scan_parameter2 in expected_limits)
    assert("limit" in expected_limits)
    if xsec_values:
        assert(scan_parameter1 in xsec_values)
        assert(scan_parameter2 in xsec_values)
        assert("xsec" in xsec_values)
    if expected_likelihoods:
        assert(scan_parameter1 in expected_likelihoods)
        assert(scan_parameter2 in expected_likelihoods)
        assert("dnll2" in expected_likelihoods)
    if expected_scan_minima:
        assert(len(expected_scan_minima) == 2)
    if observed_likelihoods:
        assert(scan_parameter1 in observed_likelihoods)
        assert(scan_parameter2 in observed_likelihoods)
        assert("dnll2" in observed_likelihoods)
    if observed_scan_minima:
        assert(len(observed_scan_minima) == 2)

    # set shown ranges
    if x_min is None:
        x_min = min(expected_limits[scan_parameter1])
    if x_max is None:
        x_max = max(expected_limits[scan_parameter1])
    if y_min is None:
        y_min = min(expected_limits[scan_parameter2])
    if y_max is None:
        y_max = max(expected_limits[scan_parameter2])

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas()
    pad.cd()
    draw_objs = []

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter1].label)
    y_title = to_root_latex(poi_data[scan_parameter2].label)
    h_dummy = ROOT.TH2F("h", ";{};{};".format(x_title, y_title), 1, x_min, x_max, 1, y_min, y_max)
    r.setup_hist(h_dummy, pad=pad)
    draw_objs.append((h_dummy, ""))
    legend_entries = []

    # extract contours for all limit values
    contours = {}
    for key in ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]:
        if key not in expected_limits:
            continue

        # store contour graphs
        contours[key] = get_contours(expected_limits[scan_parameter1],
            expected_limits[scan_parameter2], expected_limits[key], levels=[1.])[0]

    # style graphs and add to draw objects, from outer to inner graphs (-2, -1, +1, +2), followed by
    # nominal or observed
    has_unc1 = "limit_p1" in contours and "limit_m1" in contours
    has_unc2 = "limit_p2" in contours and "limit_m2" in contours

    # -2 sigma exclusion
    if has_unc2:
        for g in contours["limit_m2"]:
            r.setup_graph(g, props={"LineStyle": 2, "LineColor": colors.black, "MarkerStyle": 20,
                "MarkerSize": 0, "FillColor": colors.light_grey})
            draw_objs.append((g, "F,SAME"))
        legend_entries.append((g, "#pm 2 #sigma expected"))

    # -1 and +1 sigma exclusion
    if has_unc1:
        for g in contours["limit_m1"]:
            r.setup_graph(g, props={"LineStyle": 2, "LineColor": colors.black, "MarkerStyle": 20,
                "MarkerSize": 0, "FillColor": colors.grey})
            draw_objs.append((g, "F,SAME"))
        legend_entries.insert(0, (g, "#pm 1 #sigma expected"))

        p1_col = colors.light_grey if has_unc2 else colors.white
        for g in contours["limit_p1"]:
            r.setup_graph(g, props={"FillColor": p1_col})
            draw_objs.append((g, "F,SAME"))

    # +2 sigma exclusion
    if has_unc2:
        for g in contours["limit_p2"]:
            r.setup_graph(g, props={"FillColor": colors.white})
            draw_objs.append((g, "F,SAME"))

    # cross section contours
    xsec_contours = None
    if xsec_values:
        if not xsec_levels or xsec_levels == "auto":
            xsec_levels = get_auto_contour_levels(xsec_values["xsec"])
            xsec_label_positions = None

        # get contour graphs
        xsec_contours = get_contours(xsec_values[scan_parameter1], xsec_values[scan_parameter2],
            xsec_values["xsec"], levels=xsec_levels, min_points=10)

        # draw them
        for graphs, level in zip(xsec_contours, xsec_levels):
            for g in graphs:
                r.setup_graph(g, props={"LineColor": colors.dark_grey_trans, "LineStyle": 3,
                    "LineWidth": 1})
                draw_objs.append((g, "L,SAME"))

        # add labels
        if xsec_label_positions:
            for level, positions in zip(xsec_levels, xsec_label_positions):
                text = str(try_int(level))
                if xsec_unit:
                    text = "{} {}".format(text, xsec_unit)
                for x, y, rot in positions:
                    if not (x_min <= x <= x_max) or not (y_min <= y <= y_max):
                        continue
                    xsec_label = ROOT.TLatex(0., 0., text)
                    r.setup_latex(xsec_label, props={"NDC": False, "TextSize": 12, "TextAlign": 22,
                        "TextColor": colors.dark_grey_trans, "TextAngle": rot})
                    xsec_label.SetX(x)
                    xsec_label.SetY(y)
                    draw_objs.append((xsec_label, "SAME"))

    # nominal exclusion
    for g in contours["limit"]:
        r.setup_graph(g, props={"LineStyle": 2, "LineColor": colors.black, "MarkerStyle": 20,
            "MarkerSize": 0})
        draw_objs.append((g, "L,SAME"))
    legend_entries.insert(0, (g, "Excluded (expected)", "L"))

    # observed exclusion
    if observed_limits:
        # get contours
        obs_contours = get_contours(observed_limits[scan_parameter1],
            observed_limits[scan_parameter2], observed_limits["limit"], levels=[1.])[0]

        # draw them
        for g in obs_contours:
            r.setup_graph(g, props={"LineStyle": 1, "LineColor": colors.black, "MarkerStyle": 20,
                "MarkerSize": 0, "FillColor": colors.blue_signal_trans})
            draw_objs.append((g, "L,SAME"))
            draw_objs.append((g, "F,SAME"))
        legend_entries.append((g, "Excluded (observed)", "AF"))

    # best fit point, observed or expected
    likelihoods, scan_minima = None, None
    if observed_likelihoods:
        likelihoods, scan_minima = observed_likelihoods, observed_scan_minima
        label = "Best fit value (observed)"
    elif expected_likelihoods:
        likelihoods, scan_minima = expected_likelihoods, expected_scan_minima
        label = "Best fit value (expected)"
    if likelihoods:
        scan = evaluate_likelihood_scan_2d(likelihoods[scan_parameter1],
            likelihoods[scan_parameter2], likelihoods["dnll2"],
            poi1_min=scan_minima and scan_minima[0], poi2_min=scan_minima and scan_minima[1])
        g_fit = ROOT.TGraphAsymmErrors(1)
        g_fit.SetPoint(0, scan.num1_min(), scan.num2_min())
        if scan.num1_min.uncertainties:
            g_fit.SetPointEXhigh(0, scan.num1_min.u(direction="up"))
            g_fit.SetPointEXlow(0, scan.num1_min.u(direction="down"))
        if scan.num2_min.uncertainties:
            g_fit.SetPointEYhigh(0, scan.num2_min.u(direction="up"))
            g_fit.SetPointEYlow(0, scan.num2_min.u(direction="down"))
        r.setup_graph(g_fit, props={"FillStyle": 0}, color=colors.black)
        draw_objs.append((g_fit, "PEZ"))
        legend_entries.append((g_fit, label, "LPE"))

    # SM point
    if draw_sm_point:
        g_sm = create_tgraph(1, 1, 1)
        r.setup_graph(g_sm, props={"MarkerStyle": 33, "MarkerSize": 2.5}, color=colors.red)
        draw_objs.insert(1, (g_sm, "P"))
        legend_entries.append((g_sm, "Standard model", "P"))

    # model parameter label
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=48 + i * 24, props={"TextSize": 20}))

    # legend
    legend = r.routines.create_legend(pad=pad, width=480, n=3, x2=-44, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "lrt",
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
    pad.cd()
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    canvas.SaveAs(path)


def evaluate_limit_scan_1d(scan_values, limit_values):
    """
    Takes the results of an upper limit scan given by the *scan_values* and the corresponding *limit*
    values, performs an interpolation and returns certain results of the scan in a dict.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``excluded_ranges``: A list of 2-tuples denoting ranges in units of the poi where limits are
      below one.
    """
    scan_values = np.array(scan_values)
    limit_values = np.array(limit_values)

    # first, obtain an interpolation function
    mask = ~np.isnan(limit_values)
    scan_values = scan_values[mask]
    limit_values = limit_values[mask]
    # interp = scipy.interpolate.interp1d(scan_values, limit_values, kind="cubic")
    interp = scipy.interpolate.interp1d(scan_values, limit_values, kind="linear")

    # interpolation bounds
    bounds = (scan_values.min() + 1e-4, scan_values.max() - 1e-4)

    # helper to find intersections with one given a starting point
    def get_intersection(start):
        objective = lambda x: (interp(x) - 1) ** 2.0
        res = minimize_1d(objective, bounds, start=start)
        return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

    # get exclusion range edges from intersections and remove duplicates
    rnd = lambda v: round(v, 3)
    edges = [rnd(scan_values.min()), rnd(scan_values.max())]
    for start in np.linspace(bounds[0], bounds[1], 100):
        edge = get_intersection(start)
        if edge is None:
            continue
        edge = rnd(edge)
        if edge not in edges:
            edges.append(edge)
    edges.sort()

    # create ranges consisting of two adjacent edges
    ranges = [(edges[i - 1], edges[i]) for i in range(1, len(edges))]

    # select those ranges whose central value is below 1
    excluded_ranges = [
        r for r in ranges
        if interp(0.5 * (r[1] + r[0])) < 1
    ]

    return DotDict(
        interp=interp,
        excluded_ranges=excluded_ranges,
    )


def pad_histogram(hist, wx, wy, **kwargs):
    # first, extract histogram data into a 2D array (x-axis is inner dimension 1)
    data = np.array([
        [
            hist.GetBinContent(bx, by)
            for bx in range(1, hist.GetNbinsX() + 1)
        ]
        for by in range(1, hist.GetNbinsY() + 1)
    ])

    # pad the data, passing all kwargs to np.pad
    pad_x = 1 if wx > 0 else 0
    pad_y = 1 if wy > 0 else 0
    data = np.pad(data, pad_width=[(pad_y, pad_y), (pad_x, pad_x)], **kwargs)

    # amend bin edges
    edges_x = [hist.GetXaxis().GetBinLowEdge(bx) for bx in range(1, hist.GetNbinsX() + 2)]
    edges_y = [hist.GetYaxis().GetBinLowEdge(by) for by in range(1, hist.GetNbinsY() + 2)]
    if wx > 0:
        edges_x = [edges_x[0] - wx] + edges_x + [edges_x[-1] + wx]
    if wy > 0:
        edges_y = [edges_y[0] - wy] + edges_y + [edges_y[-1] + wy]

    # combine data and edges into a new histogram
    hist_padded = hist.__class__(str(uuid.uuid4()), "", len(edges_x) - 1, array.array("d", edges_x),
        len(edges_y) - 1, array.array("d", edges_y))
    hist_padded.SetDirectory(0)

    for by, _data in enumerate(data):
        for bx, z in enumerate(_data):
            hist_padded.SetBinContent(bx + 1, by + 1, z)

    return hist_padded


# helper to fill histograms
def fill_hist(h, x_values, y_values, z_values):
    ROOT = import_ROOT()

    # first, fill a graph to use ROOT's interpolation for missing vlaues
    g = ROOT.TGraph2D(len(z_values))
    for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
        if not np.isnan(z):
            g.SetPoint(i, x, y, z)

    # then, fill histogram bins
    for bx in range(1, h.GetNbinsX() + 1):
        for by in range(1, h.GetNbinsY() + 1):
            x = h.GetXaxis().GetBinCenter(bx)
            y = h.GetYaxis().GetBinCenter(by)
            z = g.Interpolate(x, y)
            h.SetBinContent(bx, by, z)


# helper to extract contours
def get_contours(x_values, y_values, z_values, levels, min_points=5):
    ROOT = import_ROOT()

    # infer number of bins, axis ranges and bin widths of histograms used to extract contours
    n_x = len(set(x_values))
    n_y = len(set(y_values))
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    w_x = (max_x - min_x) / (n_x - 1)
    w_y = (max_y - min_y) / (n_y - 1)

    # create and fill a hist
    h = ROOT.TH2F(str(uuid.uuid4()), "",
        n_x, min_x - 0.5 * w_x, max_x + 0.5 * w_x,
        n_y, min_y - 0.5 * w_y, max_y + 0.5 * w_y,
    )
    fill_hist(h, x_values, y_values, z_values)

    # pad with edge values, then with a rather large one to definitely close contours
    h = pad_histogram(h, w_x * 0.02, w_y * 0.02, mode="edge")
    h = pad_histogram(h, w_x * 0.02, w_y * 0.02, mode="constant", constant_values=100)

    # get contours in a nested list of graphs
    contours = [_get_contour(h, l) for l in levels]

    # store contour graphs with a minimum number of points
    return [
        [g for g in graphs if g.GetN() >= min_points]
        for graphs in contours
    ]


def _get_contour(hist, level):
    ROOT = import_ROOT()

    # make a clone to set contour levels
    h = hist.Clone(str(uuid.uuid4()))
    h.SetContour(1, array.array("d", [level]))

    # extract contour graphs after drawing into a temporary pad (see LIST option docs)
    c = ROOT.TCanvas("tmp", "tmp")
    pad = c.cd()
    pad.SetLogz(True)
    h.Draw("CONT,Z,LIST")
    pad.Update()
    graphs = ROOT.gROOT.GetListOfSpecials().FindObject("contours")

    # convert from nested TList to python list of graphs for that contour level
    contours = []
    if graphs or not graphs.GetSize():
        contours = [graphs.At(0).At(j).Clone() for j in range(graphs.At(0).GetSize())]

    # finally, close the canvas
    c.Close()

    return contours


def get_auto_contour_levels(values, steps=(1,)):
    min_value = min(values)
    max_value = max(values)
    start = int(math.floor(math.log(min_value, 10)))
    stop = int(math.ceil(math.log(max_value, 10)))

    levels = []
    for e in range(start, stop + 1):
        for s in steps:
            l = s * 10**e
            if min_value <= l <= max_value:
                levels.append(l)

    return levels
