# coding: utf-8

"""
Exclusion result plots using ROOT.
"""

import math
import array
import uuid
import six
import itertools

import numpy as np
import scipy.interpolate

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import (
    import_ROOT, DotDict, to_root_latex, create_tgraph, minimize_1d, try_int, temporary_canvas,
)
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
    h_lines=None,
    campaign=None,
):
    """
    Creates a plot showing exluded regions of a *poi* over a *scan_parameter* for multiple analysis
    (or channels) as well as best fit values and saves it at *path*. *data* should be a list of
    dictionaries with fields "name", "expected_limits" and "nll_values", and optionally
    *observed_limits*, and "scan_min". Limits and NLL values should be given as either dictionaries
    or numpy record arrays containing fields *poi* and "limit", or *poi* and "dnll2". When a value
    for "scan_min" is given, this value is used to mark the best fit value (e.g. from combine's
    internal interpolation). Otherwise, the value is extracted in a custom interpolation approach.
    When the name is a key of dhi.config.br_hh_names, its value is used as a label instead. Example:

    .. code-block:: python

        plot_exclusion_and_bestfit_1d(
            path="plot.pdf",
            poi="r",
            scan_parameter="kl",
            data=[{
                "name": "...",
                "expected_limits": {"kl": [...], "limit": [...]},
                "observed_limits": {"kl": [...], "limit": [...]},  # optional
                "nll_values": {"kl": [...], "dnll2": [...]},
                "scan_min": 1.0,  # optional
            }, {
                ...
            }],
        )

    *x_min* and *x_max* define the range of the x-axis and default to the maximum range of poi
    values passed in data. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *h_lines* can be a list of integers denoting positions where additional horizontal
    lines are drawn for visual guidance. *campaign* should refer to the name of a campaign label
    defined in *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/exclusion.html#comparison-of-exclusion-performance
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check minimal fields per data entry
    assert(all("name" in d for d in data))
    assert(all("expected_limits" in d for d in data))
    assert(all("nll_values" in d for d in data))
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
    r.setup_x_axis(h_dummy.GetXaxis(), pad=pad, props={
        "TitleOffset": 1.2, "LabelOffset": r.pixel_to_coord(canvas, y=4)})
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
        r.setup_graph(g_excl_obs, color=colors.blue_signal, color_flags="f",
            props={"FillStyle": 3354, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0})
        draw_objs.append((g_excl_obs, "SAME,2"))
        legend_entries.insert(-1, (g_excl_obs, "Excluded (observed)"))
    else:
        # dummy legend entry
        legend_entries.append((h_dummy, " ", ""))

    # best fit values
    scans = [
        evaluate_likelihood_scan_1d(d["nll_values"][scan_parameter], d["nll_values"]["dnll2"],
            poi_min=d.get("scan_min"))
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

    # theory prediction
    if x_min < 1:
        line_thy = ROOT.TLine(1., 0., 1., n)
        r.setup_line(line_thy, props={"NDC": False, "LineStyle": 1}, color=colors.red)
        draw_objs.insert(-1, line_thy)
        legend_entries.append((line_thy, "Theory prediction", "l"))

    # horizontal guidance lines
    if h_lines:
        for i in h_lines:
            line_obs = ROOT.TLine(x_min, float(i), x_max, float(i))
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

    # model parameter labels
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
    xsec_unit=None,
    nll_values=None,
    scan_minima=None,
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
    be passed via *expected_limits* which should be a mapping to lists of values or a record array
    with keys "<scan_parameter1>", "<scan_parameter2>", "limit", and optionally "limit_p1",
    "limit_m1", "limit_p2" and "limit_m2" to denote uncertainties at 1 and 2 sigma. When
    *observed_limits* it set, it should have the same format (except for the uncertainties) to draw
    a colored, observed exclusion area.

    For visual guidance, contours can be drawn to certain cross section values which depend on the
    two scan parameters. To do so, *xsec_values* should be a map to lists of values or a record
    array with keys "<scan_parameter1>", "<scan_parameter2>" and "xsec". Based on these values,
    contours are derived at levels defined by *xsec_levels*, which are automatically inferred when
    not set explicitely. *xsec_unit* can be a string that is appended to every label.

    When *nll_values* is set, it is used to extract expected best fit values and their uncertainties
    which are drawn as well. When set, it should be a mapping to lists of values or a record array
    with keys "<scan_parameter1>", "<scan_parameter2>" and "dnll2". By default, the position of the
    best value is directly extracted from the likelihood values. However, when *scan_minima* is a
    2-tuple of positions per scan parameter, this best fit value is used instead, e.g. to use
    combine's internally interpolated value. The standard model point at (1, 1) as drawn as well
    unless *draw_sm_point* is *False*.

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
    if observed_limits:
        observed_limits = rec2dict(observed_limits)
    nll_values = rec2dict(nll_values)
    xsec_values = rec2dict(xsec_values)

    # input checks
    assert(scan_parameter1 in expected_limits)
    assert(scan_parameter2 in expected_limits)
    assert("limit" in expected_limits)
    if observed_limits:
        assert(scan_parameter1 in observed_limits)
        assert(scan_parameter2 in observed_limits)
        assert("limit" in observed_limits)
    if xsec_values:
        assert(scan_parameter1 in xsec_values)
        assert(scan_parameter2 in xsec_values)
        assert("xsec" in xsec_values)
    if nll_values:
        assert(scan_parameter1 in nll_values)
        assert(scan_parameter2 in nll_values)
        assert("dnll2" in nll_values)
    if scan_minima:
        assert(len(scan_minima) == 2)

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

    # conversion factor from pixel to x-axis range
    pad_width = canvas.GetWindowWidth() * (1. - pad.GetLeftMargin() - pad.GetRightMargin())
    pad_height = canvas.GetWindowHeight() * (1. - pad.GetTopMargin() - pad.GetBottomMargin())
    px_to_x = (x_max - x_min) / pad_width

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
        contours[key] = get_contours(
            expected_limits[scan_parameter1],
            expected_limits[scan_parameter2],
            expected_limits[key],
            levels=[1.],
            frame_kwargs=[{"mode": "edge"}] + [{"mode": "contour+"}],
        )[0]

    # style graphs and add to draw objects, from outer to inner graphs (-2, -1, +1, +2), followed by
    # nominal or observed
    has_unc1 = "limit_p1" in contours and "limit_m1" in contours
    has_unc2 = "limit_p2" in contours and "limit_m2" in contours

    # +2 sigma exclusion
    if has_unc2:
        for g in contours["limit_p2"]:
            r.setup_graph(g, props={"LineStyle": 2, "FillColor": colors.light_grey})
            draw_objs.append((g, "F,SAME"))
        legend_entries.append((g, "#pm 2 #sigma expected", "LF"))

    # -1 and +1 sigma exclusion
    if has_unc1:
        for g in contours["limit_p1"]:
            r.setup_graph(g, props={"LineStyle": 2, "FillColor": colors.grey})
            draw_objs.append((g, "F,SAME"))
        legend_entries.insert(0, (g, "#pm 1 #sigma expected", "LF"))

        p1_col = colors.light_grey if has_unc2 else colors.white
        for g in contours["limit_m1"]:
            r.setup_graph(g, props={"FillColor": p1_col})
            draw_objs.append((g, "F,SAME"))

    # -2 sigma exclusion
    if has_unc2:
        for g in contours["limit_m2"]:
            r.setup_graph(g, props={"FillColor": colors.white})
            draw_objs.append((g, "F,SAME"))

    # cross section contours
    if xsec_values:
        if not xsec_levels or xsec_levels == "auto":
            xsec_levels = get_auto_contour_levels(xsec_values["xsec"])

        # get contour graphs
        xsec_contours = get_contours(
            xsec_values[scan_parameter1],
            xsec_values[scan_parameter2],
            xsec_values["xsec"],
            levels=xsec_levels,
        )

        # draw them
        for graphs, level in zip(xsec_contours, xsec_levels):
            for g in graphs:
                r.setup_graph(g, props={"LineColor": colors.dark_grey_trans_70, "LineStyle": 3,
                    "LineWidth": 1})
                draw_objs.append((g, "L,SAME"))

        # draw labels at automatic positions
        all_positions = []
        for graphs, level in zip(xsec_contours, xsec_levels):
            # get the approximate label width
            text = str(try_int(level))
            if xsec_unit:
                text = "{} {}".format(text, xsec_unit)
            label_width = get_text_extent(text, 12, 43)[0] * px_to_x

            # calculate and store the position
            label_positions = locate_xsec_labels(graphs, level, label_width, pad_width, pad_height,
                x_min, x_max, y_min, y_max, other_positions=all_positions)
            all_positions.extend(label_positions)

            # draw them
            for x, y, rot in label_positions:
                xsec_label = ROOT.TLatex(0., 0., text)
                r.setup_latex(xsec_label, props={"NDC": False, "TextSize": 12, "TextAlign": 22,
                    "TextColor": colors.dark_grey, "TextAngle": rot, "X": x, "Y": y})
                draw_objs.append((xsec_label, "SAME"))

    # nominal exclusion
    for g in contours["limit"]:
        r.setup_graph(g, props={"LineStyle": 2})
        draw_objs.append((g, "L,SAME"))
    legend_entries.insert(0, (g, "Excluded (expected)", "L"))

    # observed exclusion
    # for testing
    # observed_limits = {
    #     scan_parameter1: expected_limits[scan_parameter1],
    #     scan_parameter2: expected_limits[scan_parameter2],
    #     "limit": expected_limits["limit"] * 1.2,
    # }
    if observed_limits:
        # get contours
        obs_contours = get_contours(
            observed_limits[scan_parameter1],
            observed_limits[scan_parameter2],
            observed_limits["limit"],
            levels=[1.],
            frame_kwargs=[{"mode": "edge"}],
        )[0]

        # draw them
        for g in obs_contours:
            # create an inverted graph to close the outer polygon
            g_inv = invert_graph(g, padding=1000.)
            r.setup_graph(g, props={"LineStyle": 1})
            r.setup_graph(g_inv, props={"LineStyle": 1, "FillColor": colors.blue_signal_trans})
            draw_objs.append((g, "L,SAME"))
            draw_objs.append((g_inv, "F,SAME"))
        legend_entries.append((g_inv, "Excluded (observed)", "AF"))

    # best fit point
    if nll_values:
        scan = evaluate_likelihood_scan_2d(nll_values[scan_parameter1],
            nll_values[scan_parameter2], nll_values["dnll2"],
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
        legend_entries.append((g_fit, "Best fit value", "LPE"))

    # SM point
    if draw_sm_point:
        g_sm = create_tgraph(1, 1, 1)
        r.setup_graph(g_sm, props={"MarkerStyle": 33, "MarkerSize": 2.5}, color=colors.red)
        draw_objs.insert(-1, (g_sm, "P"))
        legend_entries.append((g_sm, "Standard model", "P"))

    # legend
    legend = r.routines.create_legend(pad=pad, width=480, n=3, x2=-44, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad, "lrt",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        for i, (p, v) in enumerate(model_parameters.items()):
            text = "{} = {}".format(poi_data.get(p, {}).get("label", p), try_int(v))
            draw_objs.append(r.routines.create_top_left_label(text, pad=pad, x_offset=25,
                y_offset=48 + i * 24, props={"TextSize": 20}))

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


def frame_histogram(hist, x_width, y_width, mode="edge", frame_value=None, contour_level=None):
    # when the mode is "contour-", edge values below the level are set to a higher value which
    # effectively closes contour areas that are below (thus the "-") the contour level
    # when the mode is "contour++", the opposite happens to close contour areas above the level
    assert(mode in ["edge", "constant", "contour+", "contour-"])
    if mode == "constant":
        assert(frame_value is not None)
    elif mode in ["contour+", "contour-"]:
        assert(contour_level is not None)

    # first, extract histogram data into a 2D array (x-axis is inner dimension 1)
    data = np.array([
        [
            hist.GetBinContent(bx, by)
            for bx in range(1, hist.GetNbinsX() + 1)
        ]
        for by in range(1, hist.GetNbinsY() + 1)
    ])

    # pad the data
    if mode == "constant":
        pad_kwargs = {"mode": "constant", "constant_values": frame_value}
    else:
        pad_kwargs = {"mode": "edge"}
    data = np.pad(data, pad_width=[1, 1], **pad_kwargs)

    # update frame values
    if mode in ["contour+", "contour-"]:
        # close contours depending on the mode
        idxs = list(itertools.product((0, data.shape[0] - 1), range(0, data.shape[1])))
        idxs += list(itertools.product(range(1, data.shape[0] - 1), (0, data.shape[1] - 1)))
        for i, j in idxs:
            if mode == "contour-":
                if data[i, j] < contour_level:
                    data[i, j] = contour_level + 10 * abs(contour_level)
            elif mode == "contour+":
                if data[i, j] > contour_level:
                    data[i, j] = contour_level - 10 * abs(contour_level)

    # amend bin edges
    x_edges = [hist.GetXaxis().GetBinLowEdge(bx) for bx in range(1, hist.GetNbinsX() + 2)]
    y_edges = [hist.GetYaxis().GetBinLowEdge(by) for by in range(1, hist.GetNbinsY() + 2)]
    x_edges = [x_edges[0] - x_width] + x_edges + [x_edges[-1] + x_width]
    y_edges = [y_edges[0] - y_width] + y_edges + [y_edges[-1] + y_width]

    # combine data and edges into a new histogram and fill it
    hist_padded = hist.__class__(str(uuid.uuid4()), "", len(x_edges) - 1, array.array("d", x_edges),
        len(y_edges) - 1, array.array("d", y_edges))
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
def get_contours(x_values, y_values, z_values, levels, frame_kwargs=None, min_points=5):
    ROOT = import_ROOT()

    # to extract contours, we need a 2D histogram with optimized bin widths, edges and padding
    def get_min_diff(values):
        values = sorted(set(values))
        diffs = [(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        return min(diffs)

    # x axis
    x_min = min(x_values)
    x_max = max(x_values)
    x_width = get_min_diff(x_values)
    x_n = (x_max - x_min) / x_width
    x_n = int(x_n + 1) if try_int(x_n) else int(math.ceil(x_n))

    # y axis
    y_min = min(y_values)
    y_max = max(y_values)
    y_width = get_min_diff(y_values)
    y_n = (y_max - y_min) / y_width
    y_n = int(y_n + 1) if try_int(y_n) else int(math.ceil(y_n))

    # create and fill a hist
    h = ROOT.TH2F(str(uuid.uuid4()), "", x_n, x_min, x_max, y_n, y_min, y_max)
    fill_hist(h, x_values, y_values, z_values)

    # get contours in a nested list of graphs
    contours = []
    frame_kwargs = frame_kwargs if isinstance(frame_kwargs, (list, tuple)) else [frame_kwargs]
    for l in levels:
        # frame the histogram
        _h = h
        for fk in filter(bool, frame_kwargs):
            w = fk.pop("width", 0.01)
            _h = frame_histogram(_h, x_width * w, y_width * w, contour_level=l, **fk)

        # get the contour graphs and filter by the number of points
        graphs = _get_contour(_h, l)
        graphs = [g for g in graphs if g.GetN() >= min_points]
        contours.append(graphs)

    return contours


def _get_contour(hist, level):
    ROOT = import_ROOT()

    # make a clone to set contour levels
    h = hist.Clone(str(uuid.uuid4()))
    h.SetContour(1, array.array("d", [level]))

    # extract contour graphs after drawing into a temporary pad (see LIST option docs)
    with temporary_canvas() as c:
        pad = c.cd()
        pad.SetLogz(True)
        h.Draw("CONT,Z,LIST")
        pad.Update()
        graphs = ROOT.gROOT.GetListOfSpecials().FindObject("contours")

        # convert from nested TList to python list of graphs for that contour level
        contours = []
        if graphs or not graphs.GetSize():
            contours = [graphs.At(0).At(j).Clone() for j in range(graphs.At(0).GetSize())]

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


def get_graph_points(g):
    ROOT = import_ROOT()

    x_values, y_values = [], []
    x, y = ROOT.Double(), ROOT.Double()
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        x_values.append(float(x))
        y_values.append(float(y))

    return x_values, y_values


def invert_graph(g, x_min=None, x_max=None, y_min=None, y_max=None, padding=0.):
    ROOT = import_ROOT()

    # get all graph values
    x_values, y_values = get_graph_points(g)

    # get default extrema
    x_min = (min(x_values) if x_min is None else x_min) - padding
    x_max = (max(x_values) if x_max is None else x_max) + padding
    y_min = (min(y_values) if y_min is None else y_min) - padding
    y_max = (max(y_values) if y_max is None else y_max) + padding

    # copy the graph with prepended extrema
    g_inv = ROOT.TGraph(g.GetN() + 5)
    g_inv.SetPoint(0, x_min, y_min)
    g_inv.SetPoint(1, x_min, y_max)
    g_inv.SetPoint(2, x_max, y_max)
    g_inv.SetPoint(3, x_max, y_min)
    g_inv.SetPoint(4, x_min, y_min)

    # copy remaining points
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        g_inv.SetPoint(i + 5, x, y)

    return g_inv


def get_text_extent(t, text_size=None, text_font=None):
    ROOT = import_ROOT()

    # convert to a tlatex if t is a string, otherwise clone
    if isinstance(t, six.string_types):
        t = ROOT.TLatex(0., 0., t)
    else:
        t = t.Clone()

    # set size and font when set
    if text_size is not None:
        t.SetTextSize(text_size)
    if text_font is not None:
        t.SetTextFont(text_font)

    # only available when the font precision is 3
    assert(t.GetTextFont() % 10 == 3)

    # create a temporary canvas and draw the text
    with temporary_canvas() as c:
        c.cd()
        t.Draw()

        # extract the bounding box dimensions
        w = array.array("I", [0])
        h = array.array("I", [0])
        t.GetBoundingBox(w, h)

    return int(w[0]), int(h[0])


def locate_xsec_labels(graphs, level, label_width, pad_width, pad_height, x_min, x_max, y_min,
        y_max, other_positions=None):
    positions = []
    other_positions = other_positions or []

    # conversions from values in x or y values (depending on the axis range) to pixels
    x_width = x_max - x_min
    y_width = y_max - y_min
    x_to_px = lambda x: x * pad_width / x_width
    y_to_px = lambda y: y * pad_height / y_width

    # define visible ranges
    x_min_vis = x_min + 0.05 * x_width
    x_max_vis = x_max - 0.05 * x_width
    y_min_vis = y_min + 0.05 * y_width
    y_max_vis = y_max - 0.05 * y_width

    for g in graphs:
        # get graph points
        x_values, y_values = get_graph_points(g)
        n_points = len(x_values)

        # compute the line contour and number of blocks
        line_contour = np.array([x_values, y_values]).T
        n_blocks = int(np.ceil(n_points / label_width)) if label_width > 1 else 1
        block_size = n_points if n_blocks == 1 else int(round(label_width))

        # split contour into blocks of length block_size, filling the last block by cycling the
        # contour start (per np.resize semantics)
        # due to cycling, the index returned is taken modulo n_points
        xx = np.resize(line_contour[:, 0], (n_blocks, block_size))
        yy = np.resize(line_contour[:, 1], (n_blocks, block_size))
        yfirst = yy[:, :1]
        ylast = yy[:, -1:]
        xfirst = xx[:, :1]
        xlast = xx[:, -1:]
        s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
        l = np.hypot(xlast - xfirst, ylast - yfirst)

        # ignore warning that divide by zero throws, as this is a valid option
        with np.errstate(divide="ignore", invalid="ignore"):
            distances = (abs(s) / l).sum(axis=-1)

        # labels are drawn in the middle of the block (hb_size) where the contour is the closest
        # to a straight line, but not too close to a preexisting label
        hb_size = block_size // 2
        adist = np.argsort(distances)

        # if all candidates are to close to existing labels, or too close to the edges, go back to the
        # straightest part adist[0]
        for idx in np.append(adist, adist[0]):
            x, y = xx[idx, hb_size], yy[idx, hb_size]
            if not (x_min_vis <= x <= x_max_vis) or not (y_min_vis <= y <= y_max_vis):
                continue
            elif any(abs(x - x_) < 0.05 * x_width for x_, _, _ in positions + other_positions):
                continue
            elif any(abs(y - y_) < 0.05 * y_width for _, y_, _ in positions + other_positions):
                continue
            break

        # rotation
        ind = (idx * block_size + hb_size) % n_points
        dx, dy = np.gradient(line_contour, axis=0)[ind]
        dx = x_to_px(dx)
        dy = y_to_px(dy)
        if dx or dy:
            rot = np.rad2deg(np.arctan2(dy, dx))
            rot = (rot + 90) % 180 - 90
        else:
            rot = 0.

        # store it
        positions.append((x, y, rot))

    return positions
