# coding: utf-8

"""
Exclusion result plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors, br_hh_names
from dhi.util import import_ROOT, to_root_latex, create_tgraph, try_int
from dhi.plots.limits import evaluate_limit_scan_1d
from dhi.plots.likelihoods import evaluate_likelihood_scan_1d, evaluate_likelihood_scan_2d
from dhi.plots.util import (
    use_style, draw_model_parameters, invert_graph, get_graph_points, get_contours, get_text_extent,
)


colors = colors.root


@use_style("dhi_default")
def plot_exclusion_and_bestfit_1d(
    path,
    data,
    poi,
    scan_parameter,
    x_min=None,
    x_max=None,
    left_margin=None,
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
    values passed in data. *left_margin* controls the left margin of the pad in pixels.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *h_lines* can be
    a list of integers denoting positions where additional horizontal lines are drawn for visual
    guidance. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

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
    left_margin = left_margin or 150  # pixels
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
    scans = []
    for d in data:
        scan = None
        if "nll_values" in d:
            scan = evaluate_likelihood_scan_1d(
                d["nll_values"][scan_parameter],
                d["nll_values"]["dnll2"],
                poi_min=d.get("scan_min"),
            )
        scans.append(scan)
    if any(scans):
        g_bestfit = create_tgraph(n,
            [(scan.num_min() if scan else -1e5) for scan in scans],
            [n - i - 0.5 for i in range(n)],
            [(scan.num_min.u(direction="down", default=0.) if scan else 0) for scan in scans],
            [(scan.num_min.u(direction="up", default=0.,) if scan else 0) for scan in scans],
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

    # legend
    legend = r.routines.create_legend(pad=pad, width=480, n=2)
    r.setup_legend(legend, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries)
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
            draw_objs.append((g, "SAME,F"))
        legend_entries.append((g, "#pm 2 #sigma expected", "LF"))

    # -1 and +1 sigma exclusion
    if has_unc1:
        for g in contours["limit_p1"]:
            r.setup_graph(g, props={"LineStyle": 2, "FillColor": colors.grey})
            draw_objs.append((g, "SAME,F"))
        legend_entries.insert(0, (g, "#pm 1 #sigma expected", "LF"))

        p1_col = colors.light_grey if has_unc2 else colors.white
        for g in contours["limit_m1"]:
            r.setup_graph(g, props={"FillColor": p1_col})
            draw_objs.append((g, "SAME,F"))

    # -2 sigma exclusion
    if has_unc2:
        for g in contours["limit_m2"]:
            r.setup_graph(g, props={"FillColor": colors.white})
            draw_objs.append((g, "SAME,F"))

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
                draw_objs.append((g, "SAME,C"))

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
        draw_objs.append((g, "SAME,L"))
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
            frame_kwargs=[{"mode": "edge"}] + [{"mode": "contour+"}],
        )[0]

        # draw them
        for g in obs_contours:
            # create an inverted graph to close the outer polygon
            g_inv = invert_graph(g, x_axis=h_dummy.GetXaxis(), y_axis=h_dummy.GetYaxis())
            r.setup_graph(g, props={"LineStyle": 1})
            r.setup_graph(g_inv, props={"LineStyle": 1, "FillColor": colors.blue_signal_trans})
            draw_objs.append((g, "SAME,L"))
            draw_objs.append((g_inv, "SAME,F"))
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
    pad.cd()
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    canvas.SaveAs(path)


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


def locate_xsec_labels(graphs, level, label_width, pad_width, pad_height, x_min, x_max, y_min,
        y_max, other_positions=None, min_points=10):
    positions = []
    other_positions = other_positions or []

    # conversions from values in x or y (depending on the axis range) to values in pixels
    x_width = x_max - x_min
    y_width = y_max - y_min
    x_to_px = lambda x: x * pad_width / x_width
    y_to_px = lambda y: y * pad_height / y_width

    # define visible ranges
    x_min_vis = x_min + 0.02 * x_width
    x_max_vis = x_max - 0.02 * x_width
    y_min_vis = y_min + 0.02 * y_width
    y_max_vis = y_max - 0.02 * y_width

    # helper to get the ellipse-transformed distance between two points, normalized to pad dimension
    pad_dist = lambda x, y, x0, y0: (((x - x0) / pad_width)**2 + ((y - y0) / pad_height)**2)**0.5

    # helper to check if two points are too close in terms of the label width
    too_close = lambda x, y, x0, y0: pad_dist(x, y, x0, y0) < 1.1 * (label_width / pad_width)

    for g in graphs:
        # get graph points
        x_values, y_values = get_graph_points(g)
        n_points = len(x_values)

        # skip when there are not enough points
        if n_points < min_points:
            continue

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

        # get the best initial position by only checking if were close to the visible window
        for idx1 in np.append(adist, adist[0]):
            x, y = xx[idx1, hb_size], yy[idx1, hb_size]
            if (x_min_vis <= x <= x_max_vis) and (y_min_vis <= y <= y_max_vis):
                break
        else:
            idx1 = 0

        # get the best position by checking the distance to other labels and the roration
        # use idx1 when no position was found
        rot = 0.
        for idx2 in np.append(adist, idx1):
            x, y = xx[idx2, hb_size], yy[idx2, hb_size]
            if not (x_min_vis <= x <= x_max_vis) or not (y_min_vis <= y <= y_max_vis):
                continue
            elif any(too_close(x, y, x0, y0) for x0, y0, _ in positions + other_positions):
                continue

            # rotation
            ind = (idx2 * block_size + hb_size) % n_points
            dx, dy = np.gradient(line_contour, axis=0)[ind]
            dx = x_to_px(dx)
            dy = y_to_px(dy)
            if dx or dy:
                rot = np.rad2deg(np.arctan2(dy, dx))
                rot = (rot + 90) % 180 - 90
            else:
                rot = 0.

            # avoid large rotations
            if abs(rot) > 70:
                continue

            # at this point, a good position was found
            break

        # store when visible
        if (x_min_vis <= x <= x_max_vis) and (y_min_vis <= y <= y_max_vis):
            positions.append((x, y, rot))

    return positions
