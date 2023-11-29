# coding: utf-8

"""
Exclusion result plots using ROOT.
"""

import math

import numpy as np

from dhi.config import poi_data, campaign_labels, colors, br_hh_names, hh_references
from dhi.plots.limits import evaluate_limit_scan_1d, _print_excluded_ranges
from dhi.plots.likelihoods import (
    _preprocess_values, evaluate_likelihood_scan_1d, evaluate_likelihood_scan_2d,
)
from dhi.plots.util import (
    use_style, create_model_parameters, invert_graph, get_contours, get_text_extent,
    locate_contour_labels, Style,
)
from dhi.util import import_ROOT, to_root_latex, create_tgraph, try_int, make_list, warn


colors = colors.root


@use_style("dhi_default")
def plot_exclusion_and_bestfit_1d(
    paths,
    data,
    poi,
    scan_parameter,
    show_best_fit_error=True,
    x_min=None,
    x_max=None,
    pad_width=None,
    left_margin=None,
    right_margin=None,
    entry_height=None,
    label_size=None,
    model_parameters=None,
    h_lines=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing exluded regions of a *poi* over a *scan_parameter* for multiple analyses
    (or channels) as well as best fit values and saves it at *paths*. *data* should be a list of
    dictionaries with fields "name", "expected_limits" and "nll_values", and optionally
    *observed_limits*, and "scan_min". Limits and NLL values should be given as either dictionaries
    or numpy record arrays containing fields *poi* and "limit", or *poi* and "dnll2". When a value
    for "scan_min" is given, this value is used to mark the best fit value (e.g. from combine's
    internal interpolation). Otherwise, the value is extracted in a custom interpolation approach.
    When the name is a key of dhi.config.br_hh_names, its value is used as a label instead. Example:

    .. code-block:: python

        plot_exclusion_and_bestfit_1d(
            paths=["plot.pdf", "plot.png"],
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

    When *show_best_fit_error* is *True*, the error bars on the best values are shown. The nominal
    values are taken from the "scan_min" entries in *data*, or they are recomputed using
    scipy.minimize when *None*. *x_min* and *x_max* define the range of the x-axis and default to
    the maximum range of poi values passed in data. *pad_width*, *left_margin*, *right_margin*,
    *entry_height* and *label_size* can be set to a size in pixels to overwrite internal defaults.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *h_lines* can be
    a list of integers denoting positions where additional horizontal lines are drawn for visual
    guidance. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"
        - "summary": Add references for data entries whose name refers to a known entry in
          hh_references.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/exclusion.html#comparison-of-exclusion-performance  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check minimal fields per data entry
    assert all("name" in d for d in data)
    assert all("expected_limits" in d for d in data)
    assert all("nll_values" in d for d in data)
    n = len(data)
    has_obs = any("observed_limits" in d for d in data)

    # set default ranges
    if x_min is None:
        x_min = min([min(d["expected_limits"][scan_parameter]) for d in data])
    if x_max is None:
        x_max = max([max(d["expected_limits"][scan_parameter]) for d in data])

    # some constants for plotting
    pad_width = pad_width or 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = left_margin or 150  # pixels
    right_margin = right_margin or 20  # pixels
    entry_height = entry_height or 90  # pixels
    head_space = 130  # pixels
    label_size = label_size or 22

    # get the pad height
    pad_height = n * entry_height + head_space + top_margin + bottom_margin

    # get relative pad margins and fill into props
    pad_margins = {
        "TopMargin": float(top_margin) / pad_height,
        "BottomMargin": float(bottom_margin) / pad_height,
        "LeftMargin": float(left_margin) / pad_width,
        "RightMargin": float(right_margin) / pad_width,
    }

    # get the y maximum
    y_max = (pad_height - top_margin - bottom_margin) / float(entry_height)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(
        width=pad_width,
        height=pad_height,
        pad_props=pad_margins,
    )
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    scan_label = to_root_latex(poi_data[scan_parameter].label)
    h_dummy = ROOT.TH1F("dummy", ";{};".format(scan_label), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"TitleOffset": 1.2, "LabelOffset": r.pixel_to_coord(canvas, y=4)},
    )
    draw_objs.append((h_dummy, "HIST"))

    # expected exclusion area from intersections of limit with 1
    def create_exclusion_graph(kind):
        data_key = kind + "_limits"
        excl_x, excl_y, excl_d, excl_u = [], [], [], []
        for i, d in enumerate(data):
            if data_key not in d:
                continue
            scan_values = np.array(d[data_key][scan_parameter])
            scan = evaluate_limit_scan_1d(scan_values, d[data_key]["limit"], interpolation="linear")
            ranges = scan.excluded_ranges
            _print_excluded_ranges(
                scan_parameter,
                "{} {}, {}".format(poi, kind, d["name"]),
                scan_values,
                ranges,
                "linear",
            )
            for start, stop in ranges:
                is_left = start < 1 and stop < 1
                excl_x.append(stop if is_left else start)
                excl_y.append(n - i - 0.5)
                excl_d.append((stop - start) if is_left else 0)
                excl_u.append(0 if is_left else (stop - start))
        if not excl_x:
            return None
        return create_tgraph(len(excl_x), excl_x, excl_y, excl_d, excl_u, 0.5, 0.5)

    # expected
    g_excl_exp = create_exclusion_graph("expected")
    if g_excl_exp:
        r.setup_graph(
            g_excl_exp,
            color=colors.black,
            color_flags="f",
            props={"FillStyle": 3345, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0},
        )
        draw_objs.append((g_excl_exp, "SAME,2"))
        legend_entries.append((g_excl_exp, "Excluded (expected)"))
    else:
        warn("no expected exclusion range found, no graph will be visible")
        legend_entries.append((h_dummy, " ", ""))

    # observed
    if has_obs:
        g_excl_obs = create_exclusion_graph("observed")
        if g_excl_obs:
            r.setup_graph(
                g_excl_obs,
                color=colors.blue_signal,
                color_flags="f",
                props={"FillStyle": 3354, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0},
            )
            draw_objs.append((g_excl_obs, "SAME,2"))
            legend_entries.insert(-1, (g_excl_obs, "Excluded (observed)"))
        else:
            warn("no observed exclusion range found, no graph will be visible")
            legend_entries.append((h_dummy, " ", ""))
    else:
        # dummy legend entry
        legend_entries.append((h_dummy, " ", ""))

    # perform scans to extract best fit values
    scans = []
    n_intervals = 1
    for d in data:
        if not d or d.get("nll_values") is None:
            scans.append([])
            continue

        # preprocess values
        poi_min = d.get("scan_min")
        dnll2_values, poi_values = _preprocess_values(
            d["nll_values"]["dnll2"],
            (poi, d["nll_values"][scan_parameter]),
            shift_negative_values=True,
            remove_nans=True,
            min_is_external=poi_min is not None,
            origin="entry '{}'".format(d["name"]),
        )

        # check for disjoint intervals
        dnll2_below_1sigma = np.where(dnll2_values < 1)[0]
        dnll2_below_1sigma_intervals = np.split(
            dnll2_below_1sigma,
            np.where(np.diff(dnll2_below_1sigma) != 1)[0] + 1,
        )

        # evaluate the scan
        if len(dnll2_below_1sigma_intervals) <= 1:
            scans.append([evaluate_likelihood_scan_1d(
                poi_values,
                dnll2_values,
                poi_min=poi_min,
                origin="entry '{}'".format(d["name"]),
            )])
        else:
            warn("{} disjoint intervals of dnll2 values below 1 sigma".format(
                len(dnll2_below_1sigma_intervals),
            ))
            # perform scans
            scan_intervals = []
            for i in range(len(dnll2_below_1sigma_intervals)):
                dnll2_values_tmp = dnll2_values.copy()

                # exclude points of other intervals
                for j in range(len(dnll2_below_1sigma_intervals)):
                    if i != j:
                        dnll2_values_tmp[dnll2_below_1sigma_intervals[j]] = 1.1
                scan_intervals.append(evaluate_likelihood_scan_1d(
                    poi_values,
                    dnll2_values_tmp,
                    poi_min=None,
                    origin="entry '{}' (interval {})".format(d["name"], i),
                ))

            # order them by minimum of dnll2
            order_dnll2 = np.argsort([
                dnll2_values[idx].min()
                for idx in dnll2_below_1sigma_intervals
            ])
            scans.append([scan_intervals[i] for i in order_dnll2])
            n_intervals = max(n_intervals, len(scan_intervals))

    # draw best fit values
    if any(scans):
        f = int(show_best_fit_error)
        g_bestfit = create_tgraph(
            n,
            [scan[0].num_min() if scan else -1e5 for scan in scans],
            [n - i - 0.5 for i in range(n)],
            [
                f * scan[0].num_min.u(direction="down", default=0.0)
                if scan
                else 0.0
                for scan in scans
            ],
            [
                f * scan[0].num_min.u(direction="up", default=0.0)
                if scan
                else 0.0
                for scan in scans
            ],
            0,
            0,
        )
        opt = lambda s: s if show_best_fit_error else ""
        r.setup_graph(g_bestfit, props={"MarkerStyle": 20, "MarkerSize": 1.2, "LineWidth": 1})
        draw_objs.append((g_bestfit, "PZ" + opt("E")))
        legend_entries.append((g_bestfit, "Best fit value", "P" + opt("L")))

        # add potential additional intervals
        if n_intervals > 1:
            for idx in range(1, n_intervals):
                g_interval = create_tgraph(
                    n,
                    [(scan[idx].num_min() if len(scan) > idx else -1e5) for scan in scans],
                    [n - i - 0.5 for i in range(n)],
                    [
                        f * scan[idx].num_min.u(direction="down", default=0.0)
                        if len(scan) > idx
                        else 0.0
                        for scan in scans
                    ],
                    [
                        f * scan[idx].num_min.u(direction="up", default=0.0)
                        if len(scan) > idx else 0.0
                        for scan in scans
                    ],
                    0,
                    0,
                )
                r.setup_graph(g_interval, props={"MarkerSize": 0., "LineWidth": 1})
                draw_objs.append((g_interval, "PZ" + opt("E")))

    # theory prediction
    if x_min < 1:
        sm_value = poi_data[scan_parameter].sm_value
        line_thy = ROOT.TLine(sm_value, 0.0, sm_value, n)
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
    label_tmpl = "%s"
    label_tmpl_scan = "#splitline{%s}{#scale[0.75]{%s = %s}}"
    label_tmpl_summary = "#splitline{%s}{#splitline{#scale[0.75]{%s = %s}}{#scale[0.65]{%s}}}"
    for i, (d, scan) in enumerate(zip(data, scans)):
        # name labels
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        if scan:
            tmpl = label_tmpl_scan
            args = (
                label,
                scan_label,
                " / ".join(
                    s.num_min.str("%.2f", style="root", force_asymmetric=True, styles={"space": ""})
                    for s in scan
                ),
            )
            if style.matches("summary") and d["name"] in hh_references:
                tmpl = label_tmpl_summary
                args += (to_root_latex(hh_references[d["name"]]),)
            label = tmpl % args
        else:
            label = label_tmpl % (label,)

        label_x = r.get_x(10, canvas)
        label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        label = ROOT.TLatex(label_x, label_y, label)
        r.setup_latex(label, props={"NDC": True, "TextAlign": 12, "TextSize": label_size})
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

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout=cms_layout,
    )
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if cms_layout.startswith("inside"):
            y_offset = 100 if cms_layout == "inside_vertical" and cms_postfix else 80
            param_kwargs = {"y_offset": y_offset}
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)


@use_style("dhi_default")
def plot_exclusion_and_bestfit_2d(
    paths,
    poi,
    scan_parameter1,
    scan_parameter2,
    expected_limits,
    observed_limits=None,
    xsec_values=None,
    xsec_levels=None,
    xsec_unit=None,
    nll_values=None,
    show_best_fit_error=False,
    recompute_best_fit=False,
    scan_min1=None,
    scan_min2=None,
    show_sm_point=True,
    interpolation_method="tgraph2d",
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    model_parameters=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a 2D plot showing excluded regions of two paramters *scan_parameter1* and
    *scan_parameter2* extracted from limits on a *poi* and saves it at *paths*. The limit values must
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
    which are drawn as well when *show_best_fit_error* is *True*. When set, it should be a mapping
    to lists of values or a record array with keys "<scan_parameter1>", "<scan_parameter2>" and
    "dnll2". By default, the position of the best value is directly extracted from the likelihood
    values. However, when *scan_min1* (*scan_min2*) is set, this best fit value is used instead,
    e.g. to use combine's internally interpolated value.

    The standard model point is drawn as well unless *show_sm_point* is *False*.
    *interpolation_method* can either be "tgraph2d" (TGraph2D), "linear" or "cubic"
    (scipy.interpolate's interp2d or griddata), or "rbf" (scipy.interpolate.Rbf). In case a tuple is
    passed, the method should be the first element, followed by optional configuration options.

    *x_min*, *x_max*, *y_min* and *y_max* define the range of the x- and y-axis, respectively, and
    default to the scan parameter ranges found in *expected_limits*. *model_parameters* can be a
    dictionary of key-value pairs of model parameters. *campaign* should refer to the name of a
    campaign label defined in *dhi.config.campaign_labels*. *cms_postfix* is shown as the postfix
    behind the CMS label.

    Supported values for *style*:

        - "paper"
        - "brazil": Brazil-band-plot color style.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/exclusion.html#2d-parameter-exclusion
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    style.color_68 = colors.grey
    style.color_95 = colors.light_grey
    if style.matches("paper"):
        cms_postfix = None
    if style.matches("brazil"):
        style.color_68 = colors.brazil_green
        style.color_95 = colors.brazil_yellow

    # convert record arrays to dicts
    def rec2dict(arr):
        if isinstance(arr, np.ndarray):
            arr = {key: arr[key] for key in arr.dtype.names}
        return arr

    expected_limits = rec2dict(expected_limits)
    nll_values = rec2dict(nll_values)
    xsec_values = rec2dict(xsec_values)
    observed_limits = rec2dict(observed_limits)

    # input checks
    assert scan_parameter1 in expected_limits
    assert scan_parameter2 in expected_limits
    assert "limit" in expected_limits
    if observed_limits:
        assert scan_parameter1 in observed_limits
        assert scan_parameter2 in observed_limits
        assert "limit" in observed_limits
    if xsec_values:
        assert scan_parameter1 in xsec_values
        assert scan_parameter2 in xsec_values
        assert "xsec" in xsec_values
    if nll_values:
        assert scan_parameter1 in nll_values
        assert scan_parameter2 in nll_values
        assert "dnll2" in nll_values

    # store content flags
    has_unc1 = "limit_p1" in expected_limits and "limit_m1" in expected_limits
    has_unc2 = "limit_p2" in expected_limits and "limit_m2" in expected_limits
    has_obs = bool(observed_limits)
    has_best_fit = bool(nll_values)

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
    pad_width = canvas.GetWindowWidth() * (1.0 - pad.GetLeftMargin() - pad.GetRightMargin())
    pad_height = canvas.GetWindowHeight() * (1.0 - pad.GetTopMargin() - pad.GetBottomMargin())
    px_to_x = (x_max - x_min) / pad_width
    py_to_y = (y_max - y_min) / pad_height

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter1].label)
    y_title = to_root_latex(poi_data[scan_parameter2].label)
    h_dummy = ROOT.TH2F("h", ";{};{};".format(x_title, y_title), 1, x_min, x_max, 1, y_min, y_max)
    r.setup_hist(h_dummy, pad=pad)
    draw_objs.append((h_dummy, ""))
    legend_entries = 6 * [(h_dummy, " ", "")]

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
            levels=[1.0],
            frame_kwargs=[{"mode": "edge"}] + [{"mode": "contour+"}],
            interpolation=interpolation_method,
        )[0]

    # +2 sigma exclusion
    if has_unc2:
        for i, g in enumerate(contours["limit_p2"]):
            r.setup_graph(g, props={"LineStyle": 2, "FillColor": style.color_95})
            draw_objs.append((g, "SAME,F"))
            if i == 0:
                legend_entries[5] = (g, "95% expected", "LF")

    # -1 and +1 sigma exclusion
    if has_unc1:
        for i, g in enumerate(contours["limit_p1"]):
            r.setup_graph(g, props={"LineStyle": 2, "FillColor": style.color_68})
            draw_objs.append((g, "SAME,F"))
            if i == 0:
                legend_entries[4] = (g, "68% expected", "LF")

        p1_col = style.color_95 if has_unc2 else colors.white
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
            interpolation=interpolation_method,
        )

        # draw them
        for graphs, level in zip(xsec_contours, xsec_levels):
            for g in graphs:
                r.setup_graph(
                    g,
                    props={"LineColor": colors.dark_grey_trans_70, "LineStyle": 3, "LineWidth": 1},
                )
                draw_objs.append((g, "SAME,C"))

        # draw labels at automatic positions
        all_positions = []
        for graphs, level in zip(xsec_contours, xsec_levels):
            # get the approximate label width
            text = str(try_int(level))
            if xsec_unit:
                text = "{} {}".format(text, xsec_unit)
            label_width, label_height = get_text_extent(text, 12, 43)
            label_width *= px_to_x
            label_height *= py_to_y

            # calculate and store the position
            label_positions = locate_contour_labels(
                graphs,
                label_width,
                label_height,
                pad_width,
                pad_height,
                x_min,
                x_max,
                y_min,
                y_max,
                other_positions=all_positions,
                label_offset=1.2,
            )
            all_positions.extend(label_positions)

            # draw them
            for x, y, rot in label_positions:
                xsec_label = ROOT.TLatex(0.0, 0.0, text)
                r.setup_latex(
                    xsec_label,
                    props={
                        "NDC": False, "TextSize": 12, "TextAlign": 22,
                        "TextColor": colors.dark_grey, "TextAngle": rot, "X": x, "Y": y,
                    },
                )
                draw_objs.append((xsec_label, "SAME"))

    # nominal exclusion
    for i, g in enumerate(contours["limit"]):
        r.setup_graph(g, props={"LineStyle": 2})
        draw_objs.append((g, "SAME,L"))
        if i == 0:
            legend_entries[3] = (g, "Excluded (expected)", "L")

    # observed exclusion
    # for testing
    # observed_limits = {
    #     scan_parameter1: expected_limits[scan_parameter1],
    #     scan_parameter2: expected_limits[scan_parameter2],
    #     "limit": expected_limits["limit"] * 1.2,
    # }
    # has_obs = True

    if has_obs:
        # get contours
        obs_contours = get_contours(
            observed_limits[scan_parameter1],
            observed_limits[scan_parameter2],
            observed_limits["limit"],
            levels=[1.0],
            frame_kwargs=[{"mode": "edge"}] + [{"mode": "contour+"}],
            interpolation=interpolation_method,
        )[0]

        # draw them
        for i, g in enumerate(obs_contours):
            # create an inverted graph to close the outer polygon
            g_inv = invert_graph(g, x_axis=h_dummy.GetXaxis(), y_axis=h_dummy.GetYaxis())
            r.setup_graph(g, props={"LineStyle": 1})
            r.setup_graph(g_inv, props={"LineStyle": 1, "FillColor": colors.blue_signal_trans})
            draw_objs.append((g, "SAME,L"))
            draw_objs.append((g_inv, "SAME,F"))
            if i == 0:
                legend_entries[0] = (g_inv, "Excluded (observed)", "AF")

    # best fit point
    if nll_values:
        # preprocess values
        dnll2, nll_scan_values1, nll_scan_values2 = _preprocess_values(
            nll_values["dnll2"],
            (scan_parameter1, nll_values[scan_parameter1]),
            (scan_parameter2, nll_values[scan_parameter2]),
            remove_nans=True,
            shift_negative_values=True,
            min_is_external=scan_min1 is None or scan_min2 is None,
        )

        # scan
        scan = evaluate_likelihood_scan_2d(
            nll_scan_values1,
            nll_scan_values2,
            dnll2,
            poi1_min=scan_min1 if show_best_fit_error else None,
            poi2_min=scan_min2 if show_best_fit_error else None,
        )

        if scan:
            g_fit = ROOT.TGraphAsymmErrors(1)
            g_fit.SetPoint(0, scan.num1_min(), scan.num2_min())
            if show_best_fit_error:
                if scan.num1_min.uncertainties:
                    g_fit.SetPointEXhigh(0, scan.num1_min.u(direction="up"))
                    g_fit.SetPointEXlow(0, scan.num1_min.u(direction="down"))
                if scan.num2_min.uncertainties:
                    g_fit.SetPointEYhigh(0, scan.num2_min.u(direction="up"))
                    g_fit.SetPointEYlow(0, scan.num2_min.u(direction="down"))
                r.setup_graph(g_fit, props={"FillStyle": 0}, color=colors.black)
                draw_objs.append((g_fit, "PEZ"))
                legend_entries[1] = (g_fit, "Best fit value", "PEL")
            else:
                r.setup_graph(
                    g_fit,
                    props={"FillStyle": 0, "MarkerStyle": 43, "MarkerSize": 2},
                    color=colors.black,
                )
                draw_objs.append((g_fit, "PZ"))
                legend_entries[1] = (g_fit, "Best fit value", "P")
        else:
            warn("2D likelihood evaluation failed")

    # SM point
    if show_sm_point:
        g_sm = create_tgraph(
            1,
            poi_data[scan_parameter1]["sm_value"],
            poi_data[scan_parameter2]["sm_value"],
        )
        r.setup_graph(g_sm, props={"MarkerStyle": 33, "MarkerSize": 2.5}, color=colors.red)
        draw_objs.insert(-1, (g_sm, "P"))
        legend_entries[2 if has_best_fit else 1] = (g_sm, "Standard Model", "P")

    # legend
    legend = r.routines.create_legend(pad=pad, width=480, n=3, x2=-44, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "lrt",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70},
    )
    draw_objs.insert(-1, legend_box)

    # cms label
    cms_layout = "outside_horizontal"
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout=cms_layout,
    )
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if cms_layout.startswith("inside"):
            y_offset = 100 if cms_layout == "inside_vertical" and cms_postfix else 80
            param_kwargs = {"y_offset": y_offset}
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

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
    for path in make_list(paths):
        canvas.SaveAs(path)


def get_auto_contour_levels(values, steps=(1,)):
    min_value = max(min(values), 1e-3)
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
