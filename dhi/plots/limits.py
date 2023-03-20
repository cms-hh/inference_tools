# coding: utf-8

"""
Limit plots using ROOT.
"""

import json
import math
import traceback
from collections import OrderedDict
from itertools import chain

import six
import numpy as np
import scipy.interpolate
from scinum import Number

from dhi.config import (
    poi_data, br_hh_names, br_hh_colors, campaign_labels, colors, color_sequence, marker_sequence,
    bm_labels, hh_references,
)
from dhi.util import (
    import_ROOT, DotDict, to_root_latex, create_tgraph, colored, minimize_1d, unique_recarray,
    make_list, try_int, dict_to_recarray, prepare_output, make_tuple,
)
from dhi.plots.util import (
    use_style, create_model_parameters, create_hh_xsbr_label, determine_limit_digits,
    get_graph_points, get_y_range, get_contours, fill_hist_from_points, infer_binning_from_grid,
    Style,
)
import dhi.hepdata_tools as hdt


colors = colors.root


@use_style("dhi_default")
def plot_limit_scan(
    paths,
    poi,
    scan_parameter,
    expected_values,
    observed_values=None,
    theory_values=None,
    ranges_path=None,
    hep_data_path=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    xsec_unit=None,
    hh_process=None,
    show_excluded_ranges=False,
    show_points=False,
    model_parameters=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot for the upper limit scan of a *poi* over a *scan_parameter* and saves it at
    *paths*. *expected_values* should be a mapping to lists of values or a record array with keys
    "<scan_parameter>" and "limit", and optionally "limit_p1" (plus 1 sigma), "limit_m1" (minus 1
    sigma), "limit_p2" and "limit_m2". When the variations by 1 or 2 sigma are missing, the plot is
    created without them. When *observed_values* is set, it should have a similar format with keys
    "<scan_parameter>" and "limit". When *theory_values* is set, it should have a similar format
    with keys "<scan_parameter>" and "xsec", and optionally "xsec_p1" and "xsec_m1". When
    *ranges_path* is set, allowed scan parameter ranges are saved to the given file. When
    *hep_data_path* is set, a yml data file compatible with the HEPData format
    (https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html) is stored at that path.

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *x_min*, *x_max*,
    *y_min* and *y_max* define the axes ranges and default to the ranges of the given values.
    *xsec_unit* denotes whether the passed values are given as real cross sections in this unit or,
    when *None*, as a ratio over the theory prediction. *hh_process* can be the name of a HH
    subprocess configured in *dhi.config.br_hh_names* and is inserted to the process name
    in the title of the y-axis and indicates that the plotted cross section data was (e.g.) scaled
    by a branching ratio.

    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_excluded_ranges* is *True* the excluded ranges are marked with a hatched area. The
    intersection of the theory prediction with observed values is used when present, otherwise the
    expected ones are taken. When *show_points* is *True*, the central scan points are drawn on top
    of the interpolated curve. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#limit-on-poi-vs-scan-parameter  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # input checks
    def check_values(values, keys=None):
        # convert record array to dict mapping to arrays
        if isinstance(values, np.ndarray):
            values = {key: values[key] for key in values.dtype.names}
        assert scan_parameter in values
        if keys:
            assert all(key in values for key in keys)
        return values

    expected_values = check_values(expected_values, ["limit"])
    has_obs = observed_values is not None
    if has_obs:
        observed_values = check_values(observed_values, ["limit"])
    has_thy = theory_values is not None
    has_thy_err = False
    if theory_values is not None:
        theory_values = check_values(theory_values, ["xsec"])
        has_thy_err = "xsec_p1" in theory_values and "xsec_m1" in theory_values

    # prepare hep data
    hep_data = None
    if hep_data_path:
        hep_data = hdt.create_hist_data()

    # set default ranges
    x_min_scan = min(expected_values[scan_parameter])
    x_max_scan = max(expected_values[scan_parameter])
    if has_obs:
        x_min_scan = min(x_min_scan, min(observed_values[scan_parameter]))
        x_max_scan = max(x_max_scan, max(observed_values[scan_parameter]))
    x_min = x_min_scan if x_min is None else x_min
    x_max = x_max_scan if x_max is None else x_max

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "95% CL limit on {} {}".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex("({})".format(xsec_unit)) if xsec_unit else "/ #sigma_{Theory}",
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # setup up to 6 legend entries that are inserted by index downstream
    # setup two lists of legend entries
    legend_entries_l = []
    legend_entries_r = []

    # helper to read values into graphs
    def create_graph(values=expected_values, key="limit", sigma=None, pad=True, insert=None):
        return create_tgraph(
            len(values[key]),
            values[scan_parameter],
            values[key],
            0,
            0,
            (values[key] - values["{}_m{}".format(key, sigma)]) if sigma else 0,
            (values["{}_p{}".format(key, sigma)] - values[key]) if sigma else 0,
            pad=pad,
            insert=insert,
        )

    # 2 sigma band
    g_2sigma = None
    if "limit_p2" in expected_values and "limit_m2" in expected_values:
        g_2sigma = create_graph(sigma=2)
        r.setup_graph(
            g_2sigma,
            props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.brazil_yellow},
        )
        draw_objs.append((g_2sigma, "SAME,4"))  # option 4 might fallback to 3, see below
        legend_entries_r.insert(0, (g_2sigma, "95% expected", "LF"))
        y_max_value = max(y_max_value, max(expected_values["limit_p2"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m2"]))

    # 1 sigma band
    g_1sigma = None
    if "limit_p1" in expected_values and "limit_m1" in expected_values:
        g_1sigma = create_graph(sigma=1)
        r.setup_graph(
            g_1sigma,
            props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.brazil_green},
        )
        draw_objs.append((g_1sigma, "SAME,4"))  # option 4 might fallback to 3, see below
        legend_entries_r.insert(0, (g_1sigma, "68% expected", "LF"))
        y_max_value = max(y_max_value, max(expected_values["limit_p1"]))
        y_min_value = min(y_min_value, min(expected_values["limit_m1"]))

    # central values
    g_exp = create_graph()
    r.setup_graph(
        g_exp,
        props={"LineWidth": 2, "LineStyle": 2},
    )
    draw_objs.append((g_exp, "SAME,CP" if show_points else "SAME,C"))
    legend_entries_r.insert(0, (g_exp, "Median expected", "L"))
    y_max_value = max(y_max_value, max(expected_values["limit"]))
    y_min_value = min(y_min_value, min(expected_values["limit"]))

    # print the expected excluded ranges
    excl_ranges_exp = evaluate_excluded_ranges(
        scan_parameter,
        poi + " expected",
        expected_values[scan_parameter],
        expected_values["limit"],
        theory_values[scan_parameter] if has_thy else None,
        theory_values["xsec"] if has_thy else None,
    )
    allowed_ranges = OrderedDict()
    key_exp = "__".join([scan_parameter, poi, "expected"])
    allowed_ranges[key_exp] = excluded_to_allowed_ranges(excl_ranges_exp, x_min_scan, x_max_scan)

    # observed values
    if has_obs:
        g_obs = create_graph(values=observed_values)
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,CP" if show_points else "SAME,C"))
        legend_entries_l.append((g_obs, "Observed", "L"))
        y_max_value = max(y_max_value, max(observed_values["limit"]))
        y_min_value = min(y_min_value, min(observed_values["limit"]))

        # print the observed excluded ranges
        excl_ranges_obs = evaluate_excluded_ranges(
            scan_parameter,
            poi + " observed",
            observed_values[scan_parameter],
            observed_values["limit"],
            theory_values[scan_parameter] if has_thy else None,
            theory_values["xsec"] if has_thy else None,
        )
        key_obs = "__".join([scan_parameter, poi, "observed"])
        allowed_ranges[key_obs] = excluded_to_allowed_ranges(excl_ranges_obs, x_min_scan, x_max_scan)

    if ranges_path:
        ranges_path = prepare_output(ranges_path)
        with open(ranges_path, "w") as f:
            json.dump(allowed_ranges, f, indent=4)
        print("saved allowed ranges to {}".format(ranges_path))

    # get theory prediction limits
    if has_thy:
        y_min_value = min(y_min_value, min(theory_values["xsec_m1" if has_thy_err else "xsec"]))

    # set limits
    y_min, y_max, y_max_vis = get_y_range(
        y_min_value,
        y_max_value,
        y_min,
        y_max,
        log=y_log,
        visible_margin=0.30,  # visible_margin depends on the legend height
    )
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # draw option 4 if error graphs are buggy, i.e. when the error point or bar is above the
    # visible, vertical range of the pad, so check these cases and fallback to option 3
    fallback_graph_option = False
    for g in filter(bool, [g_2sigma, g_1sigma]):
        _, y_values, _, _, _, y_errors_up = get_graph_points(g, errors=True)
        if any((y + y_err_up) >= y_max for y, y_err_up in zip(y_values, y_errors_up)):
            fallback_graph_option = True
            break
    if fallback_graph_option:
        for i, objs in enumerate(list(draw_objs)):
            if isinstance(objs, tuple) and len(objs) >= 2 and objs[0] in [g_2sigma, g_1sigma]:
                draw_objs[i] = (objs[0], "SAME,3") + objs[2:]
                print(
                    "{}: changed draw option of graph {} to '3' as it exceeds the vertical pad "
                    "range which is not supported for option '4'".format(
                        colored("WARNING", "yellow"), objs[0],
                    ),
                )

    # show hatched areas for excluded ranges if requested
    if show_excluded_ranges:
        g, excl_ranges = (g_obs, excl_ranges_obs) if has_obs else (g_exp, excl_ranges_exp)

        # create a spline for neat interpolation (using g.Eval(x, 0, "S") fails miserably)
        # delete repeated endpoints which leads to interpolation failures
        # gx, gy = get_graph_points(g)
        # if len(gx) > 1 and gx[0] == gx[1]:
        #     gx, gy = gx[1:], gy[1:]
        # if len(gx) > 1 and gx[-1] == gx[-2]:
        #     gx, gy = gx[:-1], gy[:-1]
        # spline = ROOT.TSpline3("spline", create_tgraph(len(gx), gx, gy), "", gx[0], gx[-1])

        # show lines at positions of crossing and mark each excluded range section with a small
        # hatched area of fixed width
        def add_line(x):
            l = ROOT.TLine(x, 0, x, y_max_vis)
            r.setup_line(
                l,
                props={"NDC": False, "LineWidth": 1},
                color=colors.dark_grey_trans_70,
            )
            draw_objs.insert(-1, l)

        def add_area(x1, x2):
            g = create_tgraph(2, [x1, x2], [y_max_vis, y_max_vis], 0, 0, y_max_vis, 0, pad=True)
            r.setup_graph(
                g,
                color=colors.dark_grey_trans_70,
                color_flags="f",
                props={"FillStyle": 3345, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0},
            )
            draw_objs.insert(-1, (g, "SAME,4"))
            return g

        def add_label(x, text):
            l = ROOT.TLatex(x, y_max_vis, text)
            r.setup_latex(l, props={"NDC": False, "TextAlign": 23, "TextSize": 18})
            draw_objs.append(l)

        g_excl = None
        for start, stop in excl_ranges:
            # wx = width of the x axis
            # wr = width of excluded range
            wx = x_max - x_min
            wr = stop - start
            width = 0.04 * wx
            on_edge = start == x_min or stop == x_max
            width = min(width, wr if on_edge else (0.4 * wr))
            f = 0.0
            if start > x_min:
                # g_excl = add_area(start, start + width)
                add_line(start)
                f += 1
            if stop < x_max:
                # g_excl = add_area(stop, stop - width)
                add_line(stop)
                f -= 1
            # add a label
            add_label(0.5 * (start + stop + f * width), "Excluded")
        if g_excl:
            legend_entries_l.append((g_excl, "Excluded", "F"))

    # theory prediction
    if has_thy:
        if has_thy_err:
            # when the maximum value is far above the maximum y range, ROOT will fail drawing the
            # first point correctly, so insert two values that are so off that it does not matter
            insert = [(0, -1e7, 0, 0, 0, 0, 0)] if max(theory_values["xsec"]) > y_max else None
            g_thy = create_graph(values=theory_values, key="xsec", sigma=1, insert=insert)
            r.setup_graph(
                g_thy,
                props={
                    "LineWidth": 2, "LineStyle": 1, "LineColor": colors.red, "FillStyle": 1001,
                    "FillColor": colors.red_trans_50,
                },
            )
            draw_objs.append((g_thy, "SAME,C3"))
            legend_entry = (g_thy, "Theory prediction", "LF")
        else:
            g_thy = create_graph(values=theory_values, key="xsec")
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red})
            draw_objs.append((g_thy, "SAME,C"))
            legend_entry = (g_thy, "Theory prediction", "L")
        # only add to the legend if values are in terms of a cross section
        if xsec_unit:
            legend_entries_l.append(legend_entry)

    # fill hep data
    if hep_data:
        poi_div = "" if xsec_unit else r" / $\sigma_{Theory}$"
        poi_qual = create_hh_xsbr_label(poi, hh_process) + poi_div
        qualifiers = lambda: [hdt.create_qualifier("POI", poi_qual)]

        # data entries
        scan_values = list(map(float, expected_values[scan_parameter]))
        hdt.create_independent_variable(
            poi_data[scan_parameter].label,
            parent=hep_data,
            values=[Number(v, default_format=-2) for v in scan_values],
        )

        # theory prediction
        if has_thy and xsec_unit:
            l = "th" if has_thy_err else None
            hdt.create_dependent_variable_from_graph(
                g_thy,
                x_values=scan_values,
                error_label=l,
                label="Theory prediction",
                unit=xsec_unit,
                parent=hep_data,
                qualifiers=qualifiers(),
                rounding_method="publication",
            )

        def exp_limit_value(i, sigma, label):
            keys = ["limit", "limit_p{}".format(sigma), "limit_m{}".format(sigma)]
            n, p, m = map(float, (expected_values[key][i] for key in keys))
            return Number(n, {label: (p - n, n - m)}, default_format="publication")

        # expected limits with 68% error
        hdt.create_dependent_variable(
            "Expected limit, 68%",
            parent=hep_data,
            unit=xsec_unit,
            values=[exp_limit_value(i, 1, "68%") for i in range(len(scan_values))],
            qualifiers=qualifiers(),
        )

        # expected limits with 95% error
        hdt.create_dependent_variable(
            "Expected limit, 95%",
            parent=hep_data,
            unit=xsec_unit,
            values=[exp_limit_value(i, 2, "95%") for i in range(len(scan_values))],
            qualifiers=qualifiers(),
        )

        # observed limits
        if has_obs:
            hdt.create_dependent_variable(
                "Observed limit",
                parent=hep_data,
                unit=xsec_unit,
                values=[Number(float(v), default_format=3) for v in observed_values["limit"]],
                qualifiers=qualifiers(),
            )

    # legend
    legend_entries_l += (3 - len(legend_entries_l)) * [(h_dummy, " ", "L")]
    legend_entries_r += (3 - len(legend_entries_r)) * [(h_dummy, " ", "L")]
    legend = r.routines.create_legend(pad=pad, width=440, n=3, props={"NColumns": 2})
    r.fill_legend(legend, legend_entries_l + legend_entries_r)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "trl",
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
    r.routines.draw_objects(draw_objs)

    # save plots
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)

    # save hep data
    if hep_data_path:
        hdt.save_hep_data(hep_data, hep_data_path)


@use_style("dhi_default")
def plot_limit_scans(
    paths,
    poi,
    scan_parameter,
    names,
    expected_values,
    observed_values=None,
    theory_values=None,
    ranges_path=None,
    hep_data_path=None,
    y_log=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    xsec_unit=None,
    hh_process=None,
    model_parameters=None,
    campaign=None,
    show_points=True,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing multiple upper limit scans of a *poi* over a *scan_parameter* and saves
    it at *paths*. *expected_values* should be a list of mappings to lists of values or a record
    array with keys "<scan_parameter>" and "limit". Each mapping in *expected_values* will result in
    a different curve. When *observed_values* is set, it should have a similar format with keys
    "<scan_parameter>" and "limit", where the i-th element is corresponding to the i-th element in
    *expected_values*. When *theory_values* is set, it should have a similar format with keys
    "<scan_parameter>" and "xsec", and optionally "xsec_p1" and "xsec_m1". *names* denote the names
    of limit curves shown in the legend. When a name is found to be in dhi.config.br_hh_names, its
    value is used as a label instead. When *ranges_path* is set, all allowed scan parameter ranges
    are saved to the given file. When *hep_data_path* is set, a yml data file compatible with the
    HEPData format (https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html) is stored at
    that path.

    When *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *x_min*, *x_max*,
    *y_min* and *y_max* define the axis ranges and default to the range of the given values.
    *xsec_unit* denotes whether the passed values are given as real cross sections in this unit or,
    when *None*, as a ratio over the theory prediction. *hh_process* can be the name of a HH
    subprocess configured in *dhi.config.br_hh_names* and is inserted to the process name in the
    title of the y-axis and indicates that the plotted cross section data was (e.g.) scaled by a
    branching ratio.

    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in dhi.config.campaign_labels. When *show_points*
    is *True*, the central scan points are drawn on top of the interpolated curve. *cms_postfix* is
    shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#multiple-limits-on-poi-vs-scan-parameter  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # convert record arrays to dicts mapping to arrays
    def check_values(values):
        _values = []
        for v in values:
            if v is not None:
                if isinstance(v, np.ndarray):
                    v = {key: v[key] for key in v.dtype.names}
                assert "limit" in v
                assert scan_parameter in v
            _values.append(v)
        return _values

    # input checks
    expected_values = check_values(expected_values)
    n_graphs = len(expected_values)
    assert n_graphs >= 1
    assert len(names) == n_graphs
    has_obs = False
    if observed_values:
        assert len(observed_values) == n_graphs
        observed_values = check_values(observed_values)
        has_obs = any(v is not None for v in observed_values)
    scan_values = expected_values[0][scan_parameter]
    has_thy = theory_values is not None
    has_thy_err = False
    if theory_values is not None:
        # convert record array to dicts mapping to arrays
        if isinstance(theory_values, np.ndarray):
            theory_values = {
                key: theory_values[key]
                for key in theory_values.dtype.names
            }
        assert scan_parameter in theory_values
        assert "xsec" in theory_values
        has_thy_err = "xsec_p1" in theory_values and "xsec_m1" in theory_values
    allowed_ranges = OrderedDict()

    # prepare hep data
    hep_data = None
    if hep_data_path:
        hep_data = hdt.create_hist_data()

    # set default ranges
    if x_min is None:
        x_min = min(min(ev[scan_parameter]) for ev in expected_values)
    if x_max is None:
        x_max = max(max(ev[scan_parameter]) for ev in expected_values)

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
    pad.cd()
    draw_objs = []
    legend_entries = []
    y_max_value = -1e5
    y_min_value = 1e5

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter].label)
    y_title = "95% CL limit on {} {}".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex("({})".format(xsec_unit)) if xsec_unit else "/ #sigma_{Theory}",
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0})
    draw_objs.append((h_dummy, "HIST"))

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(name in br_hh_colors.root for name in names):
        _color_sequence = [br_hh_colors.root[name] for name in names]

    # central values
    g_exps, g_obss = [], []
    for i, (ev, name, col, ms) in enumerate(zip(
        expected_values, names, _color_sequence[:n_graphs], marker_sequence[:n_graphs],
    )):
        # expected graph
        mask = ~np.isnan(ev["limit"])
        limit_values = ev["limit"][mask]
        scan_values = ev[scan_parameter][mask]
        n_nans = (~mask).sum()
        if n_nans:
            print("WARNING: found {} NaN(s) in expected limit values at index {}".format(n_nans, i))
        g_exp = create_tgraph(mask.sum(), scan_values, limit_values)
        r.setup_graph(
            g_exp,
            color=colors[col],
            props={
                "LineWidth": 2, "MarkerStyle": ms, "MarkerSize": 1.2,
                "LineStyle": 2 if has_obs else 1,
            },
        )
        draw_objs.append((g_exp, "SAME,CP" if show_points and not has_obs else "SAME,C"))
        legend_entries.append((
            g_exp,
            to_root_latex(br_hh_names.get(name, name)),
            "LP" if show_points and not has_obs else "L",
        ))
        g_exps.append(g_exp)
        y_max_value = max(y_max_value, max(limit_values))
        y_min_value = min(y_min_value, min(limit_values))

        # print expected excluded ranges
        excl_ranges = evaluate_excluded_ranges(
            scan_parameter,
            "{}, {}, expected".format(poi, name),
            scan_values,
            limit_values,
            theory_values[scan_parameter] if has_thy else None,
            theory_values["xsec"] if has_thy else None,
        )
        if ranges_path:
            key = "__".join([scan_parameter, poi, "expected", name])
            allowed_ranges[key] = excluded_to_allowed_ranges(
                excl_ranges,
                scan_values.min(),
                scan_values.max(),
            )

        # observed graph
        ov = observed_values[i]
        g_obss.append(None)
        if ov is not None:
            obs_mask = ~np.isnan(ov["limit"])
            obs_limit_values = ov["limit"][obs_mask]
            obs_scan_values = ov[scan_parameter][obs_mask]
            n_nans = (~obs_mask).sum()
            if n_nans:
                print("WARNING: found {} NaN(s) in observed limit values at index {}".format(
                    n_nans, i,
                ))
            g_obs = create_tgraph(obs_mask.sum(), obs_scan_values, obs_limit_values)
            r.setup_graph(
                g_obs,
                color=colors[col],
                props={"LineWidth": 2, "MarkerStyle": ms, "MarkerSize": 1.2},
            )
            draw_objs.append((g_obs, "SAME,CP" if show_points else "SAME,C"))
            legend_entries[-1] = (
                g_obs,
                to_root_latex(br_hh_names.get(name, name)),
                "LP" if show_points else "L",
            )
            g_obss[-1] = g_obs
            y_max_value = max(y_max_value, max(obs_limit_values))
            y_min_value = min(y_min_value, min(obs_limit_values))

            # print observed excluded ranges
            excl_ranges = evaluate_excluded_ranges(
                scan_parameter,
                "{}, {}, observed".format(poi, name),
                obs_scan_values,
                obs_limit_values,
                theory_values[scan_parameter] if has_thy else None,
                theory_values["xsec"] if has_thy else None,
            )
            if ranges_path:
                key = "__".join([scan_parameter, poi, "observed", name])
                allowed_ranges[key] = excluded_to_allowed_ranges(
                    excl_ranges,
                    obs_scan_values.min(),
                    obs_scan_values.max(),
                )

    if ranges_path:
        ranges_path = prepare_output(ranges_path)
        with open(ranges_path, "w") as f:
            json.dump(allowed_ranges, f, indent=4)
        print("saved allowed ranges to {}".format(ranges_path))

    # add additional legend entries to distinguish expected and observed lines
    if has_obs:
        g_exp_dummy = g_exp.Clone()
        g_obs_dummy = g_obs.Clone()
        r.apply_properties(g_exp_dummy, {"LineColor": colors.black})
        r.apply_properties(g_obs_dummy, {"LineColor": colors.black})
        legend_entries.append((g_exp_dummy, "expected", "L"))
        legend_entries.append((g_obs_dummy, "observed", "L"))

    # get theory prediction limits
    if has_thy:
        y_min_value = min(y_min_value, min(theory_values["xsec_m1" if has_thy_err else "xsec"]))

    # set axis limits
    y_min, y_max, _ = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)
    h_dummy.SetMinimum(y_min)
    h_dummy.SetMaximum(y_max)

    # draw the theory prediction
    if has_thy:
        scan_values_thy = theory_values[scan_parameter]
        if has_thy_err:
            # when the maximum value is far above the maximum y range, ROOT will fail drawing the
            # first point correctly, so insert two values that are so off that it does not matter
            insert = [(0, -1e7, 0, 0, 0, 0, 0)] if max(theory_values["xsec"]) > y_max else None
            g_thy = create_tgraph(
                len(scan_values_thy),
                scan_values_thy,
                theory_values["xsec"],
                0,
                0,
                theory_values["xsec"] - theory_values["xsec_m1"],
                theory_values["xsec_p1"] - theory_values["xsec"],
                pad=True,
                insert=insert,
            )
            r.setup_graph(
                g_thy,
                props={
                    "LineWidth": 2, "LineStyle": 1, "LineColor": colors.red, "FillStyle": 1001,
                    "FillColor": colors.red_trans_50,
                },
            )
            draw_objs.insert(1, (g_thy, "SAME,C3"))
            legend_entry = (g_thy, "Theory prediction", "LF")
        else:
            g_thy = create_tgraph(len(scan_values_thy), scan_values_thy, theory_values["xsec"])
            r.setup_graph(g_thy, props={"LineWidth": 2, "LineStyle": 1}, color=colors.red)
            draw_objs.insert(1, (g_thy, "SAME,C"))
            legend_entry = (g_thy, "Theory prediction", "L")
        # only add to the legend if values are in terms of a cross section
        if xsec_unit:
            legend_entries.append(legend_entry)

    # fill hep data
    if hep_data:
        poi_div = "" if xsec_unit else r" / $\sigma_{Theory}$"
        poi_qual = create_hh_xsbr_label(poi, hh_process) + poi_div
        qualifiers = lambda: [hdt.create_qualifier("POI", poi_qual)]

        # scan value as independent variable
        scan_values = sorted(set(chain(*(map(float, ev[scan_parameter]) for ev in expected_values))))
        hdt.create_independent_variable(
            poi_data[scan_parameter].label,
            parent=hep_data,
            values=[Number(v, default_format=-2) for v in scan_values],
        )

        # theory prediction
        if has_thy and xsec_unit:
            l = "th" if has_thy_err else None
            hdt.create_dependent_variable_from_graph(
                g_thy,
                x_values=scan_values,
                error_label=l,
                label="Theory prediction",
                unit=xsec_unit,
                parent=hep_data,
                qualifiers=qualifiers(),
                rounding_method="publication",
            )

        # limits per entry
        for name, g_exp, g_obs in zip(names, g_exps, g_obss):
            # expected
            label = name + (", expected" if has_obs else "")
            hdt.create_dependent_variable_from_graph(
                g_exp,
                x_values=scan_values,
                parent=hep_data,
                label=label,
                rounding_method=-2,
            )

            # observed
            if g_obs:
                label = name + ", observed"
                hdt.create_dependent_variable_from_graph(
                    g_obs,
                    x_values=scan_values,
                    parent=hep_data,
                    label=label,
                    rounding_method=-2,
                )

    # legend
    legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
    legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
    legend = r.routines.create_legend(
        pad=pad,
        width=legend_cols * 210,
        n=legend_rows,
        props={"NColumns": legend_cols, "TextSize": 18},
    )
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "trl",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70},
    )
    draw_objs.insert(-1, legend_box)

    # cms label
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout="outside_horizontal",
    )
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        if legend_cols == 3:
            param_kwargs["y_offset"] = 1. - 0.25 * pad.GetTopMargin() - legend.GetY1()
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save plots
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)

    # save hep data
    if hep_data_path:
        hdt.save_hep_data(hep_data, hep_data_path)


@use_style("dhi_default")
def plot_limit_points(
    paths,
    poi,
    data,
    hep_data_path=None,
    sort_by=None,
    x_min=None,
    x_max=None,
    x_log=False,
    xsec_unit=None,
    hh_process=None,
    pad_width=None,
    left_margin=None,
    right_margin=None,
    entry_height=None,
    label_size=None,
    model_parameters=None,
    h_lines=None,
    campaign=None,
    digits=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing a comparison of limits of multiple analysis (or channels) on a *poi* and
    saves it at *paths*. *data* should be a list of dictionaries with fields

    - "expected", a sequence of five values, i.e., central limit, and +1 sigma, -1 sigma, +2 sigma,
      and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a single value,
    - "theory" (optional), a single value or a sequence of three values, i.e., nominal value, and
      +1 sigma and -1 sigma variations (absolute values, not errors!),
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead,
    - "label", an extra label shown on the right side of the plot.

    Example:

    .. code-block:: python

        plot_limit_points(
            paths=["plot.pdf", "plot.png"],
            poi="r",
            data=[{
                "expected": (40., 50., 28., 58., 18.),
                "observed": 45.,
                "theory": (38., 40., 36.),
                "name": "bbXX",
                "label": "CMS-HIG-XX-YYY",
            }, {
                ...
            }],
        )

    When *hep_data_path* is set, a yml data file compatible with the HEPData format
    (https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html) is stored at that path.
    The entries can be automatically sorted by setting *sort_by* to either ``"expected"`` or, when
    existing in *data*, ``"observed"``. When *x_log* is *True*, the x-axis is scaled
    logarithmically. *x_min* and *x_max* define the range of the x-axis and default to the maximum
    range of values passed in data, including uncertainties. *xsec_unit* denotes whether the passed
    values are given as real cross sections in this unit or, when *None*, as a ratio over the theory
    prediction. *hh_process* can be the name of a HH subprocess configured in
    *dhi.config.br_hh_names* and is inserted to the process name in the title of the x-axis and
    indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *pad_width*, *left_margin*, *right_margin*, *entry_height* and *label_size* can be set to a size
    in pixels to overwrite internal defaults.

    *model_parameters* can be a dictionary of key-value pairs of model parameters. *h_lines* can be
    a list of integers denoting positions where additional horizontal lines are drawn for visual
    guidance. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels. *digits* controls the number of digits of the limit values shown for
    each entry. When *None*, a number based on the lowest limit values is determined automatically.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#multiple-limits-at-a-certain-point-of-parameters  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check inputs and get extrema
    n = len(data)
    has_obs = False
    has_thy = False
    has_thy_err = False
    x_min_value = 1e5
    x_max_value = -1e5
    for d in data:
        assert "name" in d
        assert "expected" in d
        x_min_value = min(x_min_value, min(d["expected"]))
        x_max_value = max(x_max_value, max(d["expected"]))
        if "observed" in d:
            assert isinstance(d["observed"], (float, int))
            if not np.isnan(d["observed"]) and d["observed"] is not None:
                has_obs = True
                x_min_value = min(x_min_value, d["observed"])
                x_max_value = max(x_max_value, d["observed"])
            d["observed"] = [d["observed"]]
        if "theory" in d:
            if isinstance(d["theory"], (tuple, list)):
                if len(d["theory"]) == 3:
                    has_thy_err = True
                else:
                    assert len(d["theory"]) == 1
            else:
                d["theory"] = 3 * (d["theory"],)
            has_thy = True
            x_min_value = min(x_min_value, min(d["theory"]))
            x_max_value = max(x_max_value, max(d["theory"]))

    # sort data
    if sort_by == "expected":
        data.sort(key=lambda d: -d["expected"][0] if "expected" in d else -1e6)
    elif sort_by == "observed" and has_obs:
        data.sort(key=lambda d: -d["observed"][0] if "observed" in d else -1e6)

    # prepare hep data
    hep_data = None
    if hep_data_path:
        hep_data = hdt.create_hist_data()

    # set default ranges
    if x_min is None:
        if not xsec_unit:
            x_min = 0.75 if x_log else 0.
        else:
            x_min = 0.75 * x_min_value
    if x_max is None:
        x_max = x_max_value * 1.33

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
        "Logx": x_log,
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

    # dummy histogram to control axes
    x_title = "95% CL limit on {} {}".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex("({})".format(xsec_unit)) if xsec_unit else "/ #sigma_{Theory}",
    )
    h_dummy = ROOT.TH1F("dummy", ";{};".format(x_title), 1, x_min, x_max)
    r.setup_hist(
        h_dummy,
        pad=pad,
        props={"LineWidth": 0, "Maximum": y_max},
    )
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"TitleOffset": 1.2, "LabelOffset": r.pixel_to_coord(canvas, y=4), "NoExponent": True},
    )
    draw_objs.append((h_dummy, "HIST"))

    # show custom x axis values in log mode
    if x_log:
        additional_x_values = []
        # additional_x_values = [2, 3, 4, 5, 6, 50, 500]
        for v in additional_x_values:
            tick_label = ROOT.TLatex(float(v), -0.08, str(v))
            r.setup_latex(
                tick_label,
                props={"NDC": False, "TextAlign": 23, "TextSize": h_dummy.GetXaxis().GetLabelSize()},
            )
            draw_objs.append(tick_label)

    # setup up to 6 legend entries that are inserted by index downstream
    legend_entries = 6 * [(h_dummy, " ", "L")]

    # helper to read values into graphs
    def create_graph(key="expected", sigma=None):
        # repeat the edges by padding to prevent bouncing effects of interpolated lines
        _data = [d for d in data if key in d]
        n = len(_data)
        zeros = np.zeros(n, dtype=np.float32)

        limits = np.array([d[key][0] for d in _data], dtype=np.float32)
        y = np.arange(n, dtype=np.float32)[::-1]
        x_err_u, x_err_d = zeros, zeros
        if key == "observed":
            y = np.arange(n, dtype=np.float32)[::-1] + 0.5
            y_err_u, y_err_d = zeros + 0.5, zeros + 0.5
        else:
            y = np.arange(n, dtype=np.float32)[::-1]
            y_err_u, y_err_d = zeros + 1, zeros
            if sigma:
                x_err_d = np.array([d[key][sigma * 2] for d in _data], dtype=np.float32)
                x_err_d = limits - x_err_d
                x_err_u = np.array([d[key][sigma * 2 - 1] for d in _data], dtype=np.float32)
                x_err_u = x_err_u - limits

        return create_tgraph(n, limits, y, x_err_d, x_err_u, y_err_d, y_err_u)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(
        g_2sigma,
        props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.brazil_yellow},
    )
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries[5] = (g_2sigma, r"95% expected", "LF")

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(
        g_1sigma,
        props={"LineWidth": 2, "LineStyle": 2, "FillColor": colors.brazil_green},
    )
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries[4] = (g_1sigma, r"68% expected", "LF")

    # central values
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 2})
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries[3] = (g_exp, "Median expected", "L")

    # observed values
    if has_obs:
        g_obs = create_graph(key="observed")
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,PEZ"))
        legend_entries[0] = (g_obs, "Observed", "PL")

    # vertical line for theory prediction, represented by a graph in case of uncertainties
    if has_thy and any((d["theory"][0] >= x_min) for d in data):
        # uncertainty line
        g_thy_line = create_graph(key="theory")
        r.setup_graph(g_thy_line, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors.red})
        draw_objs.append((g_thy_line, "SAME,LZ"))
        legend_entry = (g_thy_line, "Theory prediction", "L")
        # uncertainty area
        if has_thy_err:
            g_thy_area = create_graph(key="theory", sigma=1)
            r.setup_graph(
                g_thy_area,
                props={
                    "LineWidth": 2, "LineStyle": 1, "LineColor": colors.red, "FillStyle": 1001,
                    "FillColor": colors.red_trans_50,
                },
            )
            draw_objs.append((g_thy_area, "SAME,2"))
            legend_entry = (g_thy_area, "Theory prediction", "LF")
        # only add to the legend if values are in terms of a cross section
        if xsec_unit:
            legend_entries[1 if has_obs else 0] = legend_entry

    # horizontal guidance lines
    if h_lines:
        for i in h_lines:
            line_obs = ROOT.TLine(x_min, float(i), x_max, float(i))
            r.setup_line(line_obs, props={"NDC": False}, color=12)
            draw_objs.append(line_obs)

    # determine the number of digits for reported limits
    get_digits = lambda v: digits
    if digits is None:
        get_digits = lambda v: determine_limit_digits(v, is_xsec=bool(xsec_unit))

    # templates and helpers for y axis labels
    y_label_tmpl = "#splitline{%s}{#scale[0.75]{Expected: %s}}"
    y_label_tmpl_obs = "#splitline{%s}{#scale[0.75]{#splitline{Expected: %s}{Observed: %s}}}"

    def make_y_label(name, exp, obs=None):
        if xsec_unit:
            fmt = lambda v: "{{:.{}f}} {{}}".format(get_digits(v)).format(v, xsec_unit)
        else:
            fmt = lambda v: "{{:.{}f}}".format(get_digits(v)).format(v)
        if obs is None or np.isnan(obs[0]):
            return y_label_tmpl % (label, fmt(exp))
        return y_label_tmpl_obs % (label, fmt(exp), fmt(obs[0]))

    # create y axis labels and ticks
    h_dummy.GetYaxis().SetBinLabel(1, "")
    for i, d in enumerate(data):
        # name labels
        label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        label = make_y_label(label, d["expected"][0], d.get("observed"))
        label_x = r.get_x(10, canvas)
        label_y = r.get_y(bottom_margin + int((n - i - 1.3) * entry_height), pad)
        label = ROOT.TLatex(label_x, label_y, label)
        r.setup_latex(label, props={"NDC": True, "TextAlign": 12, "TextSize": label_size})
        draw_objs.append(label)

        # left and right ticks
        tick_length = 0.03
        if x_log:
            tick_length_l = x_min * ((x_max / float(x_min))**tick_length - 1)
            tick_length_r = x_max * (1 - (x_min / float(x_max))**tick_length)
        else:
            tick_length_l = tick_length_r = tick_length * (x_max - x_min)
        tl = ROOT.TLine(x_min, i + 1, x_min + tick_length_l, i + 1)
        tr = ROOT.TLine(x_max - tick_length_r, i + 1, x_max, i + 1)
        r.setup_line(tl, props={"NDC": False, "LineWidth": 1})
        r.setup_line(tr, props={"NDC": False, "LineWidth": 1})
        draw_objs.extend([tl, tr])

        # extra labels
        if d.get("label"):
            rlabel = to_root_latex(hh_references.get(d["name"], d["label"]))
            rlabel_x = r.get_x(10, pad, anchor="right")
            rlabel = ROOT.TLatex(rlabel_x, label_y, rlabel)
            r.setup_latex(rlabel, props={"NDC": True, "TextAlign": 32, "TextSize": 14})
            draw_objs.append(rlabel)

    # fill hep data
    if hep_data:
        poi_div = "" if xsec_unit else r" / $\sigma_{Theory}$"
        poi_qual = create_hh_xsbr_label(poi, hh_process) + poi_div
        qualifiers = lambda: [hdt.create_qualifier("POI", poi_qual)]

        # data entries
        hdt.create_independent_variable(
            "Measurement",
            parent=hep_data,
            values=[hdt.create_value(br_hh_names.get(d["name"], d["name"])) for d in data],
        )

        # theory prediction
        if has_thy and xsec_unit:
            g, lx = (g_thy_area, "th") if has_thy_err else (g_thy_line, None)
            hdt.create_dependent_variable_from_graph(
                g,
                coord="x",
                error_label=lx,
                parent=hep_data,
                label="Theory prediction",
                unit=xsec_unit,
                qualifiers=qualifiers(),
                rounding_method="publication",
            )

        def exp_limit_value(d, sigma, label):
            vals = list(map(float, d["expected"]))
            n, p, m = vals[0], vals[1 + 2 * (sigma - 1)], vals[2 + 2 * (sigma - 1)]
            return Number(n, {label: (p - n, n - m)}, default_format="publication")

        # expected limits with 68% error
        hdt.create_dependent_variable(
            "Expected limit, 68%",
            parent=hep_data,
            unit=xsec_unit,
            values=[exp_limit_value(d, 1, "68%") for d in data],
            qualifiers=qualifiers(),
        )

        # expected limits with 95% error
        hdt.create_dependent_variable(
            "Expected limit, 95%",
            parent=hep_data,
            unit=xsec_unit,
            values=[exp_limit_value(d, 2, "95%") for d in data],
            qualifiers=qualifiers(),
        )

        # observed limits
        if has_obs:
            hdt.create_dependent_variable_from_graph(
                g_obs,
                coord="x",
                label="Observed limit",
                parent=hep_data,
                unit=xsec_unit,
                qualifiers=qualifiers(),
                rounding_method=3,
            )

    # legend
    legend = r.routines.create_legend(pad=pad, width=430, n=3, props={"NColumns": 2})
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

    # save plots
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)

    # save hep data
    if hep_data_path:
        hdt.save_hep_data(hep_data, hep_data_path)


@use_style("dhi_default")
def plot_limit_scan_2d(
    paths,
    poi,
    scan_parameter1,
    scan_parameter2,
    expected_limits,
    observed_limits=None,
    draw_sm_point=True,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    z_log=False,
    model_parameters=None,
    campaign=None,
    h_lines=None,
    v_lines=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot for the upper limit scan of a *poi* in two dimensions over *scan_parameter1* and
    *scan_parameter2*, and saves it at *paths*. *expected_limits* should be a mapping to lists of
    values or a record array with keys "<scan_parameter1>", "<scan_parameter2>" and "limit", and
    optionally "limit_p1" (plus 1 sigma) and "limit_m1" (minus 1 sigma). When the variations are
    missing, the plot is created without them. When *observed_limits* is set, it should have a
    similar format with keys "<scan_parameter1>", "<scan_parameter2>" and "limit". When
    *draw_sm_point* is set, a marker is shown at the coordinates (1, 1).

    *x_min*, *x_max*, *y_min*, *y_max*, *z_min* and *z_max* define the axes ranges and default to
    the ranges of the given values. When *z_log* is *True*, the z-axis is plotted with a logarithmic
    scale. *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign*
    should refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.
    *h_lines* and *v_lines* can be sequences of float values denoting positions of horizontal and
    vertical lines, respectively, to be drawn. *cms_postfix* is shown as the postfix behind the CMS
    label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/limits.html#limit-on-poi-vs-two-scan-parameters  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    def check_values(values):
        if isinstance(values, list):
            return list(map(check_values, values))
        if isinstance(values, np.ndarray):
            values = {key: np.array(values[key]) for key in values.dtype.names}
        assert scan_parameter1 in values
        assert scan_parameter2 in values
        return values

    def join_limits(values):
        return unique_recarray(dict_to_recarray(values), cols=[scan_parameter1, scan_parameter2])

    # prepare inputs
    expected_limits = check_values(make_list(expected_limits))
    joined_expected_limits = join_limits(expected_limits)
    has_unc = (
        "limit_p1" in joined_expected_limits.dtype.names and
        "limit_m1" in joined_expected_limits.dtype.names
    )
    shown_limits = expected_limits
    has_obs = False
    if observed_limits is not None:
        observed_limits = check_values(make_list(observed_limits))
        joined_observed_limits = join_limits(observed_limits)
        shown_limits = observed_limits
        has_obs = True

    # determine contours independent of plotting
    exp_contours = get_contours(
        joined_expected_limits[scan_parameter1],
        joined_expected_limits[scan_parameter2],
        joined_expected_limits["limit"],
        levels=[1.],
        frame_kwargs=[{"mode": "edge", "width": 1.}],
    )[0]
    if has_unc:
        exp_p1_contours = get_contours(
            joined_expected_limits[scan_parameter1],
            joined_expected_limits[scan_parameter2],
            joined_expected_limits["limit_p1"],
            levels=[1.],
            frame_kwargs=[{"mode": "edge", "width": 1.}],
        )[0]
        exp_m1_contours = get_contours(
            joined_expected_limits[scan_parameter1],
            joined_expected_limits[scan_parameter2],
            joined_expected_limits["limit_m1"],
            levels=[1.],
            frame_kwargs=[{"mode": "edge", "width": 1.}],
        )[0]
    if has_obs:
        obs_contours = get_contours(
            joined_observed_limits[scan_parameter1],
            joined_observed_limits[scan_parameter2],
            joined_observed_limits["limit"],
            levels=[1.],
            frame_kwargs=[{"mode": "edge", "width": 1.}],
        )[0]

    # start plotting
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(pad_props={"RightMargin": 0.17, "Logz": z_log})
    pad.cd()

    # custom palette, requires that the z range is symmetrical around 1
    rvals = np.array([ROOT.gROOT.GetColor(colors.red).GetRed(), 1., 0.])
    gvals = np.array([0., 1., 0.])
    bvals = np.array([0., 1., ROOT.gROOT.GetColor(colors.blue).GetBlue()])
    lvals = np.array([0.0, 0.5, 1.0])
    ROOT.TColor.CreateGradientColorTable(len(lvals), lvals, rvals, gvals, bvals, 100)

    # create a histogram for each scan patch
    hists = []
    for i, data in enumerate(shown_limits):
        _, _, _x_bins, _y_bins, _x_min, _x_max, _y_min, _y_max = infer_binning_from_grid(
            data[scan_parameter1],
            data[scan_parameter2],
        )
        _z_min = np.nanmin(data["limit"])
        _z_max = np.nanmax(data["limit"])

        # infer axis limits from the first set of values
        if i == 0:
            x_min = _x_min if x_min is None else x_min
            x_max = _x_max if x_max is None else x_max
            y_min = _y_min if y_min is None else y_min
            y_max = _y_max if y_max is None else y_max
            z_min = _z_min if z_min is None else z_min
            z_max = _z_max if z_max is None else z_max

        # fill and store the histogram
        h = ROOT.TH2F("h" + str(i), "", _x_bins, _x_min, _x_max, _y_bins, _y_min, _y_max)
        fill_hist_from_points(h, data[scan_parameter1], data[scan_parameter2], data["limit"],
            z_min=z_min, z_max=z_max)
        hists.append(h)

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[scan_parameter1].label)
    y_title = to_root_latex(poi_data[scan_parameter2].label)
    z_title = "95% CL limit on {} / #sigma_{{Theory}}".format(
        to_root_latex(create_hh_xsbr_label(poi)),
    )
    h_dummy = ROOT.TH2F(
        "h_dummy",
        ";{};{};{}".format(x_title, y_title, z_title),
        1,
        x_min,
        x_max,
        1,
        y_min,
        y_max,
    )
    r.setup_hist(h_dummy, pad=pad, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
    draw_objs = [(h_dummy, "")]
    legend_entries = 4 * [(h_dummy, " ", "")]

    # setup actual histograms
    for i, h in enumerate(hists):
        r.setup_hist(h, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
        if i == 0:
            r.setup_z_axis(
                h.GetZaxis(),
                pad=pad,
                props={"Title": z_title, "TitleSize": 24, "TitleOffset": 1.5},
            )
        draw_objs.append((h, "SAME,COLZ"))
        # draw_objs.append((h, "SAME,TEXT"))

    # helper for parsing line options
    def parse_line_value(v, props):
        if isinstance(v, six.string_types):
            parts = v.split("@")
            v = float(parts[0])
            for part in parts[1:]:
                if "=" not in part:
                    continue
                key, prop = part.split("=", 1)
                props[key] = try_int(float(prop))  # cast to float or integer

        return v, props

    # horizontal lines
    if h_lines:
        for y in h_lines:
            y, props = parse_line_value(y, {"NDC": False, "LineColor": colors.dark_grey})
            line = ROOT.TLine(x_min, float(y), x_max, float(y))
            r.setup_line(line, props=props)
            draw_objs.append(line)

    # vertical lines
    if v_lines:
        for x in v_lines:
            x, props = parse_line_value(x, {"NDC": False, "LineColor": colors.dark_grey})
            line = ROOT.TLine(y_min, float(x), y_max, float(x))
            r.setup_line(line, props=props)
            draw_objs.append(line)

    # exclusion contours
    for i, g in enumerate(exp_contours):
        r.setup_graph(
            g,
            props={"LineWidth": 2, "LineColor": colors.black, "LineStyle": 2},
        )
        draw_objs.append((g, "SAME,C"))
        if i == 0:
            legend_entries[2] = (g, "Excluded (exp.)", "L")

    if has_unc:
        for i, g in enumerate(exp_p1_contours + exp_m1_contours):
            r.setup_graph(
                g,
                props={"LineWidth": 2, "LineColor": colors.black, "LineStyle": 3},
            )
            draw_objs.append((g, "SAME,C"))
            if i == 0:
                legend_entries[3] = (g, r"68% expected", "L")

    if has_obs:
        for i, g in enumerate(obs_contours):
            r.setup_graph(
                g,
                props={"LineWidth": 2, "LineColor": colors.black},
            )
            draw_objs.append((g, "SAME,C"))
            if i == 0:
                legend_entries[0] = (g, "Excluded (obs.)", "L")

    # SM point
    if draw_sm_point:
        g_sm = create_tgraph(1, 1, 1)
        r.setup_graph(
            g_sm,
            props={"MarkerStyle": 33, "MarkerSize": 2.5},
            color=colors.black,
        )
        draw_objs.append((g_sm, "P"))
        legend_entries[1] = (g_sm, "Standard model", "P")

    # legend
    legend = r.routines.create_legend(pad=pad, x2=-20, width=380, n=2, props={"NColumns": 2})
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
def plot_benchmark_limits(
    paths,
    data,
    poi,
    y_min=None,
    y_max=None,
    y_log=False,
    xsec_unit="fb",
    hh_process=None,
    campaign=None,
    bar_width=0.6,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing the limits of BSM benchmarks for a *poi* and saves it at *paths*. *data*
    should be a list of dictionaries with fields

    - "expected", a sequence of five values, i.e., central limit, and +1 sigma, -1 sigma, +2 sigma,
      and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a single value,
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead. It can also be a 2-tuple (name, group name) and if so,
      lines are draw between groups and a group name label is shown.

    Example:

    .. code-block:: python

        plot_benchmark_limits(
            paths=["plot.pdf", "plot.png"],
            data=[
                {
                    "expected": (40., 50., 28., 58., 18.),
                    "observed": 45.,
                    "name": "1",
                }, {
                    ...
                },
            ],
        )

    *y_min* and *y_max* define the y axis range and default to the range of the given values. When
    *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *xsec_unit* denotes the unit
    of the passed values and defaults to fb. *hh_process* can be the name of a HH subprocess
    configured in *dhi.config.br_hh_names* and is inserted to the process name in the title of the
    y-axis and indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    The *bar_width* should be a value between 0 and 1 and controls the fraction of the limit bar
    width relative to the bin width. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/eft.html#benchmark-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check inputs and get extrema
    n = len(data)
    has_obs = False
    y_min_value = 1e5
    y_max_value = -1e5
    for d in data:
        assert "name" in d
        assert "expected" in d
        y_min_value = min(y_min_value, min(d["expected"]))
        y_max_value = max(y_max_value, max(d["expected"]))
        if "observed" in d:
            assert isinstance(d["observed"], (float, int))
            has_obs = True
            y_min_value = min(y_min_value, d["observed"])
            y_max_value = max(y_max_value, d["observed"])
            d["observed"] = [d["observed"]]

    # when certain names are detected (keys of bm_labels), group and reoder the cards
    groups = OrderedDict()
    if any(d["name"] in bm_labels for d in data):
        # sort
        bm_keys = list(bm_labels.keys())
        data.sort(key=lambda d: bm_keys.index(d["name"]) if d["name"] in bm_keys else 1e9)

        # fill the group dict
        for d in data:
            bm_name, group_name = d["name"], ""
            if bm_name in bm_labels:
                bm_name, group_name = bm_labels[bm_name]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(bm_name)
            d["name"] = bm_name

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(
        width=1200,
        height=600,
        pad_props={"Logy": y_log, "LeftMargin": 0.09, "BottomMargin": 0.15 if groups else 0.1},
    )
    pad.cd()
    draw_objs = []

    # dummy histogram to control axes
    x_title = "Benchmark scenario"
    y_title = "95% CL limit on {} ({})".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex(xsec_unit),
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), n, -0.5, n - 0.5)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"Ndivisions": n, "TitleOffset": 1.7 if groups else 1.1},
    )
    draw_objs.append((h_dummy, "HIST"))

    # reserve legend entries
    legend_entries = 4 * [(h_dummy, " ", "L")]

    # benchmark labels
    for i, d in enumerate(data):
        h_dummy.GetXaxis().SetBinLabel(i + 1, to_root_latex(make_tuple(d["name"])[0]))

    # helper to read values into graphs
    def create_graph(key="expected", sigma=None):
        args = x, y, x_err_d, x_err_u, y_err_d, y_err_u = [n * [0.] for _ in range(6)]
        for i, d in enumerate(data):
            if key not in d:
                y[i] = -1.e5
                continue
            x[i] = i - 0.5 * bar_width
            x_err_u[i] = bar_width
            y[i] = d[key][0]
            if sigma:
                y_err_d[i] = y[i] - d[key][sigma * 2]
                y_err_u[i] = d[key][sigma * 2 - 1] - y[i]
        return create_tgraph(n, *args)

    # 2 sigma band
    g_2sigma = create_graph(sigma=2)
    r.setup_graph(
        g_2sigma,
        color=colors.brazil_yellow,
        color_flags="lf",
        props={"LineWidth": 2, "LineStyle": 1},
    )
    draw_objs.append((g_2sigma, "SAME,2"))
    legend_entries[3] = (g_2sigma, r"95% expected", "LF")

    # 1 sigma band
    g_1sigma = create_graph(sigma=1)
    r.setup_graph(
        g_1sigma,
        color=colors.brazil_green,
        color_flags="lf",
        props={"LineWidth": 2, "LineStyle": 1},
    )
    draw_objs.append((g_1sigma, "SAME,2"))
    legend_entries[2] = (g_1sigma, r"68% expected", "LF")

    # prepare graphs
    g_exp = create_graph(sigma=0)
    r.setup_graph(g_exp, props={"LineWidth": 2, "LineStyle": 2})
    draw_objs.append((g_exp, "SAME,EZ"))
    legend_entries[0] = (g_exp, "Median expected", "L")

    # observed values
    if has_obs:
        g_obs = create_graph(key="observed")
        r.setup_graph(g_obs, props={"LineWidth": 2, "LineStyle": 1})
        draw_objs.append((g_obs, "SAME,EZ"))
        legend_entries[1] = (g_obs, "Observed", "L")

    # texts over and lines between groups
    group_names = list(groups)
    for i, group_name in enumerate(group_names):
        # label
        x = sum(len(groups[group_names[j]]) for j in range(i + 1))
        y_offset = 0.1
        label = ROOT.TLatex(
            x - 0.5 * (1 + len(groups[group_name])),
            (y_min * (y_min / y_max)**y_offset) if y_log else (y_min - y_offset * (y_max - y_min)),
            to_root_latex(group_name),
        )
        r.setup_latex(label, props={"NDC": False, "TextAlign": 21, "TextSize": 20})
        draw_objs.append(label)

        # separating line
        if i < len(groups) - 1:
            line = ROOT.TLine(x - 0.5, y_min, x - 0.5, y_max_value)
            r.setup_line(
                line,
                props={"LineColor": colors.black, "LineStyle": 1, "NDC": False},
            )
            draw_objs.append(line)

    # legend
    legend = r.routines.create_legend(
        pad=pad,
        width=440,
        y2=-20,
        n=2,
        props={"NColumns": 2},
    )
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
def plot_multi_benchmark_limits(
    paths,
    data,
    names,
    poi,
    y_min=None,
    y_max=None,
    y_log=False,
    xsec_unit="fb",
    hh_process=None,
    campaign=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a plot showing multiple limits for various BSM benchmarks for a *poi* and saves it at
    *paths*. *data* should be a list of dictionaries with fields

    - "expected", a list of sequences of five values, i.e., central limit, and +1 sigma, -1 sigma,
      +2 sigma, and -2 sigma variations (absolute values, not errors!),
    - "observed" (optional), a sequence of single float values and when set, it should have the same
      length as "expected",
    - "name", shown as the y-axis label and when it is a key of dhi.config.br_hh_names,
      its value is used as a label instead. It can also be a 2-tuple (name, group name) and if so,
      lines are draw between groups and a group name label is shown.

    Example:

    .. code-block:: python

        plot_benchmark_limits(
            paths=["plot.pdf", "plot.png"],
            data=[
                {
                    "expected": [
                        (40.0, 50.0, 28.0, 58.0, 18.0),
                        (38.0, 47.0, 27.0, 61.0, 19.0),
                    ],
                    "observed": [
                        45.0,
                        42.0,
                    ],
                    "name": "1",
                }, {
                    ...
                },
            ],
            names=["Analysis A", "Aenalysis B"],
        )

    *names* should be a list of names that are shown in the legend and refer to each element in the
    *exected* and *observed* lists in *data*.

    *y_min* and *y_max* define the y axis range and default to the range of the given values. When
    *y_log* is *True*, the y-axis is plotted with a logarithmic scale. *xsec_unit* denotes the unit
    of the passed values and defaults to fb. *hh_process* can be the name of a HH subprocess
    configured in *dhi.config.br_hh_names* and is inserted to the process name in the title of the
    y-axis and indicates that the plotted cross section data was (e.g.) scaled by a branching ratio.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/eft.html#multiple-benchmark-limits
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # check inputs and get extrema
    n = len(data)
    n_graphs = len(names)
    has_obs = False
    y_min_value = 1e5
    y_max_value = -1e5
    for d in data:
        assert "name" in d
        assert "expected" in d
        assert isinstance(d["expected"], (list, tuple))
        assert all(isinstance(e, (list, tuple)) for e in d["expected"])
        assert len(d["expected"]) == n_graphs
        y_min_value = min(y_min_value, min(e[0] for e in d["expected"]))
        y_max_value = max(y_max_value, max(e[0] for e in d["expected"]))
        if "observed" in d:
            assert isinstance(d["observed"], (list, tuple))
            assert all(isinstance(o, (int, float)) for o in d["observed"])
            assert len(d["observed"]) == n_graphs
            has_obs = True
            y_min_value = min(y_min_value, min(d["observed"]))
            y_max_value = max(y_max_value, max(d["observed"]))

    # when certain names are detected (keys of bm_labels), group and reoder the cards
    groups = OrderedDict()
    if any(d["name"] in bm_labels for d in data):
        # sort
        bm_keys = list(bm_labels.keys())
        data.sort(key=lambda d: bm_keys.index(d["name"]) if d["name"] in bm_keys else 1e9)

        # fill the group dict
        for d in data:
            bm_name, group_name = d["name"], ""
            if bm_name in bm_labels:
                bm_name, group_name = bm_labels[bm_name]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(bm_name)
            d["name"] = bm_name

    # set limits
    y_min, y_max, _ = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)

    # setup the default style and create canvas and pad
    r.setup_style()
    canvas, (pad,) = r.routines.create_canvas(
        width=1200,
        height=600,
        pad_props={"Logy": y_log, "LeftMargin": 0.09, "BottomMargin": 0.15 if groups else 0.1},
    )
    pad.cd()
    draw_objs = []
    legend_entries = []

    # dummy histogram to control axes
    x_title = "Benchmark scenario"
    y_title = "95% CL limit on {} ({})".format(
        to_root_latex(create_hh_xsbr_label(poi, hh_process)),
        to_root_latex(xsec_unit),
    )
    h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), n, -0.5, n - 0.5)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(
        h_dummy.GetXaxis(),
        pad=pad,
        props={"Ndivisions": n, "TitleOffset": 1.7 if groups else 1.1},
    )
    draw_objs.append((h_dummy, "HIST"))

    # benchmark labels
    for i, d in enumerate(data):
        h_dummy.GetXaxis().SetBinLabel(i + 1, to_root_latex(make_tuple(d["name"])[0]))

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(name in br_hh_colors.root for name in names):
        _color_sequence = [br_hh_colors.root[name] for name in names]

    # helper to read values into graphs
    def create_graph(index, key):
        args = x, y, x_err_d, x_err_u, y_err_d, y_err_u = [n * [0.] for _ in range(6)]
        for i, d in enumerate(data):
            if key not in d:
                y[i] = -1.e5
                continue
            x[i] = i - 0.5
            x_err_u[i] = 1.0
            y[i] = d[key][index] if key == "observed" else d[key][index][0]
        return create_tgraph(n, *args)

    # central values
    for i, (name, col) in enumerate(zip(names, _color_sequence[:n_graphs])):
        # prepare graphs
        g_exp = create_graph(i, "expected")
        r.setup_graph(
            g_exp,
            color=colors[col],
            props={"LineWidth": 2, "LineStyle": 2},
        )
        draw_objs.append((g_exp, "SAME,EZ"))
        legend_entries.append((g_exp, to_root_latex(br_hh_names.get(name, name)), "L"))

        # observed values
        if has_obs:
            g_obs = create_graph(i, "observed")
            r.setup_graph(
                g_obs,
                color=colors[col],
                props={"LineWidth": 2, "LineStyle": 1},
            )
            draw_objs.append((g_obs, "SAME,EZ"))

    # add additional legend entries to distinguish expected and observed lines
    for _ in range(3 - len(legend_entries) % 3):
        legend_entries.append((h_dummy, " ", "L"))
    g_exp_dummy = g_exp.Clone()
    r.apply_properties(g_exp_dummy, {"LineColor": colors.black})
    legend_entries.append((g_exp_dummy, "Median expected", "L"))
    if has_obs:
        g_obs_dummy = g_obs.Clone()
        r.apply_properties(g_obs_dummy, {"LineColor": colors.black})
        legend_entries.append((g_obs_dummy, "Observed", "L"))
    else:
        legend_entries.append((h_dummy, " ", "L"))

    # texts over and lines between groups
    group_names = list(groups)
    for i, group_name in enumerate(group_names):
        # label
        x = sum(len(groups[group_names[j]]) for j in range(i + 1))
        y_offset = 0.1
        label = ROOT.TLatex(
            x - 0.5 * (1 + len(groups[group_name])),
            (y_min * (y_min / y_max)**y_offset) if y_log else (y_min - y_offset * (y_max - y_min)),
            to_root_latex(group_name),
        )
        r.setup_latex(label, props={"NDC": False, "TextAlign": 21, "TextSize": 20})
        draw_objs.append(label)

        # separating line
        if i < len(groups) - 1:
            line = ROOT.TLine(x - 0.5, y_min, x - 0.5, y_max_value)
            r.setup_line(
                line,
                props={"LineColor": colors.black, "LineStyle": 1, "NDC": False},
            )
            draw_objs.append(line)

    # legend
    legend_cols = int(math.ceil(len(legend_entries) / 3.0))
    legend = r.routines.create_legend(
        pad=pad,
        width=legend_cols * 180,
        y2=-20,
        n=3,
        props={"NColumns": legend_cols, "TextSize": 18},
    )
    r.fill_legend(legend, legend_entries)
    draw_objs.append(legend)
    legend_box = r.routines.create_legend_box(
        legend,
        pad,
        "trl",
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


def evaluate_limit_scan_1d(
    scan_values,
    limit_values,
    xsec_scan_values=None,
    xsec_values=None,
    interpolation="linear",
):
    """
    Takes the results of an upper limit scan given by the *scan_values* and the corresponding
    *limit* values, performs an interpolation and returns certain results of the scan in a dict.

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
    interp = scipy.interpolate.interp1d(
        scan_values,
        limit_values,
        kind=interpolation,
        fill_value="extrapolate",
    )
    n_nans = (~mask).sum()
    if n_nans:
        print("WARNING: found {} NaN(s) in limit values for 1d evaluation".format(n_nans))

    # same for cross section values when given
    if xsec_scan_values is not None and xsec_values is not None and not (xsec_values == 1).all():
        mask = ~np.isnan(xsec_values)
        xsec_scan_values = xsec_scan_values[mask]
        xsec_values = xsec_values[mask]
        xsec_interp = scipy.interpolate.interp1d(
            xsec_scan_values,
            xsec_values,
            kind=interpolation,
            fill_value="extrapolate",
        )
        n_nans = (~mask).sum()
        if n_nans:
            print("WARNING: found {} NaN(s) in xsec values for 1d evaluation".format(n_nans))
    else:
        xsec_interp = lambda x: 1.

    # interpolated difference between of limit and prediction (== cross section or 1)
    diff_interp = lambda x: interp(x) - xsec_interp(x)
    # diff_interp = lambda x: math.log(interp(x)) - math.log(xsec_interp(x))

    # interpolation bounds
    bounds = (scan_values.min() + 1e-4, scan_values.max() - 1e-4)

    # helper to find intersections with one given a starting point
    def get_intersection(start):
        objective = lambda x: abs(diff_interp(x))
        res = minimize_1d(objective, bounds, start=start)
        # catch bad minimizations
        if res.status != 0 or not (bounds[0] <= res.x[0] <= bounds[1]):
            return None
        # catch local minima by exploiting that objectives of actual minima must be close to 0
        x = res.x[0]
        if objective(x) > 1e-2:
            return None
        return x

    # get exclusion range edges from intersections
    rnd = lambda v: round(float(v), 6)
    edges = {rnd(scan_values.min()), rnd(scan_values.max())}
    for start in np.linspace(scan_values.min(), scan_values.max(), 20):
        x = get_intersection(start)
        if x is not None:
            edges.add(rnd(x))

    # drop edges that are too close to one another (within 4 digits)
    _edges = []
    for x in sorted(edges):
        if not _edges or x - _edges[-1] >= 1e-4:
            _edges.append(x)
    edges = _edges

    # create ranges consisting of two adjacent edges
    ranges = [(edges[i - 1], edges[i]) for i in range(1, len(edges))]

    # select those ranges whose central value is below 1
    excluded_ranges = [r for r in ranges if diff_interp(0.5 * (r[1] + r[0])) < 0]

    return DotDict(
        interp=interp,
        excluded_ranges=excluded_ranges,
    )


def evaluate_excluded_ranges(
    param_name,
    scan_name,
    scan_values,
    limit_values,
    xsec_scan_values=None,
    xsec_values=None,
    interpolation="linear",
):
    # more than 5 points are required
    if len(scan_values) <= 5:
        print("insufficient number of scan points for extracting excluded ranges")
        return None

    try:
        ranges = evaluate_limit_scan_1d(
            scan_values,
            limit_values,
            xsec_scan_values=xsec_scan_values,
            xsec_values=xsec_values,
            interpolation=interpolation,
        ).excluded_ranges
    except Exception:
        print("1D limit scan evaluation failed")
        traceback.print_exc()
        return None

    # print them
    _print_excluded_ranges(param_name, scan_name, scan_values, ranges, interpolation)

    return ranges


def _print_excluded_ranges(param_name, scan_name, scan_values, ranges, interpolation):
    # helper to check if the granularity of scan values is too small at a certain point
    def granularity_check(value, digits=2):
        # get the closest values
        for i in range(len(scan_values) - 1):
            l, r = scan_values[i], scan_values[i + 1]
            if l <= value <= r:
                break
        else:
            raise Exception("could not find closest scan value pair around value {}".format(value))

        # case 1: pass when the value is very close to any of the two closest points
        if min([r - value, value - l]) <= 10**(-1 * digits):
            return True

        # case2: pass when the granularity is sufficiently high
        if r - l <= 10**(1 - digits):
            return True

        return False

    # helper to format a range
    def format_range(start, stop):
        if abs(start - scan_values.min()) < 1e-4:
            r = "{} < {:.5f}".format(param_name, stop)
            # check = granularity_check(stop)
        elif abs(stop - scan_values.max()) < 1e-4:
            r = "{} > {:.5f}".format(param_name, start)
            # check = granularity_check(start)
        else:
            r = "{:.5f} < {} < {:.5f}".format(start, param_name, stop)
            # check = granularity_check(start) and granularity_check(stop)
        # check currently disabled
        # if not check:
        #     r += " (granularity potentially insufficient for accurate interpolation)"
        return r

    # start printing
    print("")
    title = "Excluded ranges of parameter '{}' in scan '{}' (from {} interpolation):".format(
        param_name, scan_name, interpolation,
    )
    print(title)
    print(len(title) * "=")
    if not ranges:
        print("no excluded ranges found")
    else:
        print("(granularity potentially insufficient for accurate interpolation)")
        for start, stop in ranges:
            print("  - " + format_range(start, stop))
    print("")


def excluded_to_allowed_ranges(excluded_ranges, min_value, max_value):
    """
    Converts a list *ranges* of 2-tuples with edges referring to excluded ranges to allowed ones,
    taking into account the endpoints given by *min_value* and *max_value*. An open range is denoted
    by a *None*. Thus, (*None*, *None*) would mean that the entire range is allowed.
    """
    if excluded_ranges is None:
        return None

    # shorthands and checks
    e_ranges = excluded_ranges
    a_ranges = []
    min_value = float(min_value)
    max_value = float(max_value)

    # excluded_ranges might mark open ends with None's as well, so before linearizing values,
    # replace them with the global endpoints
    if e_ranges:
        if e_ranges[0][0] is None:
            e_ranges[0] = (min_value, e_ranges[0][0])
        if e_ranges[-1][1] is None:
            e_ranges[-1] = (e_ranges[-1][0], max_value)

    # linearize excluded ranges (assuming edges are always reasonably far apart)
    e_edges = map(float, sum(map(list, excluded_ranges), []))

    # loop through adjacent pairs and create allowed ranges in an alternating fashion
    for i, (start, stop) in enumerate(zip(e_edges[:-1], e_edges[1:])):
        # first range
        if i == 0 and start > min_value:
            a_ranges.append((min_value, start))
        # last range
        if i == len(e_edges) - 2 and stop < max_value:
            a_ranges.append((stop, max_value))
        # intermediate ranges
        if i % 2 != 0:
            a_ranges.append((start, stop))

    return a_ranges
