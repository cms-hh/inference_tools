# coding: utf-8

"""
Likelihood plots using ROOT.
"""

import math
import json
from collections import OrderedDict
from itertools import chain, product

import numpy as np
import scipy.interpolate
import scipy.optimize
from scinum import Number

from dhi.config import (
    poi_data, br_hh_names, br_hh_colors, campaign_labels, chi2_levels, colors, color_sequence,
    marker_sequence, get_chi2_level, get_chi2_level_from_cl,
)
from dhi.util import (
    import_ROOT, to_root_latex, create_tgraph, DotDict, minimize_1d, multi_match, convert_rooargset,
    make_list, unique_recarray, dict_to_recarray, warn, prepare_output,
)
from dhi.plots.util import (
    use_style, create_model_parameters, fill_hist_from_points, get_contours, get_y_range,
    infer_binning_from_grid, get_contour_box, make_parameter_label_map, get_text_extent,
    locate_contour_labels, Style,
)
import dhi.hepdata_tools as hdt


colors = colors.root


@use_style("dhi_default")
def plot_likelihood_scans_1d(
    paths,
    poi,
    data,
    theory_value=None,
    ranges_path=None,
    hep_data_path=None,
    show_best_fit=False,
    show_best_fit_error=True,
    show_best_fit_line=None,
    show_best_fit_indicators=None,
    show_significances=(1, 2, 3, 5),
    shift_negative_values=False,
    interpolate_above=None,
    v_lines=None,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_points=True,
    cms_postfix=None,
    style=None,
):
    """
    Plots multiple curves of 1D likelihood scans of a POI *poi1* and *poi2*, and saves it at *paths*.
    All information should be passed as a list *data*. Entries must be dictionaries with the
    following content:

        - "values": A mapping to lists of values or a record array with keys "<poi1_name>" and
          "dnll2".
        - "poi_min": A float describing the best fit value of the POI. When not set, the minimum is
          estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    *theory_value* can be a 3-tuple denoting the nominal theory prediction of the POI and its up and
    down uncertainties which is drawn as a vertical bar. When *ranges_path* is set, one and two
    sigma intervals of the scan parameter are saved to the given file. When *hep_data_path* is set,
    a yml data file compatible with the HEPData format
    (https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html) is stored at that path.

    When *show_best_fit* (*show_best_fit_error*) is *True*, the best fit error value (and its
    uncertainty) is shown in the corresponding legend entry. When *show_best_fit_line* is *True*, a
    vertical line is shown at the position of the best fit value. When *show_best_fit_indicators* is
    *True* and only a single scan is shown, vertical indicators of the one and two sigma intervals
    of the best fit value, when requested in *show_significances*, are shown. The two latter
    arguments default to the value of *show_best_fit*.

    To overlay lines and labels denoting integer significances corresponding to 1D likelihood scans,
    *show_significances* can be set to *True* to show significances up to 9 sigma, or a list of
    sigmas (integer, >= 1) or confidence levels (float, < 1). In case there are negative dnll2
    values, *shift_negative_values* can be set to *True* to shift them vertically so that the
    minimum is located at 0 again. When *interpolate_above* is defined, values that exceed this
    threshold are removed and interpolated using adjacent values instead.

    *v_lines* can be a list of x-values at which vertical, dashed lines are drawn for visual
    guidance. *x_min* and *x_max* define the x-axis range of POI, and *y_min* and *y_max* control
    the range of the y-axis. When *y_log* is *True*, the y-axis is plotted with a logarithmic scale.
    When *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign*
    should refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When
    *show_points* is *True*, the central scan points are drawn on top of the interpolated curve.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#1d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # input checks and transformations
    if theory_value is not None:
        theory_value = make_list(theory_value)
    if not show_best_fit:
        show_best_fit_error = False
    if show_best_fit_line is None:
        show_best_fit_line = show_best_fit
    if show_best_fit_indicators is None:
        show_best_fit_indicators = show_best_fit

    # validate data entries
    for i, d in enumerate(data):
        # convert likelihood values to arrays
        assert "values" in d
        values = d["values"]
        if isinstance(values, np.ndarray):
            values = {k: values[k] for k in values.dtype.names}
        assert poi in values
        assert "dnll2" in values
        # check poi minimum
        d.setdefault("poi_min", None)
        if not isinstance(d["poi_min"], (list, tuple)):
            d["poi_min"] = [d["poi_min"]]
        # default name
        d.setdefault("name", str(i + 1))
        # origin (for printouts)
        d["origin"] = None if not d["name"] else "entry '{}'".format(d["name"])
        # drop all fields except for required ones
        values = {
            k: np.array(v, dtype=np.float32)
            for k, v in values.items()
            if k in [poi, "dnll2"]
        }
        # preprocess values (nan detection, negative shift)
        values["dnll2"], values[poi] = _preprocess_values(
            values["dnll2"],
            (poi, values[poi]),
            remove_nans=True,
            remove_above=interpolate_above,
            shift_negative_values=shift_negative_values,
            origin=d["origin"],
            min_is_external=d["poi_min"] is not None,
        )
        d["values"] = values

    # prepare hep data
    hep_data = None
    if hep_data_path:
        hep_data = hdt.create_hist_data()

    # perform scans
    scans = [
        evaluate_likelihood_scan_1d(
            d["values"][poi],
            d["values"]["dnll2"],
            poi_min=d["poi_min"][0],
            origin=d["origin"],
        )
        for d in data
    ]

    # set x range
    if x_min is None:
        x_min = min([min(d["values"][poi]) for d in data])
    if x_max is None:
        x_max = max([max(d["values"][poi]) for d in data])

    # set y range
    y_max_value = max([
        d["values"]["dnll2"][(d["values"][poi] >= x_min) & (d["values"][poi] <= x_max)].max()
        for d in data
    ])
    y_min_value = min([
        d["values"]["dnll2"][(d["values"][poi] >= x_min) & (d["values"][poi] <= x_max)].min()
        for d in data
    ])
    y_min, y_max, y_max_line = get_y_range(y_min_value, y_max_value, y_min, y_max, log=y_log)

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
    r.setup_hist(
        h_dummy,
        pad=pad,
        props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max},
    )
    draw_objs.append((h_dummy, "HIST"))

    if show_significances:
        # horizontal significance lines and labels
        if isinstance(show_significances, (list, tuple)):
            sigs = list(show_significances)
        else:
            sigs = list(range(1, 9 + 1))
        for sig in sigs:
            # get the dnll2 value and build the label
            is_cl = isinstance(sig, float) and sig < 1
            if is_cl:
                # convert confidence level to dnll2 value
                dnll2 = get_chi2_level_from_cl(sig, 1)
            else:
                # convert significance to dnll2 value
                sig = int(round(sig))
                dnll2 = get_chi2_level(sig, 1)

            # do not show when vertically out of range
            if dnll2 >= y_max_line:
                continue

            # vertical indicators at 1 and 2 sigma when only one curve is shown
            if show_best_fit_indicators and len(data) == 1 and scans[0] and sig in [1, 2]:
                values = map(lambda s: getattr(scans[0], "poi_{}{}".format(s, sig)), "pm")
                for value in values:
                    if value is None or not (x_min < value < x_max):
                        continue
                    line = ROOT.TLine(value, y_min, value, scans[0].interp(value))
                    r.setup_line(
                        line,
                        props={"LineColor": colors.black, "LineStyle": 2, "NDC": False},
                    )
                    draw_objs.append(line)

            # create the line
            sig_line = ROOT.TLine(x_min, dnll2, x_max, dnll2)
            r.setup_line(
                sig_line,
                props={"NDC": False, "LineWidth": 1},
                color=colors.grey,
            )
            draw_objs.append(sig_line)

            # create and position the label
            if y_log:
                sig_label_y = math.log(dnll2 / y_min) / math.log(y_max / y_min)
            else:
                sig_label_y = dnll2 / (y_max - y_min)
            sig_label_y *= 1. - pad.GetTopMargin() - pad.GetBottomMargin()
            sig_label_y += pad.GetBottomMargin() + 0.00375
            if is_cl:
                sig_label = "{:f}".format(sig * 100).rstrip("0").rstrip(".") + "%"
            else:
                sig_label = "{}#sigma".format(sig)
            sig_label = r.routines.create_top_right_label(
                sig_label,
                pad=pad,
                x_offset=5,
                y=sig_label_y,
                props={"TextSize": 18, "TextColor": colors.grey, "TextAlign": 31},
            )
            draw_objs.append(sig_label)

    # draw verical lines at requested values
    if v_lines:
        for x in v_lines:
            if x_min < x < x_max:
                line = ROOT.TLine(x, y_min, x, y_max_line)
                r.setup_line(
                    line,
                    props={"LineColor": colors.black, "LineStyle": 2, "NDC": False},
                )
                draw_objs.append(line)

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(d["name"] in br_hh_colors.root for d in data):
        _color_sequence = [br_hh_colors.root[d["name"]] for d in data]

    # perform scans and draw nll curves
    parameter_ranges = OrderedDict()
    g_nlls = []
    for d, scan, col, ms in zip(data, scans, _color_sequence, marker_sequence):
        if not scan:
            warn("1D likelihood evaluation failed for entry '{}'".format(d["name"]))

        # draw the curve
        g_nll = create_tgraph(
            len(d["values"][poi]),
            d["values"][poi],
            d["values"]["dnll2"],
        )
        r.setup_graph(
            g_nll,
            props={"LineWidth": 2, "MarkerStyle": ms, "MarkerSize": 1.2},
            color=colors[col],
        )
        draw_objs.append((g_nll, "SAME,CP" if show_points else "SAME,C"))
        g_nlls.append(g_nll)

        # legend entry with optional best fit value
        g_label = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        if scan and show_best_fit:
            if show_best_fit_error:
                bf_label = scan.num_min.str(
                    format="%.2f",
                    style="root",
                    force_asymmetric=True,
                    styles={"space": ""},
                )
            else:
                bf_label = "{:.2f}".format(scan.num_min())
            if g_label:
                g_label += ", {}".format(bf_label)
            else:
                g_label = "{} = {}".format(to_root_latex(poi_data[poi].label), bf_label)
        legend_entries.append((g_nll, g_label, "LP" if show_points else "L"))

        # vertical line denoting the best fit value
        if show_best_fit_line and scan and (x_min <= scan.poi_min <= x_max):
            line_fit = ROOT.TLine(scan.poi_min, y_min, scan.poi_min, y_max_line)
            r.setup_line(
                line_fit,
                props={"LineWidth": 2, "NDC": False},
                color=colors[col],
            )
            draw_objs.append(line_fit)

        # store parameter ranges
        key = poi
        if d["name"]:
            key += "__{}".format(d["name"])
        parameter_ranges[key] = scan["summary"] if scan else None

    # theory prediction with uncertainties
    if theory_value:
        has_thy_err = len(theory_value) == 3
        if has_thy_err:
            # theory graph
            g_thy = create_tgraph(
                1,
                theory_value[0],
                y_min,
                theory_value[2],
                theory_value[1],
                0,
                y_max_line - y_min,
            )
            r.setup_graph(
                g_thy,
                props={"LineColor": colors.red, "FillStyle": 1001, "FillColor": colors.red_trans_50},
            )
            draw_objs.append((g_thy, "SAME,02"))
            legend_entries.append((g_thy, "Standard Model", "LF"))
        # theory line
        line_thy = ROOT.TLine(theory_value[0], y_min, theory_value[0], y_max_line)
        r.setup_line(line_thy, props={"NDC": False}, color=colors.red)
        draw_objs.append(line_thy)
        if not has_thy_err:
            legend_entries.append((line_thy, "Standard Model", "L"))

    # fill hep data
    if hep_data:
        # scan value as independent variable
        scan_values = sorted(set(chain(*(map(float, d["values"][poi]) for d in data))))
        hdt.create_independent_variable(
            poi_data[poi].label,
            parent=hep_data,
            values=[Number(v, default_format=-2) for v in scan_values],
        )

        # dnll2 values as dependent variables
        for d, g_nll in zip(data, g_nlls):
            label = r"$-2\Delta\log(L)$"
            if d.get("name"):
                label += ", " + d["name"]
            hdt.create_dependent_variable_from_graph(
                g_nll,
                x_values=scan_values,
                parent=hep_data,
                label=label,
                rounding_method=-2,
                transform=(lambda i, x, y, err: (x, max(y, 0.0), err)),
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

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)

    # save parameter ranges
    if ranges_path:
        ranges_path = prepare_output(ranges_path)
        with open(ranges_path, "w") as f:
            json.dump(parameter_ranges, f, indent=4)
        print("saved parameter ranges to {}".format(ranges_path))

    # save hep data
    if hep_data_path:
        hdt.save_hep_data(hep_data, hep_data_path)


@use_style("dhi_default")
def plot_likelihood_scan_2d(
    paths,
    poi1,
    poi2,
    values,
    hep_data_path=None,
    poi1_min=None,
    poi2_min=None,
    show_best_fit=False,
    show_best_fit_error=False,
    show_significances=(1, 2, 3, 5),
    shift_negative_values=False,
    interpolate_nans=False,
    interpolate_above=None,
    interpolation_method="tgraph2d",
    show_sm_point=True,
    show_box=False,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    model_parameters=None,
    campaign=None,
    eft_lines=None,
    cms_postfix=None,
    style=None,
):
    """
    Creates a likelihood plot of the 2D scan of two POIs *poi1* and *poi2*, and saves it at *paths*.
    *values* should be a mapping to lists of values or a record array with keys "<poi1_name>",
    "<poi2_name>" and "dnll2". When *poi1_min* and *poi2_min* are set, they should be the values of
    the POIs that lead to the best likelihood. Otherwise, they are estimated from the interpolated
    curve. By default, this plot fills a 2D histogram with likelihood values and optionally draws
    contour lines and additional information on top. When *hep_data_path* is set, a yml data file
    compatible with the HEPData format
    (https://hepdata-submission.readthedocs.io/en/latest/data_yaml.html) is stored at that path.

    When *show_best_fit* (*show_best_fit_error*) is *True*, the nominal (uncertainty on the) best
    fit value is drawn. To overlay lines and labels denoting integer significances corresponding to
    1D likelihood scans, *show_significances* can be set to *True* to show significances up to 3
    sigma, or a list of sigmas (integer, >= 1) or confidence levels (float, < 1). In case there are
    negative dnll2 values, *shift_negative_values* can be set to *True* to shift them vertically so
    that the minimum is located at 0 again. Points where the dnll2 value is NaN are visualized as
    white pixels by default. However, when *interpolate_nans* is set, these values are smoothed out
    with information from neighboring pixels through ROOT's TGraph2D.Interpolate feature (similar to
    how its line interpolation draws values between two discrete points in a 1D graph). When
    *interpolate_above* is defined, the same interpolation is applied to values that exceed this
    threshold. *interpolation_method* can either be "tgraph2d" (TGraph2D), "linear" or "cubic"
    (scipy.interpolate's interp2d or griddata), or "rbf" (scipy.interpolate.Rbf). In case a tuple is
    passed, the method should be the first element, followed by optional configuration options.

    The standard model point at (1, 1) as drawn as well unless *show_sm_point* is *False*. The best
    fit value is drawn with uncertainties on one POI being estimated while setting the other POI to
    its best value. When *show_box* is *True*, a box containing the 1 sigma contour is shown and
    used to estimate the dimensions of the standard error following the prescription at
    https://pdg.lbl.gov/2020/reviews/rpp2020-rev-statistics.pdf (e.g. Fig. 40.5).

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. *z_min* and *z_max* limit the range of the z-axis.
    *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign* should
    refer to the name of a campaign label defined in *dhi.config.campaign_labels*. When *paper* is
    *True*, certain plot configurations are adjusted for use in publications.

    *eft_lines* can the the path to a file containing options to draw predefined theory lines
    to be added to 2D likelihood plots.

    Setting *style* leads to slight variations of the plot style. Valid options are:

    - "paper"
    - "contours": Only draw 1 and 2 sigma contour lines over a white background and hide the z-axis.
    - "contours_hcomb": Same as "contours", but line styles, text sizes, etc. are similar to the
                        kf-kV plots of the HComb group.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None
    if style.matches("contours*"):
        show_significances = (1, 2)
        style.significance_labels = ["68% CL", "95% CL"]

    # prepare hep data
    hep_data = None
    if hep_data_path:
        hep_data = hdt.create_hist_data()

    # activate a different style of contours_hcomb
    style_changed = False
    if style.matches("contours_hcomb"):
        s = r.styles.copy(r.styles.current_style_name, "contours_hcomb")
        s.pad.TopMargin = 0.075
        s.pad.BottomMargin = 0.16
        s.pad.LeftMargin = 0.16
        s.legend.TextSize = 26
        s.legend.LineStyle = 1
        s.legend.LineColor = 1
        s.legend.LineWidth = 1
        s.legend.ColumnSeparation = 0.1
        s.legend_dy = 0.07
        s.x_axis.LabelSize = 30
        s.x_axis.TitleSize = 50
        s.x_axis.TitleOffset = 0.8
        s.y_axis.LabelSize = 30
        s.y_axis.TitleSize = 50
        s.y_axis.TitleOffset = 0.8
        # use it
        r.styles.push("contours_hcomb")
        style_changed = True

    # check values
    values = make_list(values)
    for i, _values in enumerate(list(values)):
        if isinstance(_values, np.ndarray):
            _values = {key: np.array(_values[key]) for key in _values.dtype.names}
        assert poi1 in _values
        assert poi2 in _values
        assert "dnll2" in _values
        # drop all fields except for required ones
        _values = {k: v for k, v in _values.items() if k in [poi1, poi2, "dnll2"]}
        # preprocess values (nan detection, negative shift)
        _values["dnll2"], _values[poi1], _values[poi2] = _preprocess_values(
            _values["dnll2"],
            (poi1, _values[poi1]),
            (poi2, _values[poi2]),
            remove_nans=interpolate_nans,
            shift_negative_values=shift_negative_values,
            remove_above=interpolate_above,
            min_is_external=poi1_min is not None and poi2_min is not None,
        )
        values[i] = _values

    # join values for contour calculation
    joined_values = unique_recarray(dict_to_recarray(values), cols=[poi1, poi2])

    # determine contours independent of plotting
    contour_levels = [1, 2, 3, 4, 5]
    contour_colors = {
        1: colors.brazil_green,
        2: colors.brazil_yellow,
        3: colors.blue_cream,
        4: colors.orange,
        5: colors.red_cream,
    }
    if show_significances and isinstance(show_significances, (list, tuple)):
        contour_levels = list(show_significances)
    n_contours = len(contour_levels)

    # convert to dnll2 values for 2 degrees of freedom
    contour_levels_dnll2 = []
    for l in contour_levels:
        is_cl = isinstance(l, float) and l < 1
        dnll2 = get_chi2_level_from_cl(l, 2) if is_cl else get_chi2_level(l, 2)
        contour_levels_dnll2.append(dnll2)

    # refine colors and styles
    if style.matches("contours*"):
        contour_colors = n_contours * [colors.black]
        contour_styles = ([1, 7, 2, 9] + max(n_contours - 4, 0) * [8])[:n_contours]
    else:
        rest_colors = list(color_sequence)
        contour_colors = [
            (contour_colors[l] if l in contour_colors else rest_colors.pop(0))
            for l in contour_levels
        ]
        contour_styles = n_contours * [1]

    # get the contours
    contours = get_contours(
        joined_values[poi1],
        joined_values[poi2],
        joined_values["dnll2"],
        levels=contour_levels_dnll2,
        frame_kwargs=[{"mode": "edge", "width": 1.}],
        interpolation=interpolation_method,
    )

    # evaluate the scan, run interpolation and error estimation
    scan = evaluate_likelihood_scan_2d(
        joined_values[poi1],
        joined_values[poi2],
        joined_values["dnll2"],
        poi1_min=poi1_min,
        poi2_min=poi2_min,
        contours=contours[:1],
    )
    if not scan:
        warn("2D likelihood evaluation failed")

    # reset the box flag if necessary
    if show_box and (not scan or not scan.box_nums[0][0] or not scan.box_nums[0][1]):
        warn("disabling show_box due to missing or failed scan")
        show_box = False

    # start plotting
    r.setup_style()
    pad_props = {} if style.matches("contours*") else {"RightMargin": 0.17, "Logz": True}
    canvas, (pad,) = r.routines.create_canvas(pad_props=pad_props)
    pad.cd()
    draw_objs = []

    # create a histogram for each scan patch
    hists = []
    for i, _values in enumerate(values):
        _, _, _x_bins, _y_bins, _x_min, _x_max, _y_min, _y_max = infer_binning_from_grid(
            _values[poi1],
            _values[poi2],
        )

        # get the z range
        dnll2 = np.array(_values["dnll2"])
        _z_min = np.nanmin(dnll2)
        _z_max = np.nanmax(dnll2)

        # when there is no negative value, shift zeros to 0.1 of the smallest, non-zero value
        if _z_min == 0:
            _z_min = 0.1 * dnll2[dnll2 > 0].min()
            dnll2[dnll2 == 0] = _z_min

        # infer axis limits from the first set of values
        if i == 0:
            x_min = _x_min if x_min is None else x_min
            x_max = _x_max if x_max is None else x_max
            y_min = _y_min if y_min is None else y_min
            y_max = _y_max if y_max is None else y_max
            z_min = _z_min if z_min is None else z_min

        # when there are still NaN's, set them to values right below the z_min,
        # which causes ROOT to draw white pixels
        z_min_fill = z_min
        nan_mask = np.isnan(dnll2)
        if nan_mask.sum() and not interpolate_nans:
            warn(
                "WARNING: {} NaN(s) will be drawn as white pixels; consider enabling NaN "
                "interpolation (--interpolate-nans when triggered by a law task)".format(
                    nan_mask.sum(),
                ),
            )
            dnll2[nan_mask] = 0.9 * z_min
            z_min_fill = None

        # fill and store the histogram
        h = ROOT.TH2F("h" + str(i), "", _x_bins, _x_min, _x_max, _y_bins, _y_min, _y_max)
        fill_hist_from_points(
            h,
            _values[poi1],
            _values[poi2],
            dnll2,
            z_min=z_min_fill,
            interpolation=interpolation_method,
        )
        hists.append(h)

        # infer z_max separately after possible extrapolation
        if i == 0:
            z_max = _z_max if z_max is None else z_max

    # dummy histogram to control axes
    x_title = to_root_latex(poi_data[poi1].label)
    y_title = to_root_latex(poi_data[poi2].label)
    z_title = "-2 #Delta log(L)"
    h_dummy = ROOT.TH2F(
        "h_nll",
        ";{};{};{}".format(x_title, y_title, z_title),
        1,
        x_min,
        x_max,
        1,
        y_min,
        y_max,
    )
    r.setup_hist(
        h_dummy,
        pad=pad,
        props={"Contour": 100, "Minimum": z_min, "Maximum": z_max},
    )
    draw_objs.append((h_dummy, ""))
    legend_entries = []

    # setup actual histograms
    if not style.matches("contours*"):
        for i, h in enumerate(hists):
            r.setup_hist(h, props={"Contour": 100, "Minimum": z_min, "Maximum": z_max})
            if i == 0:
                r.setup_z_axis(
                    h.GetZaxis(),
                    pad=pad,
                    props={"Title": z_title, "TitleSize": 24, "TitleOffset": 1.5},
                )
            draw_objs.append((h, "SAME,COLZ"))
            # for debugging purposes
            # draw_objs.append((h, "SAME,TEXT"))

    # significance contours
    if show_significances:
        # conversion factor from pixel to x-axis range
        pad_width = canvas.GetWindowWidth() * (1. - pad.GetLeftMargin() - pad.GetRightMargin())
        pad_height = canvas.GetWindowHeight() * (1. - pad.GetTopMargin() - pad.GetBottomMargin())
        px_to_x = (x_max - x_min) / pad_width
        py_to_y = (y_max - y_min) / pad_height

        # cache for label positions
        all_positions = []
        for graphs, level, col, ls in zip(contours, contour_levels, contour_colors, contour_styles):
            for g in graphs:
                r.setup_graph(g, props={"LineWidth": 2, "LineColor": colors(col), "LineStyle": ls})
                draw_objs.append((g, "SAME,C"))

            # stop here when only drawing contours
            if style.matches("contours*"):
                continue

            # get the approximate label width
            is_cl = isinstance(level, float) and level < 1
            if is_cl:
                text = "{:f}".format(level * 100).rstrip("0").rstrip(".") + "%"
            else:
                text = "{}#sigma".format(level)
            label_width, label_height = get_text_extent(text, 18, 43)
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
                label_offset=0.9,
            )
            all_positions.extend(label_positions)
            pad.cd()

            # draw them
            for x, y, rot in label_positions:
                sig_label = ROOT.TLatex(0., 0., text)
                r.setup_latex(
                    sig_label,
                    props={
                        "NDC": False, "TextSize": 16, "TextAlign": 21, "TextColor": colors(col),
                        "TextAngle": rot, "X": x, "Y": y,
                    },
                )
                draw_objs.append((sig_label, "SAME"))

    # draw the first contour box
    if show_box and scan:
        box_num1, box_num2 = scan.box_nums[0]
        box_t = ROOT.TLine(box_num1("down"), box_num2("up"), box_num1("up"), box_num2("up"))
        box_b = ROOT.TLine(box_num1("down"), box_num2("down"), box_num1("up"), box_num2("down"))
        box_r = ROOT.TLine(box_num1("up"), box_num2("up"), box_num1("up"), box_num2("down"))
        box_l = ROOT.TLine(box_num1("down"), box_num2("up"), box_num1("down"), box_num2("down"))
        for box_line in [box_t, box_r, box_b, box_l]:
            r.setup_line(
                box_line,
                props={"LineColor": colors.gray, "NDC": False},
            )
            draw_objs.append(box_line)
        box_legend_entry = ROOT.TH1F("box_hist", "", 1, 0, 1)
        r.setup_hist(box_legend_entry, props={"FillStyle": 0})

    # SM point
    if show_sm_point:
        g_sm = create_tgraph(1, poi_data[poi1].sm_value, poi_data[poi2].sm_value)
        r.setup_graph(
            g_sm,
            color=colors.red,
            props={"MarkerStyle": 33, "MarkerSize": 2.5},
        )
        draw_objs.append((g_sm, "P"))
        legend_entries.append((g_sm, "SM Higgs", "P"))
        # yellow overlay for hcomb style
        if style.matches("contours_hcomb"):
            g_sm2 = create_tgraph(1, poi_data[poi1].sm_value, poi_data[poi2].sm_value)
            r.setup_graph(
                g_sm2,
                color=89,
                props={"MarkerStyle": 33, "MarkerSize": 1.4},
            )
            draw_objs.append((g_sm2, "P"))

    # central best fit point
    if show_best_fit and scan:
        g_fit = ROOT.TGraphAsymmErrors(1)
        g_fit.SetPoint(0, scan.num1_min(), scan.num2_min())
        if scan.num1_min.uncertainties and show_best_fit_error:
            g_fit.SetPointEXhigh(0, scan.num1_min.u(direction="up"))
            g_fit.SetPointEXlow(0, scan.num1_min.u(direction="down"))
        if scan.num2_min.uncertainties and show_best_fit_error:
            g_fit.SetPointEYhigh(0, scan.num2_min.u(direction="up"))
            g_fit.SetPointEYlow(0, scan.num2_min.u(direction="down"))
        props = {"MarkerStyle": 43, "MarkerSize": 2}
        if show_best_fit_error:
            props = {}
        elif style.matches("contours*"):
            props = {"MarkerStyle": 34, "MarkerSize": 2}
        r.setup_graph(g_fit, props=props, color=colors.black)
        draw_objs.append((g_fit, "PEZ" if show_best_fit_error else "PZ"))

    # EFT lines
    if eft_lines:
        with open(eft_lines, "r") as f:
            lines = json.load(f)
            for line in lines:
                if line["poi1"] != poi1 or line["poi2"] != poi2:
                    continue
                # creata a function object
                line_func = ROOT.TF1(line["label"], line["eq"], x_min, x_max)
                r.setup_func(
                    line_func,
                    props={"LineWidth": 3, "LineStyle": int(line["style"])},
                    color=int(line["color"]),
                )
                draw_objs.append((line_func, "SAME"))
                legend_entries.append((line_func, line["label"], "L"))

    # fill hep data
    if hep_data:
        # use the first underlying histogram and add its axes as two independent variables
        h = hists[0]
        x_bins = list(range(1, h.GetXaxis().GetNbins() + 1))
        y_bins = list(range(1, h.GetYaxis().GetNbins() + 1))
        coords = list(product(x_bins, y_bins))
        hdt.create_independent_variable(
            poi_data[poi1].label,
            parent=hep_data,
            values=[
                Number(h.GetXaxis().GetBinCenter(bx), default_format=-2)
                for bx, _ in coords
            ],
        )
        hdt.create_independent_variable(
            poi_data[poi2].label,
            parent=hep_data,
            values=[
                Number(h.GetYaxis().GetBinCenter(by), default_format=-2)
                for _, by in coords
            ],
        )

        # dnll2 values as dependent variable
        hdt.create_dependent_variable(
            r"$-2\Delta\log(L)$",
            parent=hep_data,
            values=[
                Number(max(h.GetBinContent(bx, by), 0.0), default_format=-2)
                for bx, by in coords
            ])

    # legend
    def make_bf_label(num1, num2):
        if show_best_fit_error:
            return "{} = {} ,  {} = {}".format(
                to_root_latex(poi_data[poi1].label),
                (
                    "-"
                    if num1 is None
                    else num1.str(
                        format="%.2f",
                        style="root",
                        force_asymmetric=True,
                        styles={"space": ""},
                    )
                ),
                to_root_latex(poi_data[poi2].label),
                (
                    "-"
                    if num2 is None
                    else num2.str(
                        format="%.2f",
                        style="root",
                        force_asymmetric=True,
                        styles={"space": ""},
                    )
                ),
            )
        else:
            return "{} = {:.2f} ,  {} = {:.2f}".format(
                to_root_latex(poi_data[poi1].label),
                "-" if num1 is None else num1(),
                to_root_latex(poi_data[poi2].label),
                "-" if num2 is None else num2(),
            )

    if show_box:
        legend_entries.insert(0, (box_legend_entry, make_bf_label(box_num1, box_num2), "F"))
    if show_best_fit and scan:
        label = "Observed" if style == "paper" else make_bf_label(scan.num1_min, scan.num2_min)
        legend_entries.insert(0, (g_fit, label, "PLE" if show_best_fit_error else "P"))
    if style.matches("contours*"):
        for graphs, level in zip(contours, style.significance_labels):
            for g in graphs:
                legend_entries.append((g, level, "L"))
    if legend_entries:
        legend_kwargs = {"pad": pad, "width": 340, "n": len(legend_entries)}
        if eft_lines:
            legend_kwargs["width"] = 550
        if style.matches("contours*"):
            legend_kwargs["n"] = 2
            legend_kwargs["props"] = {"NColumns": 2}
            legend_kwargs["width"] = 400 if style == "contours_hcomb" else 260
        legend = r.routines.create_legend(**legend_kwargs)
        r.fill_legend(legend, legend_entries)
        draw_objs.append(legend)

        # draw the overlay SM point again for hcomb style (depends highly on the legend position)
        if show_sm_point and style.matches("contours_hcomb"):
            g_sm2_legend = g_sm2.Clone()
            g_sm2_legend.SetPoint(1, 1.525, 1.685)
            draw_objs.append((g_sm2_legend, "P"))

    # cms label
    cms_layout = "outside_horizontal"
    cms_props = {"text_size": 44} if style.matches("contours_hcomb") else {}
    cms_labels = r.routines.create_cms_labels(
        pad=pad,
        postfix=cms_postfix or "",
        layout=cms_layout,
        **cms_props  # noqa
    )
    draw_objs.extend(cms_labels)

    # model parameter labels
    if model_parameters:
        param_kwargs = {}
        param_kwargs["props"] = {"TextSize": 30} if style.matches("contours_hcomb") else {}
        if cms_layout.startswith("inside"):
            y_offset = 100 if cms_layout == "inside_vertical" and cms_postfix else 80
            param_kwargs = {"y_offset": y_offset}
        draw_objs.extend(create_model_parameters(model_parameters, pad, **param_kwargs))

    # campaign label
    if campaign:
        props = {"TextSize": 40} if style.matches("contours_hcomb") else {}
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad, props=props)
        draw_objs.append(campaign_label)

    # draw all objects
    r.routines.draw_objects(draw_objs)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)

    # remove custom styles
    if style_changed:
        r.styles.pop()

    # save hep data
    if hep_data_path:
        hdt.save_hep_data(hep_data, hep_data_path)


@use_style("dhi_default")
def plot_likelihood_scans_2d(
    paths,
    poi1,
    poi2,
    data,
    shift_negative_values=False,
    interpolate_nans=True,
    interpolate_above=None,
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
    Creates the likelihood contour plots of multiple 2D scans of two POIs *poi1* and *poi2*, and
    saves it at *paths*. All information should be passed as a list *data*. Entries must be
    dictionaries with the following content:

        - "values": A mapping to lists of values or a record array with keys "<poi1_name>",
          "<poi2_name>" and "dnll2".
        - "poi_mins": A list of two floats describing the best fit value of the two POIs. When not
          set, the minima are estimated from the interpolated curve.
        - "name": A name of the data to be shown in the legend.

    In case there are negative dnll2 values, *shift_negative_values* can be set to *True* to shift
    them vertically so that the minimum is located at 0 again.

    *x_min*, *x_max*, *y_min* and *y_max* define the axis range of *poi1* and *poi2*, respectively,
    and default to the ranges of the poi values. When *interpolate_nans* is *True*, points with
    failed fits, denoted by nan values, are filled with the averages of neighboring fits. When
    *interpolate_above* is defined, the same interpolation is applied to values that exceed this
    threshold. *interpolation_method* can either be "tgraph2d" (TGraph2D), "linear" or "cubic"
    (scipy.interpolate's interp2d or griddata), or "rbf" (scipy.interpolate.Rbf). In case a tuple is
    passed, the method should be the first element, followed by optional configuration options.

    When *model_parameters* can be a dictionary of key-value pairs of model parameters. *campaign*
    should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.
    *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: Example: https://cms-hh.web.cern.ch/tools/inference/tasks/likelihood.html#2d_1
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # validate data entries
    for i, d in enumerate(data):
        # convert likelihood values to arrays
        assert "values" in d
        values = d["values"]
        if isinstance(values, np.ndarray):
            values = {k: values[k] for k in values.dtype.names}
        assert poi1 in values
        assert poi2 in values
        assert "dnll2" in values
        # check poi minima
        d["poi_mins"] = d.get("poi_mins") or [None, None]
        assert len(d["poi_mins"]) == 2
        # default name
        d.setdefault("name", str(i + 1))
        # origin (for printouts)
        d["origin"] = None if not d["name"] else "entry '{}'".format(d["name"])
        # drop all fields except for required ones and convert to arrays
        values = {
            k: np.array(v, dtype=np.float32)
            for k, v in values.items()
            if k in [poi1, poi2, "dnll2"]
        }
        # preprocess values (nan detection, negative shift)
        values["dnll2"], values[poi1], values[poi2] = _preprocess_values(
            values["dnll2"],
            (poi1, values[poi1]),
            (poi2, values[poi2]),
            shift_negative_values=shift_negative_values,
            remove_nans=interpolate_nans,
            remove_above=interpolate_above,
            origin=d["origin"],
            min_is_external=None not in d["poi_mins"],
        )
        d["values"] = values

    # determine contours independent of plotting
    contours = [
        get_contours(
            d["values"][poi1],
            d["values"][poi2],
            d["values"]["dnll2"],
            levels=[chi2_levels[2][1], chi2_levels[2][2]],
            frame_kwargs=[{"mode": "edge"}],
            interpolation=interpolation_method,
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

    # special case regarding color handling: when all entry names are valid keys in br_hh_colors,
    # replace the default color sequence to deterministically assign same colors to channels
    _color_sequence = color_sequence
    if all(d["name"] in br_hh_colors.root for d in data):
        _color_sequence = [br_hh_colors.root[d["name"]] for d in data]

    # loop through data entries
    for d, (cont1, cont2), col in zip(data, contours, _color_sequence[:len(data)]):
        # evaluate the scan
        scan = evaluate_likelihood_scan_2d(
            d["values"][poi1],
            d["values"][poi2],
            d["values"]["dnll2"],
            poi1_min=d["poi_mins"][0],
            poi2_min=d["poi_mins"][1],
        )
        if not scan:
            warn("2D likelihood evaluation failed for entry '{}'".format(d["name"]))

        # plot 1 and 2 sigma contours
        g1, g2 = None, None
        for g1 in cont1:
            r.setup_graph(g1, props={"LineWidth": 2, "LineStyle": 1, "LineColor": colors[col]})
            draw_objs.append((g1, "SAME,C"))
        for g2 in cont2:
            r.setup_graph(g2, props={"LineWidth": 2, "LineStyle": 2, "LineColor": colors[col]})
            draw_objs.append((g2, "SAME,C"))
        name = to_root_latex(br_hh_names.get(d["name"], d["name"]))
        if g1:
            legend_entries.append((g1, name, "L"))

        # best fit point
        if scan:
            g_fit = create_tgraph(1, scan.num1_min(), scan.num2_min())
            r.setup_graph(g_fit, props={"MarkerStyle": 33, "MarkerSize": 2}, color=colors[col])
            draw_objs.append((g_fit, "SAME,PEZ"))

    # append legend entries to show styles
    g_fit_style = g_fit.Clone()
    r.apply_properties(g_fit_style, {"MarkerColor": colors.black})
    legend_entries.append((g_fit_style, "Best fit value", "P"))
    if g1:
        g1_style = g1.Clone()
        r.apply_properties(g1_style, {"LineColor": colors.black})
        legend_entries.append((g1_style, "#pm 1 #sigma", "L"))
    else:
        warn("no primary contour found, no line will be visible")
    if g2:
        g2_style = g2.Clone()
        r.apply_properties(g2_style, {"LineColor": colors.black})
        legend_entries.append((g2_style, "#pm 2 #sigma", "L"))
    else:
        warn("no secondary contour found, no line will be visible")

    # prepend empty values
    n_empty = 3 - (len(legend_entries) % 3)
    if n_empty not in (0, 3):
        for _ in range(n_empty):
            legend_entries.insert(3 - n_empty, (h_dummy, " ", "L"))

    # legend with actual entries in different colors
    legend_cols = int(math.ceil(len(legend_entries) / 3.))
    legend_rows = min(len(legend_entries), 3)
    legend = r.routines.create_legend(
        pad=pad,
        width=legend_cols * 150,
        height=legend_rows * 30,
        props={"NColumns": legend_cols},
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

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)


@use_style("dhi_default")
def plot_nuisance_likelihood_scans(
    paths,
    poi,
    workspace,
    dataset,
    fit_diagnostics_path,
    fit_name="fit_s",
    skip_parameters=None,
    only_parameters=None,
    parameters_per_page=1,
    sort_max=False,
    show_diff=False,
    labels=None,
    scan_points=401,
    x_min=-2.,
    x_max=2,
    y_min=None,
    y_max=None,
    y_log=False,
    model_parameters=None,
    campaign=None,
    show_derivatives=False,
    cms_postfix=None,
    style=None,
):
    r"""
    Creates a plot showing the change of the negative log-likelihood, previously obtained for a
    *poi*, when varying values of nuisance paramaters and saves it at *paths*. The calculation of
    the likelihood change requires the RooFit *workspace* to read the model config, a RooDataSet
    *dataset* to construct the functional likelihood, and the output file *fit_diagnostics_path* of
    the combine fit diagnostics for reading pre- and post-fit parameters for the fit named
    *fit_name*, defaulting to ``"fit_s"``.

    Nuisances to skip, or to show exclusively can be configured via *skip_parameters* and
    *only_parameters*, respectively, which can be lists of patterns. *parameters_per_page* defines
    the number of parameter curves that are drawn in the same canvas page. When *sort_max* is
    *True*, the parameter are sorted by their highest likelihood change in the full scan range.
    By default, the x-axis shows absolute variations of the nuisance parameters (in terms of the
    prefit range). When *show_diff* is *True*, it shows differences with respect to the best fit
    value instead.

    *labels* should be a dictionary or a json file containing a dictionary that maps nuisances names
    to labels shown in the plot, a python file containing a function named "rename_nuisance", or a
    function itself. When it is a dictionary and a key starts with "^" and ends with "$" it is
    interpreted as a regular expression. Matched groups can be reused in the substituted name via
    '\n' to reference the n-th group (following the common re.sub format). When it is a function, it
    should accept the current nuisance label as a single argument and return the new value.

    The scan range and granularity is set via *scan_points*, *x_min* and *x_max*. *y_min* and
    *y_max* define the range of the y-axis, which is plotted with a logarithmic scale when *y_log*
    is *True*. *model_parameters* can be a dictionary of key-value pairs of model parameters.
    *campaign* should refer to the name of a campaign label defined in *dhi.config.campaign_labels*.
    The first and second order derivatives of the negative log likelihood function are added by
    *show_derivatives*. *cms_postfix* is shown as the postfix behind the CMS label.

    Supported values for *style*:

        - "paper"

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/postfit.html#nuisance-parameter-influence-on-likelihood  # noqa
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # style-based adjustments
    style = Style.new(style)
    if style.matches("paper"):
        cms_postfix = None

    # get the best fit value and prefit data from the diagnostics file
    f = ROOT.TFile(fit_diagnostics_path, "READ")
    best_fit = f.Get(fit_name)
    fit_args = best_fit.floatParsFinal()
    prefit_params = convert_rooargset(f.Get("nuisances_prefit"))

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

    # filter parameters
    param_names = []
    for param_name in prefit_params:
        if only_parameters and not multi_match(param_name, only_parameters):
            continue
        if skip_parameters and multi_match(param_name, skip_parameters):
            continue
        param_names.append(param_name)
    print("preparing scans of {} parameter(s)".format(len(param_names)))

    # prepare the scan values, extend the range by 10 points in each directon, ensure 0 is contained
    assert scan_points > 1
    width = float(x_max - x_min) / (scan_points - 1)
    scan_values = np.linspace(x_min - 10 * width, x_max + 10 * width, scan_points + 20).tolist()
    if 0 not in scan_values:
        scan_values = sorted(scan_values + [0.])

    # get nll curve values for all parameters before plotting to be able to sort
    curve_data = {}
    for name in param_names:
        pre_u, pre_d = prefit_params[name][1:3]
        workspace.loadSnapshot(snapshot_name)
        param = workspace.var(name)
        if not param:
            raise Exception("parameter {} not found in workspace".format(name))
        param_bf = param.getVal()
        nll_base = nll.getVal()
        if show_derivatives:
            grad1 = nll.derivative(param, 1)
            grad2 = nll.derivative(param, 2)
            dy_values, ddy_values = [], []
        print("scanning parameter {}".format(name))
        x_values, y_values = [], []
        for x in scan_values:
            x_diff = x * (pre_u if x >= 0 else -pre_d)
            param.setVal(param_bf + x_diff)
            x_values.append(x_diff if show_diff else (param_bf + x_diff))
            y_values.append(2 * (nll.getVal() - nll_base))
            if show_derivatives:
                dy_values.append(2 * grad1.getVal())
                ddy_values.append(2 * grad2.getVal())
        curve_data[name] = OrderedDict({"nll": (x_values, y_values)})
        if show_derivatives:
            curve_data[name]["grad1"] = (x_values, dy_values)
            curve_data[name]["grad2"] = (x_values, ddy_values)

    # sort?
    if sort_max:
        param_names.sort(key=lambda name: -max(curve_data[name]["nll"][1]))

    # group parameters
    param_groups = [[]]
    for name in param_names:
        if only_parameters and not multi_match(name, only_parameters):
            continue
        if skip_parameters and multi_match(name, skip_parameters):
            continue
        if parameters_per_page < 1 or len(param_groups[-1]) < parameters_per_page:
            param_groups[-1].append(name)
        else:
            param_groups.append([name])

    # prepare labels
    labels = make_parameter_label_map(param_names, labels)

    # go through nuisances
    canvas = None
    for names in param_groups:
        # setup the default style and create canvas and pad
        first_canvas = canvas is None
        r.setup_style()
        canvas, (pad,) = r.routines.create_canvas(pad_props={"Logy": y_log})
        pad.cd()

        # start the multi pdf file
        if first_canvas:
            for path in make_list(paths):
                canvas.Print(path + ("[" if path.endswith(".pdf") else ""))

        # get y range
        y_min_value = min(min(curve_data[name]["nll"][1]) for name in names)
        y_max_value = max(max(curve_data[name]["nll"][1]) for name in names)
        _y_min = y_min
        _y_max = y_max
        _y_min, _y_max, y_max_line = get_y_range(
            0. if y_log else y_min_value,
            y_max_value,
            y_min,
            y_max,
            log=y_log,
        )

        # dummy histogram to control axes
        if show_diff:
            x_title = "(#theta - #theta_{best}) / #Delta#theta_{pre}"
        else:
            x_title = "#theta / #Delta#theta_{pre}"
        y_title = "-2 #Delta log(L)"
        h_dummy = ROOT.TH1F("dummy", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
        r.setup_hist(
            h_dummy,
            pad=pad,
            props={"LineWidth": 0, "Minimum": _y_min, "Maximum": _y_max},
        )
        draw_objs = [(h_dummy, "HIST")]
        legend_entries = []

        # horizontal and vertical guidance lines
        if (_y_min < 1 < y_max_line) and (y_log or y_max_line < 100):
            # horizontal
            line = ROOT.TLine(x_min, 1., x_max, 1.)
            r.setup_line(
                line,
                props={"LineColor": 12, "LineStyle": 2, "NDC": False},
            )
            draw_objs.append(line)

            # vertical
            for x in [-1, 1]:
                line = ROOT.TLine(x, _y_min, x, min(1., y_max_line))
                r.setup_line(
                    line,
                    props={"LineColor": 12, "LineStyle": 2, "NDC": False},
                )
                draw_objs.append(line)

        # nll graphs
        line_styles = {"nll": 1, "grad1": 2, "grad2": 3}
        for name, col in zip(names, color_sequence[:len(names)]):
            for key, (x, y) in curve_data[name].items():
                g_nll = create_tgraph(len(x), x, y)
                r.setup_graph(
                    g_nll,
                    props={"LineWidth": 2, "LineStyle": line_styles.get(key, 1)},
                    color=colors[col],
                )
                draw_objs.append((g_nll, "SAME,L"))
                label = to_root_latex(labels.get(name, name))
                if key != "nll":
                    label += " ({})".format(key)
                legend_entries.append((g_nll, label, "L"))

        # legend
        legend_cols = min(int(math.ceil(len(legend_entries) / 4.)), 3)
        legend_rows = int(math.ceil(len(legend_entries) / float(legend_cols)))
        legend_kwargs = dict(width=legend_cols * 210, n=legend_rows)
        if show_derivatives:
            legend_kwargs["x2"] = -420
        legend = r.routines.create_legend(
            pad=pad,
            props={"NColumns": legend_cols, "TextSize": 16},
            **legend_kwargs  # noqa
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

        # draw objects, update and save
        r.routines.draw_objects(draw_objs)
        r.update_canvas(canvas)
        for path in make_list(paths):
            canvas.Print(path)

    # finish the pdf
    if canvas:
        for path in make_list(paths):
            canvas.Print(path + ("]" if path.endswith(".pdf") else ""))


def _preprocess_values(
    dnll2_values,
    poi1_data,
    poi2_data=None,
    remove_nans=True,
    remove_above=None,
    shift_negative_values=False,
    min_is_external=False,
    origin=None,
    epsilon=1e-5,
):
    # unpack data
    poi1, poi1_values = poi1_data
    poi2, poi2_values = poi2_data or (None, None)
    pois = ", ".join(filter(None, [poi1, poi2]))
    origin = " ({})".format(origin) if origin else ""

    # helper to find poi values for coordinates of a given mask
    def find_coords(mask):
        poi_values = np.stack(filter((lambda v: v is not None), [poi1_values, poi2_values]), axis=1)
        coords = "\n  - ".join(", ".join(map(str, vals)) for vals in poi_values[mask])
        return coords

    # warn about NaNs and remove them
    nan_mask = np.isnan(dnll2_values)
    if nan_mask.sum():
        warn(
            "WARNING: found {} NaN(s) in dnll2 values{}; POI coordinates ({}):\n  - {}".format(
                nan_mask.sum(), origin, pois, find_coords(nan_mask),
            ),
        )
        if remove_nans:
            dnll2_values = dnll2_values[~nan_mask]
            poi1_values = poi1_values[~nan_mask]
            if poi2:
                poi2_values = poi2_values[~nan_mask]
            print("removed {} NaN(s)".format(nan_mask.sum()))

    # warn about values that exceed the interpolation threshold when set
    if remove_above and remove_above > 0:
        above_mask = dnll2_values > remove_above
        if above_mask.sum():
            warn(
                "INFO: found {} high (> {}) dnll2 values{}; POI coordinates ({}):\n  - {}".format(
                    above_mask.sum(), remove_above, origin, pois, find_coords(above_mask),
                ),
            )
            dnll2_values = dnll2_values[~above_mask]
            poi1_values = poi1_values[~above_mask]
            if poi2:
                poi2_values = poi2_values[~above_mask]
            print("removed {} high values".format(above_mask.sum()))

    # warn about negative dnll2 values
    neg_mask = dnll2_values < 0
    if neg_mask.sum():
        neg_min = dnll2_values[dnll2_values < 0].min()
        slightly_neg = neg_min > -epsilon
        # issue a warning about potentially wrong external best fit values
        if min_is_external and not slightly_neg:
            warn(
                "WARNING: {} dnll2 values{} have negative values, implying that combine might have "
                "found a local rather than the global minimum; consider re-running combine with "
                "different fit options or allow this function to recompute the minimum via "
                "scipy.interpolate and scipy.minimize on the likelihood curve by not passing "
                "combine's result (--recompute-best-fit when triggered by a law task); POI "
                "coordinates ({}):\n  - {}".format(
                    neg_mask.sum(), origin, pois, find_coords(neg_mask)),
                color="red",
            )
        # warn again that negative shifts are small and will be changed, or in case they are large
        if slightly_neg:
            shift_negative_values = True
            warn("WARNING: detected slightly negative minimum dnll2 value of {}".format(neg_min))
        elif not shift_negative_values:
            warn(
                "WARNING: consider shifting the dnll2 values vertically to move the minimum back "
                "to 0, which would otherwise lead to wrong uncertainties being extracted from "
                "intersections with certain dnll2 values (--shift-negative-values when triggered "
                "by a law task)",
                color="red",
            )
        # apply the actua shift, skipping nan's in case they were not removed above
        if shift_negative_values:
            neg_min = np.nanmin(dnll2_values)
            dnll2_values[~np.isnan(dnll2_values)] -= neg_min
            print("shifting dnll2 values up by {}".format(-neg_min))

    # when the previous step did not shift values to 0,
    # detect cases where the positive minimum is > 0 and shift values
    if not neg_mask.sum() and (dnll2_values > 0).sum():
        pos_min = dnll2_values[dnll2_values >= 0].min()
        if pos_min > 0:
            slightly_pos = pos_min < epsilon
            if slightly_pos:
                warn(
                    "WARNING: detected slightly positive minimum dnll2 value of {}".format(
                        pos_min,
                    ),
                )
            else:
                warn(
                    "WARNING: minimum dnll2 value found to be {} while it should be 0".format(
                        pos_min,
                    ),
                    color="red",
                )
            dnll2_values[~np.isnan(dnll2_values)] -= pos_min
            print("shifting dnll2 values down by {}".format(pos_min))

    return (dnll2_values, poi1_values) + ((poi2_values,) if poi2 else ())


def evaluate_likelihood_scan_1d(poi_values, dnll2_values, poi_min=None, origin=None):
    """
    Takes the results of a 1D likelihood profiling scan given by the *poi_values* and the
    corresponding *delta_2nll* values, performs an interpolation and returns certain results of the
    scan in a dict. When *poi_min* is *None*, it is estimated from the interpolated curve.

    Please consider preprocessing values with :py:func:`_preprocess_values` first.

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
    - ``summary``: A dictionary with poi minimum, uncertainties and ranges.
    """
    origin = " ({})".format(origin) if origin else ""

    # ensure we are dealing with arrays
    poi_values = np.array(poi_values)
    dnll2_values = np.array(dnll2_values)

    # store ranges
    poi_values_min = poi_values.min()
    poi_values_max = poi_values.max()

    # remove values where dnll2 is NaN
    nan_mask = np.isnan(dnll2_values)
    mask = ~nan_mask
    poi_values = poi_values[mask]
    dnll2_values = dnll2_values[mask]
    n_nans = (~mask).sum()
    if n_nans:
        warn("WARNING: found {} NaN(s) in values{} in 1D likelihood evaluation".format(
            n_nans, origin,
        ))

    # first, obtain an interpolation function
    try:
        # interp = scipy.interpolate.interp1d(poi_values, dnll2_values, kind="linear")
        interp = scipy.interpolate.interp1d(
            poi_values,
            dnll2_values,
            kind="cubic",
            fill_value="extrapolate",
        )
    except:
        return None

    # recompute the minimum and compare with the existing one when given
    xcheck = poi_min is not None
    print("extracting POI minimum{} {}...".format(origin, "as cross check " if xcheck else ""))
    objective = lambda x: interp(x)
    bounds = (poi_values_min + 1e-4, poi_values_max - 1e-4)
    res = minimize_1d(objective, bounds)
    if res.status != 0:
        if not xcheck:
            raise Exception("could not find minimum of dnll2 interpolation: {}".format(res.message))
    else:
        poi_min_new = res.x[0]
        print("done{}, found {:.4f}".format(origin, poi_min_new))
        if xcheck:
            # compare and optionally issue a warning (threshold to be optimized)
            if abs(poi_min - poi_min_new) >= 0.03:
                warn(
                    "WARNING: external POI minimum {:.4f}{} (from combine) differs from the "
                    "recomputed value {:.4f} (from scipy.interpolate and scipy.minimize)".format(
                        poi_min, origin, poi_min_new,
                    ),
                )
        else:
            poi_min = poi_min_new

    # helper to get the outermost intersection of the nll curve with a certain value
    def get_intersections(v):
        def minimize(bounds):
            # cap the farther bound using a simple scan
            for x in np.linspace(bounds[0], bounds[1], 50):
                if interp(x) > v * 1.05:
                    bounds[1] = x
                    break
            bounds.sort()

            # minimize
            objective = lambda x: abs(interp(x) - v)
            res = minimize_1d(objective, bounds)

            # retry once
            success = lambda: res.status == 0 and (bounds[0] < res.x[0] < bounds[1])
            if not success():
                res = minimize_1d(objective, bounds)

            return res.x[0] if success() else None

        return (
            minimize([poi_min, poi_values_max - 1e-4]),
            minimize([poi_min, poi_values_min + 1e-4]),
        )

    # get the intersections with values corresponding to 1, 2 and 3 sigma
    # (taken from solving chi2_1_cdf(x) = 1,2,3 sigma gauss intervals)
    poi_p1, poi_m1 = get_intersections(chi2_levels[1][1])
    poi_p2, poi_m2 = get_intersections(chi2_levels[1][2])
    poi_p3, poi_m3 = get_intersections(chi2_levels[1][3])

    # create a Number object wrapping the best fit value and its 1 sigma error when given
    unc = None
    if poi_p1 is not None and poi_m1 is not None:
        unc = (poi_p1 - poi_min, poi_min - poi_m1)
    num_min = Number(poi_min, unc)

    # build summary
    summary = OrderedDict([
        ("best_fit", poi_min),
        ("range", [
            [poi_m1, poi_p1],
            [poi_m2, poi_p2],
            [poi_m3, poi_p3],
        ]),
        ("uncertainty", [
            [(poi_p1 and (poi_p1 - poi_min)), (poi_m1 and (poi_m1 - poi_min))],
            [(poi_p2 and (poi_p2 - poi_min)), (poi_m2 and (poi_m2 - poi_min))],
            [(poi_p3 and (poi_p3 - poi_min)), (poi_m3 and (poi_m3 - poi_min))],
        ]),
    ])

    # print values
    def sigma_line(n, p, m):
        rnd = lambda v: "{:+.4f}".format(v)
        return "{} sigma: {} / {} ([{}, {}])".format(
            n,
            "--" if p is None else rnd(p - poi_min),
            "--" if m is None else rnd(m - poi_min),
            "--" if m is None else rnd(m),
            "--" if p is None else rnd(p),
        )
    print("best fit value{}: {:+.4f}".format(origin, poi_min))
    print("    " + sigma_line(1, poi_p1, poi_m1))
    if poi_p2 is not None or poi_m2 is not None:
        print("    " + sigma_line(2, poi_p2, poi_m2))
    if poi_p3 is not None or poi_m3 is not None:
        print("    " + sigma_line(3, poi_p3, poi_m3))

    return DotDict(
        interp=interp,
        poi_min=poi_min,
        poi_p1=poi_p1,
        poi_m1=poi_m1,
        poi_p2=poi_p2,
        poi_m2=poi_m2,
        poi_p3=poi_p3,
        poi_m3=poi_m3,
        num_min=num_min,
        summary=summary,
    )


def evaluate_likelihood_scan_2d(
    poi1_values,
    poi2_values,
    dnll2_values,
    poi1_min=None,
    poi2_min=None,
    contours=None,
):
    """
    Takes the results of a 2D likelihood profiling scan given by *poi1_values*, *poi2_values* and
    the corresponding *dnll2_values* values, performs an interpolation and returns certain results
    of the scan in a dict. The two lists of poi values should represent an expanded grid, so that
    *poi1_values*, *poi2_values* and *dnll2_values* should all be 1D with the same length. When
    *poi1_min* and *poi2_min* are *None*, they are estimated from the interpolated curve.
    When *contours* are given, it should be a nested list of graph objects, where each contained
    list represents the graphs that constitute a contour.

    Please consider preprocessing values with :py:func:`_preprocess_values` first.

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
    - ``box_nums``: A list of Number instance pairs, each one representing the two box dimensions of
      a passed  contour with uncertainties denoting edges.
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

    # remove values where dnll2 is NaN
    nan_mask = np.isnan(dnll2_values)
    mask = ~nan_mask
    poi1_values = poi1_values[mask]
    poi2_values = poi2_values[mask]
    dnll2_values = dnll2_values[mask]
    n_nans = (~mask).sum()
    if n_nans:
        warn("WARNING: found {} NaN(s) in dnll2 values".format(n_nans))

    # obtain an interpolation function
    # interp = scipy.interpolate.interp2d(poi1_values, poi2_values, dnll2_values)
    # interp = scipy.interpolate.SmoothBivariateSpline(poi1_values, poi2_values, dnll2_values,
    #     kx=2, ky=2)
    coords = np.stack([poi1_values, poi2_values], axis=1)
    try:
        interp = scipy.interpolate.CloughTocher2DInterpolator(coords, dnll2_values)
    except:
        return None

    # recompute the minimum and compare with the existing one when given
    xcheck = poi1_min is not None and poi2_min is not None
    print("extracting POI minimum {}...".format("as cross check " if xcheck else ""))
    objective = lambda x: interp(*x)
    bounds1 = (poi1_values_min + 1e-4, poi1_values_max - 1e-4)
    bounds2 = (poi2_values_min + 1e-4, poi2_values_max - 1e-4)
    res = scipy.optimize.minimize(objective, [1.0, 1.0], tol=1e-7, bounds=[bounds1, bounds2])
    if res.status != 0:
        if not xcheck:
            raise Exception("could not find minimum of nll2 interpolation: {}".format(res.message))
    else:
        poi1_min_new = res.x[0]
        poi2_min_new = res.x[1]
        print("done, found {:.4f}, {:.4f}".format(poi1_min_new, poi2_min_new))
        if xcheck:
            # compare and optionally issue a warning (threshold to be optimized)
            if abs(poi1_min - poi1_min_new) >= 0.03:
                warn(
                    "WARNING: external POI1 minimum (from combine) {:.4f} differs from the "
                    "recomputed value (from scipy.interpolate and scipy.minimize) {:.4f}".format(
                        poi1_min, poi1_min_new,
                    ),
                )
            if abs(poi2_min - poi2_min_new) >= 0.03:
                warn(
                    "WARNING: external POI2 minimum (from combine) {:.4f} differs from the "
                    "recomputed value (from scipy.interpolate and scipy.minimize) {:.4f}".format(
                        poi2_min, poi2_min_new,
                    ),
                )
        else:
            poi1_min = poi1_min_new
            poi2_min = poi2_min_new

    # helper to get the outermost intersection of the dnll2 curve with a certain value of a poi,
    # while fixing the other poi at its best fit value
    def get_central_intersections(v, n_poi):
        if n_poi == 1:
            poi_values_min, poi_values_max = poi1_values_min, poi1_values_max
            poi_min = poi1_min
            _interp = lambda x: interp(x, poi2_min)
        else:
            poi_values_min, poi_values_max = poi2_values_min, poi2_values_max
            poi_min = poi2_min
            _interp = lambda x: interp(poi1_min, x)

        def minimize(bounds):
            # cap the farther bound using a simple scan
            for x in np.linspace(bounds[0], bounds[1], 50):
                if _interp(x) > v * 1.05:
                    bounds[1] = x
                    break
            bounds.sort()

            # minimize
            objective = lambda x: abs(_interp(x) - v)
            res = minimize_1d(objective, bounds)

            # retry once
            success = lambda: res.status == 0 and (bounds[0] < res.x[0] < bounds[1])
            if not success():
                res = minimize_1d(objective, bounds)

            return res.x[0] if success() else None

        return (
            minimize([poi_min, poi_values_max - 1e-4]),
            minimize([poi_min, poi_values_min + 1e-4]),
        )

    # get the intersections with values corresponding to 1 and 2 sigma
    # (taken from solving chi2_1_cdf(x) = 1 or 2 sigma gauss intervals)
    poi1_p1, poi1_m1 = get_central_intersections(chi2_levels[2][1], 1)
    poi2_p1, poi2_m1 = get_central_intersections(chi2_levels[2][1], 2)
    poi1_p2, poi1_m2 = get_central_intersections(chi2_levels[2][2], 1)
    poi2_p2, poi2_m2 = get_central_intersections(chi2_levels[2][2], 2)

    # create Number objects wrapping the best fit values and their 1 sigma error when given
    unc1 = None
    unc2 = None
    if poi1_p1 is not None and poi1_m1 is not None:
        unc1 = (poi1_p1 - poi1_min, poi1_min - poi1_m1)
    if poi2_p1 is not None and poi2_m1 is not None:
        unc2 = (poi2_p1 - poi2_min, poi2_min - poi2_m1)
    num1_min = Number(poi1_min, unc1)
    num2_min = Number(poi2_min, unc2)

    # build contour boxes
    box_nums = None
    if contours:
        box_nums = []
        for graphs in contours:
            if not graphs:
                # when graphs is empty, store None's instead of actual Number's with uncertainties
                box_nums.append((None, None))
                continue

            box1_m, box1_p, box2_m, box2_p = get_contour_box(graphs)
            unc1, unc2 = None, None
            if abs(box1_m - poi1_values_min) > 1e-3 and abs(box1_p - poi1_values_max) > 1e-3:
                unc1 = (box1_p - poi1_min, poi1_min - box1_m)
            if abs(box2_m - poi2_values_min) > 1e-3 and abs(box2_p - poi2_values_max) > 1e-3:
                unc2 = (box2_p - poi2_min, poi2_min - box2_m)
            box_nums.append((Number(poi1_min, unc1), Number(poi2_min, unc2)))

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
        box_nums=box_nums,
    )
