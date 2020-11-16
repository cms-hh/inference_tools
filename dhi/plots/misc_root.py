# coding: utf-8

"""
Miscellaneous result plots using ROOT.
"""

import numpy as np
import scipy.interpolate

from dhi.config import poi_data, campaign_labels, colors as _colors, br_hh_names
from dhi.util import import_ROOT, DotDict, to_root_latex, create_tgraph, minimize_1d
from dhi.plots.likelihoods_mpl import evaluate_likelihood_scan_1d


def plot_bestfit_and_exclusion(
    path,
    data,
    poi,
    x_min=None,
    x_max=None,
    campaign="2017",
):
    """
    Creates a plot showing the best fit values as well as exluded regions of a *poi* for multiple
    analysis (or channels) and saves it at *path*. *data* should be a list of dictionaries with
    fields "expected_limits" and "expected_nll", and optionally *observed_limits*, *observed_nll*
    and "name" (shown on the y-axis). The former four should be given as either dictionaries or
    numpy record arrays containing fields *poi* and "limit", or *poi* and "dnll2". When the name is
    a key of dhi.config.br_hh_names, its value is used as a label instead. Example:

    .. code-block:: python

        plot_bestfit_and_exclusion(
            path="plot.pdf",
            data=[{
                "expected_limits": {"kl": [...], "limit": [...]},
                "expected_nll": {"kl": [...], "dnll2": [...]},
            }, {
                ...
            }],
            poi="kl",
        )

    *x_min* and *x_max* define the range of the x-axis and default to the maximum range of poi
    values passed in data. *campaign* should refer to the name of a campaign label defined in
    dhi.config.campaign_labels.

    Example: http://cms-hh.web.cern.ch/cms-hh/tools/inference/tasks/misc.html#plotbestfitandexclusion
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # check minimal fields per data entry
    assert(all("name" in d for d in data))
    assert(all("expected_limits" in d for d in data))
    assert(all("expected_nll" in d for d in data))
    n = len(data)
    has_obs = any("observed_limits" in d for d in data)
    poi_values = np.array(data[0]["expected_limits"][poi])

    # set default ranges
    if x_min is None:
        x_min = min(poi_values)
    if x_max is None:
        x_max = max(poi_values)

    # some constants for plotting
    canvas_width = 800  # pixels
    top_margin = 35  # pixels
    bottom_margin = 70  # pixels
    left_margin = 150  # pixels
    entry_height = 90  # pixels
    head_space = 100  # pixels

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
    poi_label = to_root_latex(poi_data[poi].label)
    h_dummy = ROOT.TH1F("dummy", ";{};".format(poi_label), 1, x_min, x_max)
    r.setup_hist(h_dummy, pad=pad, props={"LineWidth": 0, "Maximum": y_max})
    draw_objs.append((h_dummy, "HIST"))

    # expected exclusion area from intersections of limit with 1
    def create_exclusion_graph(data_key):
        excl_x, excl_y, excl_d, excl_u = [], [], [], []
        for i, d in enumerate(data):
            if data_key not in d:
                continue
            ranges = evaluate_limit_scan_1d(poi_values, d[data_key]["limit"]).excluded_ranges
            for start, stop in ranges:
                is_left = start < 1 and stop < 1
                excl_x.append(stop if is_left else start)
                excl_y.append(n - i - 0.5)
                excl_d.append((stop - start) if is_left else 0)
                excl_u.append(0 if is_left else (stop - start))
        return create_tgraph(len(excl_x), excl_x, excl_y, excl_d, excl_u, 0.5, 0.5)

    # expected
    g_excl_exp = create_exclusion_graph("expected_limits")
    r.setup_graph(g_excl_exp, color=_colors.root.black, color_flags="f",
        props={"FillStyle": 3345, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0})
    draw_objs.append((g_excl_exp, "SAME,2"))
    legend_entries.append((g_excl_exp, "Excluded (expected)"))

    # observed
    if has_obs:
        g_excl_obs = create_exclusion_graph("observed_limits")
        r.setup_graph(g_excl_obs, color=_colors.root.red, color_flags="f",
            props={"FillStyle": 3354, "MarkerStyle": 20, "MarkerSize": 0, "LineWidth": 0})
        draw_objs.append((g_excl_obs, "SAME,2"))
        legend_entries.append((g_excl_obs, "Excluded (observed)"))
    else:
        # dummy legend entry
        legend_entries.append((h_dummy, " ", ""))

    # best fit values
    scans = [
        evaluate_likelihood_scan_1d(d["expected_nll"][poi], d["expected_nll"]["dnll2"],
            poi_min=d.get("poi_min"))
        for d in data
    ]
    g_bestfit = create_tgraph(n,
        [scan.num_min() for scan in scans],
        [n - i - 0.5 for i in range(n)],
        [scan.num_min.u(direction="down") for scan in scans],
        [scan.num_min.u(direction="up") for scan in scans],
        0,
        0,
    )
    r.setup_graph(g_bestfit, props={"MarkerStyle": 20, "MarkerSize": 1.2, "LineWidth": 1})
    draw_objs.append((g_bestfit, "PEZ"))
    legend_entries.append((g_bestfit, "Best fit value"))

    # vertical line at 1
    if x_min < 1:
        line_one = ROOT.TLine(1., 0., 1., n)
        r.setup_line(line_one, props={"NDC": False, "LineStyle": 7}, color=_colors.root.black)
        draw_objs.insert(-1, line_one)
        legend_entries.append((line_one, "SM prediction", "l"))

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
        label = label_tmpl % (label, poi_label, scan.num_min.str("%.1f", style="root"))
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
    legend = r.routines.create_legend(pad=pad, width=500, height=70,
        x2=r.get_x(-45, pad, anchor="right"), y2=r.get_y(16, pad, anchor="top"))
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


def evaluate_limit_scan_1d(poi_values, limit_values):
    """
    Takes the results of an upper limit scan given by the *poi_values* and the corresponding *limit*
    values, performs an interpolation and returns certain results of the scan in a dict.

    The returned fields are:

    - ``interp``: The generated interpolation function.
    - ``excluded_ranges``: A list of 2-tuples denoting ranges in units of the poi where limits are
      below one.
    """
    poi_values = np.array(poi_values)
    limit_values = np.array(limit_values)

    # first, obtain an interpolation function
    mask = ~np.isnan(limit_values)
    poi_values = poi_values[mask]
    limit_values = limit_values[mask]
    interp = scipy.interpolate.interp1d(poi_values, limit_values, kind="cubic")

    # interpolation bounds
    bounds = (poi_values.min() + 1e-4, poi_values.max() - 1e-4)

    # helper to find intersections with one given a starting point
    def get_intersection(start):
        objective = lambda x: (interp(x) - 1) ** 2.0
        res = minimize_1d(objective, bounds, start=start)
        return res.x[0] if res.status == 0 and (bounds[0] < res.x[0] < bounds[1]) else None

    # get exclusion range edges from intersections and remove duplicates
    rnd = lambda v: round(v, 3)
    edges = [rnd(poi_values.min()), rnd(poi_values.max())]
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
