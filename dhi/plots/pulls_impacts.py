# coding: utf-8

"""
Pull and impact plots using ROOT.
"""

import os
import json
import re
import math
import array

import six
import numpy as np

from dhi.config import poi_data, campaign_labels, colors
from dhi.util import import_ROOT, multi_match, to_root_latex, linspace, colored, make_list
from dhi.plots.util import use_style


colors = colors.root


@use_style("dhi_default")
def plot_pulls_impacts(
    paths,
    data,
    poi=None,
    parameters_per_page=-1,
    selected_page=-1,
    only_parameters=None,
    skip_parameters=None,
    order_parameters=None,
    order_by_impact=False,
    pull_range=2,
    impact_range=-1,
    best_fit_value=True,
    labels=None,
    pad_width=None,
    left_margin=None,
    right_margin=None,
    entry_height=None,
    label_size=None,
    pad_width=None,
    left_margin=None,
    right_margin=None,
    entry_height=None,
    campaign=None,
):
    """
    Creates a plot containing both pulls and impacts and saves it at *paths*. *data* should either
    be a path to a json file or the content of a json file in the structure provided by
    CombineHarvester's combineTool.py. For more info, see
    https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard. *poi* should be
    the name of the POI parameter the plotted quantities should be relative to. When *None* and
    *data* contains only one POI, this value is used.

    *parameters_per_page* configures how many parameters are shown per plot page. When negative, all
    parameters are shown in the same plot. This feature is only supported for pdfs. When
    *selected_page* is non-negative, only this page is created. *only_parameters*
    (*skip_parameters*) can be a list of name patterns or files containing name patterns
    line-by-line to include (exclude) parameters from plotting. *order_parameters* accepts the same
    type of values, except they are used to order parameters. When *order_by_impact* is *True*,
    *order_parameters* is neglected and the order is derived using the largest absolute impact.

    The symmetric range of pulls on the bottom x-axis is defined by *pull_range* (an integer)
    whereas the range of the top x-axis associated to impact values is set by *impact_range*. For
    the purpose of visual clearness, the tick marks on both axes are identical, so one needs to make
    sure that *pull_range* and *impact_range* match. When *impact_range* is not positive, an
    automatic value is chosen per page.

    *best_fit_value* can be a 3-tuple (central value, unsigned +error, unsigned -error) that is
    shown as a text at the top of the plot. When just *True*, it is extracted from *data* which
    contains the value following combine's NLL interpolation. *labels* should
    be a dictionary or a json file containing a dictionary that maps nuisances names to labels shown
    in the plot. When a key starts with "^" and ends with "$" it is interpreted as a regular
    expression. Matched groups can be reused in the substituted name via '\n' to reference the n-th
    group (following the common re.sub format). *pad_width*, *left_margin*, *right_margin*,
    *entry_height* and *label_size* can be set to a size in pixels to overwrite internal defaults.
    *campaign* should refer to the name of a campaign label defined in dhi.config.campaign_labels.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/pullsandimpacts.html
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    pull_range = int(pull_range)
    auto_impact_range = impact_range <= 0

    # read data when a path is given
    if isinstance(data, six.string_types):
        data = os.path.expandvars(os.path.expanduser(data))
        with open(data, "r") as f:
            data = json.load(f)

    # check or get the poi
    if poi is None:
        if len(data["POIs"]) != 1:
            raise Exception("when poi is None, data must contain exactly one POI, found {}".format(
                len(data["POIs"])))
        poi = str(data["POIs"][0]["name"])
        print("selected POI '{}' from data".format(poi))
    else:
        all_pois = [d["name"] for d in data["POIs"]]
        if poi not in all_pois:
            raise Exception("requested POI '{}' not found in data, available POIs are {}".format(
                poi, ",".join(all_pois)))

    # create Parameter objects
    params = [Parameter(d, poi) for d in data["params"]]
    print("{} total parameters found in data".format(len(params)))

    # apply filtering
    if only_parameters:
        patterns = read_patterns(only_parameters)
        params = [param for param in params if multi_match(param.name, patterns, mode=any)]
    if skip_parameters:
        patterns = read_patterns(skip_parameters)
        params = [param for param in params if not multi_match(param.name, patterns, mode=any)]
    if only_parameters or skip_parameters:
        print("{} remaining parameters after filtering".format(len(params)))

    # apply ordering
    params.sort(key=lambda param: param.name)
    if order_by_impact:
        # failures to np.inf
        v = lambda x: np.inf if np.isnan(x) else x
        params.sort(key=lambda param: -v(param.max_impact))
    elif order_parameters:
        patterns = read_patterns(order_parameters) + ["*"]
        indices = []
        for pattern in patterns:
            for i, param in enumerate(params):
                if i in indices:
                    continue
                elif multi_match(param.name, pattern):
                    indices.append(i)
        params = [params[i] for i in indices]

    # prepare labels
    if isinstance(labels, six.string_types):
        labels = os.path.expandvars(os.path.expanduser(labels))
        with open(labels, "r") as f:
            labels = json.load(f)
    elif not labels:
        labels = {}

    # expand regular expressions through eager interpolation using parameter names
    for k, v in labels.items():
        if not k.startswith("^") or not k.endswith("$"):
            continue
        for param in params:
            # skip explicit translations, effectively giving them priority
            if param.name in labels:
                continue
            # apply the pattern
            new_name = re.sub(k, v, param.name)
            # store a translation label when the name has changed
            if new_name != param.name:
                labels[param.name] = new_name

    # determine the number of pages
    if parameters_per_page < 1:
        parameters_per_page = len(params)
    n_pages = int(math.ceil(float(len(params)) / parameters_per_page))

    # some constants for plotting
    pad_width = pad_width or 1000  # pixels
    top_margin = 70  # pixels
    bottom_margin = 70  # pixels
    left_margin = left_margin or 250  # pixels
    right_margin = right_margin or 20  # pixels
    entry_height = entry_height or 30  # pixels
    head_space = 130  # pixels
    x_min, x_max = -pull_range, pull_range

    # plot per page
    for page in range(n_pages):
        # when set, only plot the selected page
        if selected_page >= 0 and selected_page != page:
            continue

        _params = params[page * parameters_per_page:(page + 1) * parameters_per_page]
        n = len(_params)

        # get the pad height
        pad_height = n * entry_height + head_space + top_margin + bottom_margin

        # get relative pad margins
        pad_margins = {
            "TopMargin": float(top_margin) / pad_height,
            "BottomMargin": float(bottom_margin) / pad_height,
            "LeftMargin": float(left_margin) / pad_width,
            "RightMargin": float(right_margin) / pad_width,
        }

        # get the y maximum
        y_max = (pad_height - top_margin - bottom_margin) / float(entry_height)

        # define the impact_range and derive the x_ratio
        if auto_impact_range:
            # get the maximum absolute impact on this page
            max_impact = max(max(abs(v) if np.isfinite(v) else 0.0 for v in p.impact) for p in _params)
            # determine possible ratios and pick the one that is slightly larger (using a threshold)
            ratios = sum(([round(f * 10**x, 4) for f in range(1, 6)] for x in range(-4, 4)), [])
            for ratio in ratios:
                _impact_range = pull_range * ratio
                if _impact_range * 0.85 > max_impact:
                    impact_range = _impact_range
                    break
            else:
                impact_range = 1.
                print("{}: could not determine an automatic impact range for maximum impact {} and "
                    "pull_range {}, using 1 instead".format(
                        colored("WARNING", "red"), max_impact, pull_range
                    )
                )
        else:
            # warn when there is a mismatch between pull and impact ranges
            # which will not lead to a wrong plot, but misplaced impact ticks on the top x-axis
            # definition: there is a mismatch if "impact_range / pull_range" has too many decimal
            # digits, which is especially true for irrational numbers
            n_frac = len(str(float(impact_range) / pull_range).split(".", 1)[1])
            if n_frac > 3:
                print("{}: the upper impact_range of {} does not seem to match the lower "
                    "pull_range of {}, which will lead to a mismatch between numbers and ticks of "
                    "the top x-axis or longish decimal numbers".format(
                        colored("WARNING", "red"), impact_range, pull_range
                    )
                )
        x_ratio = float(impact_range) / pull_range

        # setup the default style and create canvas and pad
        r.setup_style()
        canvas, (pad,) = r.routines.create_canvas(width=pad_width, height=pad_height,
            pad_props=pad_margins)
        pad.cd()
        draw_objs = []

        # dummy histogram to control axes
        x_title = "Pull (#theta_{post} - #theta_{pre}) / #Delta#theta"
        h_dummy = ROOT.TH1F("dummy", ";{};".format(x_title), 1, x_min, x_max)
        r.setup_hist(h_dummy, props={"LineWidth": 0, "Maximum": y_max}, pad=pad)
        x_axis = h_dummy.GetXaxis()
        n_div = (800 if pull_range <= 3 else 400) + 2 * pull_range
        r.setup_x_axis(x_axis, pad, props={"Ndivisions": -n_div, "TitleOffset": 1.3,
            "LabelOffset": r.pixel_to_coord(canvas, y=4)})
        draw_objs.append((h_dummy, "HIST"))

        # draw a second axis at the top denoting impact values
        x2_axis = ROOT.TGaxis(x_min, y_max, x_max, y_max, x_min * x_ratio, x_max * x_ratio,
            n_div, "SN")
        x2_title = "Impact #Delta" + to_root_latex(poi_data[poi].label)
        r.setup_x_axis(x2_axis, pad, props={"Title": x2_title, "TickLength": 0.,
            "LabelOffset": r.pixel_to_coord(canvas, y=-24), "TitleOffset": -1.2})
        draw_objs.append(x2_axis)

        # y axis labels and ticks
        h_dummy.GetYaxis().SetBinLabel(1, "")
        label_size = label_size or 20
        for i, param in enumerate(_params):
            # parameter labels
            label = to_root_latex(labels.get(param.name, param.name))
            if param.invalid:
                label = "#bf{{{label}}}".format(label=label)
            label = ROOT.TLatex(x_min - (x_max - x_min) * 0.0125, n - i - 0.5, label)
            r.setup_latex(label, props={"NDC": False, "TextAlign": 32, "TextSize": label_size})
            draw_objs.append(label)

            # left and right ticks
            tick_length = (x_max - x_min) * 0.0375
            tl = ROOT.TLine(x_min, i + 1, x_min + tick_length, i + 1)
            tr = ROOT.TLine(x_max, i + 1, x_max - tick_length, i + 1)
            r.setup_line(tl, props={"NDC": False})
            r.setup_line(tr, props={"NDC": False})
            draw_objs.extend([tl, tr])

        # impact graphs
        g_impact_hi = ROOT.TGraphAsymmErrors(n)
        g_impact_lo = ROOT.TGraphAsymmErrors(n)
        g_impact_overlap = ROOT.TGraphAsymmErrors(n)
        for i, param in enumerate(_params):
            idx = n - i
            # use x scale to transform values given in x axis 2 to x axis 1
            u, d = param.impact[1] / x_ratio, param.impact[0] / x_ratio
            # place points always on zero, set errors to act as impact values
            g_impact_hi.SetPoint(idx, 0, idx - 0.5)
            g_impact_lo.SetPoint(idx, 0, idx - 0.5)
            g_impact_hi.SetPointError(idx, 0 if u > 0 else -u, u if u > 0 else 0, 0.5, 0.5)
            g_impact_lo.SetPointError(idx, -d if d < 0 else 0, 0 if d < 0 else d, 0.5, 0.5)
            # fill overlap graph with up values in case of equal signs and larger down impact
            if u * d > 0 and abs(d) > abs(u):
                g_impact_overlap.SetPoint(idx, 0, idx - 0.5)
                g_impact_overlap.SetPointError(idx, 0 if u > 0 else -u, u if u > 0 else 0, 0.5, 0.5)
        r.setup_graph(g_impact_hi, color=colors.red_cream, color_flags="lmf")
        r.setup_graph(g_impact_lo, color=colors.blue_cream, color_flags="lmf")
        r.setup_graph(g_impact_overlap, color=colors.red_cream, color_flags="lmf")
        draw_objs.append((g_impact_hi, "2"))
        draw_objs.append((g_impact_lo, "2"))
        draw_objs.append((g_impact_overlap, "2"))

        # vertical, dashed lines at all integer and half-integer values
        for x in linspace(x_min, x_max, 2 * (x_max - x_min) + 1):
            if not (x_min < x < x_max):
                continue
            l = ROOT.TLine(x, 0, x, y_max)
            c = colors.light_grey if x % 1. else colors.dark_grey
            r.setup_line(l, props={"NDC": False, "LineWidth": 1, "LineColor": c, "LineStyle": 3})
            draw_objs.append(l)

        # pull graph
        arr = lambda vals: array.array("f", vals)
        g_pull = ROOT.TGraphAsymmErrors(n,
            arr([(-1e5 if param.is_rate_param else param.pull[1]) for param in _params]),
            arr([n - i - 0.5 for i in range(n)]),
            arr([-param.pull[0] for param in _params]),
            arr([param.pull[2] for param in _params]),
            arr(n * [0.]),
            arr(n * [0.]),
        )
        r.setup_graph(g_pull, props={"MarkerStyle": 20, "MarkerSize": 1.2, "LineWidth": 1})
        draw_objs.append((g_pull, "PEZ"))

        # plain post-fit intervals as texts for rateParam's
        rate_label_tmpl = "%.2f^{ +%.2f}_{ -%.2f}"
        for i, param in enumerate(_params):
            if param.is_rate_param:
                down, nominal, up = param.postfit
                rate_label = rate_label_tmpl % (nominal, up - nominal, nominal - down)
                rate_label = ROOT.TLatex(0, n - i - 0.5, rate_label)
                r.setup_latex(rate_label, props={"NDC": False, "TextAlign": 22, "TextSize": 16})
                draw_objs.append(rate_label)
            if param.invalid:
                rate_label = ROOT.TLatex(0, n - i - 0.5, "#bf{Invalid - Failed Fit}")
                r.setup_latex(rate_label, props={"NDC": False, "TextAlign": 22, "TextSize": label_size})
                draw_objs.append(rate_label)

        # legend
        legend = r.routines.create_legend(pad=pad, width=170, n=3)
        r.setup_legend(legend)
        legend.AddEntry(g_impact_hi, "Impact +1 #sigma", "F")
        legend.AddEntry(g_impact_lo, "Impact -1 #sigma", "F")
        legend.AddEntry(g_pull, "Pull", "LP")
        draw_objs.append(legend)
        legend_box = r.routines.create_legend_box(legend, pad, "tlr", y1_padding=-10,
            props={"LineWidth": 0, "FillColor": colors.white_trans_70})
        draw_objs.insert(-1, legend_box)

        # best fit value label
        if best_fit_value is True:
            for p in data["POIs"]:
                if p["name"] == poi:
                    d, n, u = p["fit"]
                    best_fit_value = (n, u - n, n - d)
                    break
        if best_fit_value:
            fit_label = "{} = {:.2f}^{{ +{:.2f}}}_{{ -{:.2f}}}".format(
                to_root_latex(poi_data[poi].label), *best_fit_value)
            fit_label = r.routines.create_top_left_label(fit_label, pad=pad, y_offset=80,
                x=0.5 * (r.get_x(0, pad, "right") + r.get_x(0, pad)), props={"TextAlign": 21})
            draw_objs.append(fit_label)

        # cms label
        cms_labels = r.routines.create_cms_labels(pad=pad, x_offset=10, y_offset=50)
        draw_objs.extend(cms_labels)

        # campaign label
        if campaign:
            campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
            campaign_label = r.routines.create_top_left_label(campaign_label, pad=pad, x_offset=10,
                y_offset=80)
            draw_objs.append(campaign_label)

        # draw objects
        r.routines.draw_objects(draw_objs)

        # update and save
        # when there is more than one page, use roots "logic" to write multiple pages
        r.update_canvas(canvas)
        for path in make_list(paths):
            if selected_page < 0 and n_pages > 1 and path.endswith(".pdf"):
                flag = {0: "(", n_pages - 1: ")"}.get(page, "")
                canvas.Print(path + flag)
            else:
                canvas.SaveAs(path)


class Parameter(object):
    """
    Lightweight helper class that wraps a "param" entry in the json input data for a given poi and
    provides easy access to quantities.
    """

    def __init__(self, data, poi):
        super(Parameter, self).__init__()

        # store plain entries
        self.name = data["name"]
        self.poi = data[poi]
        self.prefit = data["prefit"]
        self.postfit = data["fit"]
        self.max_impact = data["impact_" + poi]
        self.type = data["type"]
        self.groups = data["groups"]
        self.invalid = np.any(np.isnan(self.poi))

        # compute two sided impacts, store as [low, high] preserving signs
        self.impact = [
            self.poi[0] - self.poi[1],
            self.poi[2] - self.poi[1],
        ]

        # compute relative pulls (following logic in plotImpacts.py)
        def pull(i):
            abs_pull = self.postfit[i] - self.prefit[1]
            # normalize
            if abs_pull >= 0:
                return abs_pull / (self.prefit[2] - self.prefit[1])
            else:
                return abs_pull / (self.prefit[1] - self.prefit[0])

        # store as [low, central, high] preserving signs
        self.pull = [pull(0) - pull(1), pull(1), pull(2) - pull(1)]

        # whether or not this parameter comes from autoMCStats
        self.is_mc_stats = self.name.startswith("prop_bin")

        # whether or not this parameter comes from a rateParam
        self.is_rate_param = self.type == "Unrecognised" and tuple(self.prefit) == (2.0, 1.0, 0.0)


def read_patterns(patterns):
    _patterns = []
    for pattern in patterns:
        path = os.path.expandvars(os.path.expanduser(pattern))
        if os.path.isfile(path):
            with open(path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            _patterns.extend(lines)
        else:
            _patterns.append(pattern)
    return _patterns
