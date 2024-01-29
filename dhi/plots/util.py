# coding: utf-8

"""
Different helpers and ROOT style configurations to be used with plotlib.
"""

import os
import re
import math
import array
import uuid
import json
import functools
import itertools
import contextlib
import ctypes
from collections import OrderedDict

import six
import numpy as np
import scipy.interpolate
import law

from dhi.config import poi_data, br_hh_names
from dhi.util import (
    import_ROOT, import_file, try_int, to_root_latex, make_list, make_tuple, InterExtrapolator,
    GridDataInterpolator, DotDict, round_scientific,
)


_styles = {}


def _setup_styles():
    global _styles
    if _styles:
        return

    import plotlib.root as r

    # dhi_default
    s = _styles["dhi_default"] = r.styles.copy("default", "dhi_default")
    s.legend_y2 = -15
    s.legend_dy = 32
    s.legend.TextSize = 20
    s.legend.FillStyle = 1
    s.x_axis.SetDecimals = True
    s.y_axis.SetDecimals = True
    s.z_axis.SetDecimals = True
    s.style.PaintTextFormat = "1.2f"


def use_style(style_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import plotlib.root as r

            # setup styles
            _setup_styles()

            # invoke the wrapped function in that style
            with r.styles.use(style_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class Style(DotDict):

    @classmethod
    def new(cls, style, *args, **kwargs):
        # takes a string or tuple of strings and returns a new style object
        inst = cls(*args, **kwargs)
        inst.styles = make_tuple(style)
        return inst

    def __init__(self, *args, **kwargs):
        super(Style, self).__init__(*args, **kwargs)

        self.styles = ()

    def __eq__(self, other):
        if isinstance(other, (str, list, tuple)):
            return self.matches(other)

        super(Style, self).__eq__(other)

    def __ne__(self, other):
        if isinstance(other, (str, list, tuple)):
            return not self.matches(other)

        super(Style, self).__ne__(other)

    def matches(self, pattern):
        return any(law.util.multi_match(style, pattern) for style in self.styles)


def create_random_name():
    return str(uuid.uuid4())


def create_model_parameters(
    model_parameters,
    pad,
    grouped=False,
    x_offset=25,
    y_offset=40,
    dy=24,
    props=None,
):
    """
    Creates a list of ``ROOT.TLatex`` objects for *model_parameters*, properly positioned for *pad*
    with options to change the offsets *x_offset* and *y_offset*, the vertical distance between
    labels *dy*, and optional properties *props*.

    *model_parameters* should be a dictionary mapping one or multiple parameter names (in a tuple)
    to values. When multiple names are given, they are contained in the same label using the format
    ``"p1 = p2 = ... = value"``. When *grouped* is *True*, groups are created over all
    *model_parameters* with same values.
    """
    import plotlib.root as r

    # handle grouping
    if grouped:
        # assign to groups
        groups = OrderedDict()
        for names, value in model_parameters.items():
            for name in make_list(names):
                groups.setdefault(value, []).append(name)

        # fill back to model parameters
        model_parameters = OrderedDict((tuple(names), value) for value, names in groups.items())

    # create labels
    parameter_labels = []

    for i, (names, value) in enumerate(model_parameters.items()):
        # each parameter key can be a list
        labels = [
            to_root_latex(poi_data.get(name, {}).get("label", name))
            for name in make_list(names)
        ]
        label = "{} = {}".format(" = ".join(labels), try_int(value))
        label = r.routines.create_top_left_label(
            label,
            pad=pad,
            props=props,
            x_offset=x_offset,
            y_offset=y_offset + i * dy,
        )
        parameter_labels.append(label)

    return parameter_labels


def create_hh_process_label(poi="r", prefix=r"pp $\rightarrow$ "):
    # please note the possible ambiguity in the process between r and r_gghh, and consider using
    # sth like "HH (incl.)" for r (however, this was recently discouraged)
    proc = {
        "r": "HH",
        "r_gghh": "HH",
        "r_qqhh": "qqHH",
        "r_vhh": "VHH",
        "r_xhh": r"X $\rightarrow$ HH",
    }.get(poi, "HH")
    return prefix + proc


def create_hh_br_label(br):
    if not br or br not in br_hh_names:
        return ""
    return "B({})".format(br_hh_names[br])


def create_hh_xsbr_label(poi="r", br=None):
    br_label = create_hh_br_label(br)
    br_label = (" x " + br_label) if br_label else ""
    return r"$\sigma$({}){}".format(create_hh_process_label(poi), br_label)


def expand_hh_channel_label(name, to_root=True, allow_prefix=True):
    prefix = ""
    if allow_prefix:
        m = re.match(r"^(\+|-)(.+)$", name)
        if m:
            prefix, name = m.groups()
            prefix += " "

    label = prefix + br_hh_names.get(name, name)

    if to_root:
        label = to_root_latex(label)

    return label


def determine_limit_digits(limit, is_xsec=False):
    digits = 0
    if is_xsec:
        if limit < 10:
            digits = 2
        elif limit < 200:
            digits = 1
    else:
        if limit < 10:
            digits = 1
    return digits


def make_parameter_label_map(parameter_names, labels=None):
    # prepare labels
    if isinstance(labels, six.string_types):
        labels = os.path.expandvars(os.path.expanduser(labels))
        # try to load a renaming function called "rename_nuisance"
        if labels.endswith(".py"):
            labels = import_file(labels, attr="rename_nuisance")
            if not callable(labels):
                raise Exception("rename_nuisance loaded from {} is not callable".format(labels))
        else:
            with open(labels, "r") as f:
                labels = json.load(f)
    elif not labels:
        labels = {}

    if not isinstance(labels, dict):
        # labels is a renaming function, call it for all parameters and store the result when
        # names changed
        _labels = {}
        for name in parameter_names:
            new_name = labels(name)
            if new_name:
                _labels[name] = new_name
        labels = _labels
    else:
        # expand regular expressions through eager interpolation using parameter names
        for k, v in labels.items():
            if not k.startswith("^") or not k.endswith("$"):
                continue
            for name in parameter_names:
                # skip explicit translations, effectively giving them priority
                if name in labels:
                    continue
                # apply the pattern
                new_name = re.sub(k, v, name)
                # store a translation label when set
                if new_name:
                    labels[name] = new_name

    return labels


def get_y_range(
    y_min_value,
    y_max_value,
    y_min=None,
    y_max=None,
    log=False,
    y_min_log=1e-3,
    top_margin=0.38,
    visible_margin=0.4,
):
    if log:
        if y_min is None:
            y_min = (0.75 * y_min_value) if y_min_value > 0 else y_min_log
        if y_max is None:
            y_max = y_min * 10**(math.log10(y_max_value / y_min) * (1.0 + top_margin))
        y_max_vis = y_min * 10**(math.log10(y_max / y_min) / (1.0 + visible_margin))
    else:
        if y_min is None:
            y_min = 0.0 if y_min_value is None else y_min_value
        if y_max is None:
            y_max = y_min + (y_max_value - y_min) * (1.0 + top_margin)
        y_max_vis = y_max / (1.0 + visible_margin) + y_min

    return y_min, y_max, y_max_vis


def frame_histogram(hist, x_width, y_width, mode="edge", frame_value=None, contour_level=None):
    # when the mode is "contour-", edge values below the level are set to a higher value which
    # effectively closes contour areas that are below (thus the "-") the contour level
    # when the mode is "contour++", the opposite happens to close contour areas above the level
    assert mode in ["edge", "constant", "contour+", "contour-"]
    if mode == "constant":
        assert frame_value is not None
    elif mode in ["contour+", "contour-"]:
        assert contour_level is not None

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

    # update frame values in contour mode
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
    hist_padded = hist.__class__(create_random_name(), "", len(x_edges) - 1,
        array.array("d", x_edges), len(y_edges) - 1, array.array("d", y_edges))
    hist_padded.SetDirectory(0)
    for by, _data in enumerate(data):
        for bx, z in enumerate(_data):
            hist_padded.SetBinContent(bx + 1, by + 1, z)

    return hist_padded


# helper to fill each bin in a 2D histogram from potentially sparse points via interpolation
def fill_hist_from_points(
    h,
    x_values,
    y_values,
    z_values,
    z_min=None,
    z_max=None,
    replace_nan=None,
    interpolation="tgraph2d",
):
    ROOT = import_ROOT()

    # remove or replace nans in z_values
    z_values = np.array(z_values)
    nan_indices = np.argwhere(np.isnan(z_values))
    if replace_nan is None:
        x_values = [x for i, x in enumerate(x_values) if i not in nan_indices]
        y_values = [y for i, y in enumerate(y_values) if i not in nan_indices]
        z_values = [z for i, z in enumerate(z_values) if i not in nan_indices]
    else:
        z_values[nan_indices] = replace_nan

    # check if the grid is even
    def values_even(values):
        values = sorted(set(values))
        diffs = set(round(b - a, 5) for a, b in zip(values[:-1], values[1:]))
        return len(diffs) == 1

    even_grid = values_even(x_values) and values_even(y_values)

    # create an interpolation function
    interp_args = ()
    if isinstance(interpolation, (list, tuple)):
        interpolation, interp_args = interpolation[0], interpolation[1:]
    if interpolation == "tgraph2d":
        g = ROOT.TGraph2D(len(z_values))
        for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
            g.SetPoint(i, x, y, z)
        interp = lambda x, y: g.Interpolate(x, y)
    elif interpolation in ("linear", "cubic"):
        if even_grid:
            interp = InterExtrapolator(
                x_values,
                y_values,
                z_values,
                kind2d=interpolation,
                kind1d=interpolation,
            )
        else:
            # prepare ad-hoc interpolation points as required by scipy's griddata
            interp_points = []
            if isinstance(h, ROOT.TH2Poly):
                for poly_bin in h.GetBins():
                    interp_points.append(list(find_poly_bin_center(poly_bin)))
            else:
                # strictly rectangular bins
                for bx in range(1, h.GetNbinsX() + 1):
                    for by in range(1, h.GetNbinsY() + 1):
                        x = h.GetXaxis().GetBinCenter(bx)
                        y = h.GetYaxis().GetBinCenter(by)
                        interp_points.append([x, y])
            interp = GridDataInterpolator(
                x_values,
                y_values,
                z_values,
                interp_points,
                kind=interpolation,
            )
    elif interpolation == "rbf":
        # parse arguments in order
        spec = [("function", str), ("smooth", float), ("epsilon", float)]
        rbf_args = {"norm": "seuclidean"}
        for val, (name, _type) in zip(interp_args, spec):
            try:
                rbf_args[name] = _type(val)
            except:
                print("WARNING: cannot parse value {} for rbf argument {} to {}".format(
                    val, name, _type,
                ))
        interp = scipy.interpolate.Rbf(x_values, y_values, z_values, **rbf_args)
    else:
        raise ValueError("unknown interpolation method '{}'".format(interpolation))

    # helper for limiting z values
    def cap_z(z):
        if z_min is not None and z < z_min:
            return z_min * (1 + 1e-5)
        if z_max is not None and z > z_max:
            return z_max * (1 - 1e-5)
        return z

    # then, fill histogram bins
    if isinstance(h, ROOT.TH2Poly):
        for poly_bin in h.GetBins():
            z = interp(*find_poly_bin_center(poly_bin))
            poly_bin.SetContent(cap_z(z))
    else:
        # strictly rectangular bins
        for bx in range(1, h.GetNbinsX() + 1):
            for by in range(1, h.GetNbinsY() + 1):
                x = h.GetXaxis().GetBinCenter(bx)
                y = h.GetYaxis().GetBinCenter(by)
                h.SetBinContent(bx, by, cap_z(interp(x, y)))


def find_poly_bin_center(poly_bin, n=1000):
    # get an initial guess
    x = poly_bin.GetXMin() + 0.5 * (poly_bin.GetXMax() - poly_bin.GetXMin())
    y = poly_bin.GetYMin() + 0.5 * (poly_bin.GetYMax() - poly_bin.GetYMin())

    # vary until the point is inside the bin
    for _ in range(n):
        if poly_bin.IsInside(x, y):
            return x, y
        raise NotImplementedError("center determination of complex poly bins not implemented yet")

    raise Exception("could not determine poly bin center after {} iterations".format(n))


def infer_binning_from_grid(x_values, y_values):
    # get the smallest difference between two points in each direction and call it bin width
    x_values = np.array(x_values, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.float32)
    ex = np.unique(x_values)
    ey = np.unique(y_values)
    x_width = min(ex[1:] - ex[:-1])
    y_width = min(ey[1:] - ey[:-1])

    # get axis limits
    x_min = min(ex) - 0.5 * x_width
    x_max = max(ex) + 0.5 * x_width
    y_min = min(ey) - 0.5 * y_width
    y_max = max(ey) + 0.5 * y_width

    # infer the number of bins
    x_bins = (x_max - x_min) / x_width
    y_bins = (y_max - y_min) / y_width
    if round_scientific(x_bins, 3) != int(x_bins):
        raise Exception("x axis range [{:3f},{:3f}) cannot be evenly split by bin width {}".format(
            x_min, x_max, x_width,
        ))
    if round_scientific(y_bins, 3) != int(y_bins):
        raise Exception("y axis range [{:3f},{:3f}) cannot be evenly split by bin width {}".format(
            y_min, y_max, y_width,
        ))
    x_bins = int(x_bins)
    y_bins = int(y_bins)

    return x_width, y_width, x_bins, y_bins, x_min, x_max, y_min, y_max


# helper to extract contours
def get_contours(
    x_values,
    y_values,
    z_values,
    levels,
    frame_kwargs=None,
    min_points=10,
    **kwargs  # noqa
):
    ROOT = import_ROOT()

    if frame_kwargs is None:
        frame_kwargs = [{"mode": "edge"}]
    elif not isinstance(frame_kwargs, (list, tuple)):
        frame_kwargs = [frame_kwargs]

    # remove nans in z_values
    nan_indices = np.argwhere(np.isnan(np.array(z_values)))
    x_values = [x for i, x in enumerate(x_values) if i not in nan_indices]
    y_values = [y for i, y in enumerate(y_values) if i not in nan_indices]
    z_values = [z for i, z in enumerate(z_values) if i not in nan_indices]

    # when there are no values left, return empty lists
    if not x_values:
        return [[] for _ in range(len(levels))]

    # to extract contours, we need a 2D histogram with optimized bin widths, edges and padding
    def get_diffs(values):
        values = sorted(set(values))
        diffs = [(values[i + 1] - values[i]) for i in range(len(values) - 1)]
        return min(diffs), max(diffs)

    # x axis
    x_min = min(x_values)
    x_max = max(x_values)
    x_width_min, x_width_max = get_diffs(x_values)
    x_n = (x_max - x_min) / x_width_min
    x_n = int(x_n + 1) if try_int(x_n) else int(math.ceil(x_n))
    x_min -= 0.5 * x_width_min
    x_max += 0.5 * x_width_min

    # y axis
    y_min = min(y_values)
    y_max = max(y_values)
    y_width_min, y_width_max = get_diffs(y_values)
    y_n = (y_max - y_min) / y_width_min
    y_n = int(y_n + 1) if try_int(y_n) else int(math.ceil(y_n))
    y_min -= 0.5 * y_width_min
    y_max += 0.5 * y_width_min

    # create and fill a hist
    with temporary_canvas() as c:
        c.cd()
        h = ROOT.TH2F(create_random_name(), "", x_n, x_min, x_max, y_n, y_min, y_max)
        fill_hist_from_points(h, x_values, y_values, z_values, **kwargs)

        # get contours in a nested list of graphs
        contours = []
        for l in levels:
            # frame the histogram
            _h = h
            for fk in filter(bool, frame_kwargs):
                fk = dict(fk)
                xw = fk.pop("x_width", fk.get("width", x_width_max))
                yw = fk.pop("y_width", fk.pop("width", y_width_max))
                _h = frame_histogram(_h, xw, yw, contour_level=l, **fk)

            # get the contour graphs and filter by the number of points
            graphs = _get_contour(_h, l)
            graphs = [g for g in graphs if g.GetN() >= min_points]
            contours.append(graphs)

    return contours


def _get_contour(hist, level):
    ROOT = import_ROOT()

    # make a clone to set contour levels
    h = hist.Clone()
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


def get_contour_box(graphs):
    assert graphs

    x_values, y_values = [], []
    for g in graphs:
        x, y = get_graph_points(g, errors=False)
        x_values.extend(x)
        y_values.extend(y)

    if not x_values or not y_values:
        return None, None, None, None

    return min(x_values), max(x_values), min(y_values), max(y_values)


def get_graph_points(g, errors=False):
    ROOT = import_ROOT()

    x_values, y_values = [], []
    asym_errors = False
    if errors:
        if isinstance(g, ROOT.TGraphAsymmErrors):
            x_errors_up, x_errors_down = [], []
            y_errors_up, y_errors_down = [], []
            asym_errors = True
        elif isinstance(g, ROOT.TGraphErrors):
            x_errors, y_errors = [], []
        else:
            errors = False

    x = ctypes.c_double()
    y = ctypes.c_double()
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        x_values.append(x.value)
        y_values.append(y.value)
        if asym_errors:
            x_errors_up.append(g.GetErrorXhigh(i))
            x_errors_down.append(g.GetErrorXlow(i))
            y_errors_up.append(g.GetErrorYhigh(i))
            y_errors_down.append(g.GetErrorYlow(i))
        elif errors:
            x_errors.append(g.GetErrorX(i))
            y_errors.append(g.GetErrorY(i))

    if asym_errors:
        return x_values, y_values, x_errors_down, x_errors_up, y_errors_down, y_errors_up

    if errors:
        return x_values, y_values, x_errors, y_errors

    return x_values, y_values


def repeat_graph(g, n):
    points = sum((n * [p] for p in zip(*get_graph_points(g))), [])

    g_repeated = g.__class__(len(points))

    for i, (x, y) in enumerate(points):
        g_repeated.SetPoint(i, x, y)

    return g_repeated


def invert_graph(
    g,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    x_axis=None,
    y_axis=None,
    offset=0.0,
):
    # get all graph values
    x_values, y_values = get_graph_points(g)

    # get default frame values
    if x_min is None:
        if x_axis:
            x_min = x_axis.GetXmin() - 0.01 * (x_axis.GetXmax() - x_axis.GetXmin())
        else:
            x_min = min(x_values)
    if x_max is None:
        if x_axis:
            x_max = x_axis.GetXmax() + 0.01 * (x_axis.GetXmax() - x_axis.GetXmin())
        else:
            x_max = max(x_values)
    if y_min is None:
        if y_axis:
            y_min = y_axis.GetXmin() - 0.01 * (y_axis.GetXmax() - y_axis.GetXmin())
        else:
            y_min = min(y_values)
    if y_max is None:
        if y_axis:
            y_max = y_axis.GetXmax() + 0.01 * (y_axis.GetXmax() - y_axis.GetXmin())
        else:
            y_max = max(y_values)

    # define outer "frame" points
    bl = (x_min - offset, y_min - offset)
    tl = (x_min - offset, y_max + offset)
    tr = (x_max + offset, y_max + offset)
    br = (x_max + offset, y_min - offset)
    corners = [tr, br, bl, tl]

    # find the corner that is closest to the graph start point
    dist = lambda x, y: ((x - x_values[0])**2 + (y - y_values[0])**2)**0.5
    start_index = min(list(range(4)), key=lambda i: dist(corners[i][0], corners[i][1]))

    # to invert the graph, create a new graph whose points start with an outer frame consisting of
    # the 4 corners, the graph points itself, the first point of the graph again to close it, and
    # ending with the closest corner again
    points = (2 * corners)[start_index:start_index + 5]
    points.extend(list(zip(x_values, y_values)))
    points.append((x_values[0], y_values[0]))
    points.append(corners[start_index])

    # copy the graph and fill points
    g_inv = g.__class__(len(points))
    for i, (x, y) in enumerate(points):
        g_inv.SetPoint(i, x, y)

    return g_inv


def fill_legend_column(legend_entries, n_column_entries, dummy_obj):
    n_entries = len(legend_entries)
    if not isinstance(dummy_obj, tuple):
        dummy_obj = (dummy_obj, " ", "L")

    # number of missing entries in the column
    n_missing = int(math.ceil((1.0 * n_entries) / n_column_entries)) * n_column_entries - n_entries

    # fill up legend entries
    for _ in range(n_missing):
        legend_entries.append(dummy_obj)


def get_text_extent(t, text_size=None, text_font=None):
    ROOT = import_ROOT()

    # convert to a tlatex if t is a string, otherwise clone
    t = ROOT.TLatex(0.0, 0.0, t) if isinstance(t, six.string_types) else t.Clone()

    # set size and font when set
    if text_size is not None:
        t.SetTextSize(text_size)
    if text_font is not None:
        t.SetTextFont(text_font)

    # only available when the font precision is 3
    assert t.GetTextFont() % 10 == 3, "font precision must be 3 to estimate text extent"

    # create a temporary canvas and draw the text
    with temporary_canvas() as c:
        c.cd()
        t.Draw()

        # extract the bounding box dimensions
        w = array.array("I", [0])
        h = array.array("I", [0])
        t.GetBoundingBox(w, h)

    return int(w[0]), int(h[0])


@contextlib.contextmanager
def temporary_canvas(*args, **kwargs):
    ROOT = import_ROOT()

    c = ROOT.TCanvas(*args, **kwargs)

    try:
        yield c
    finally:
        if c and not c.IsZombie():
            c.Close()


def locate_contour_labels(
    graphs,
    label_width,
    label_height,
    pad_width,
    pad_height,
    x_min,
    x_max,
    y_min,
    y_max,
    other_positions=None,
    min_points=10,
    label_offset=None,
):
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
        block_size = n_points if n_blocks == 1 else int(round_scientific(label_width))

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
        rot = 0.0
        for idx2 in np.append(adist, idx1):
            # rotation
            ind = (idx2 * block_size + hb_size) % n_points
            dx, dy = np.gradient(line_contour, axis=0, edge_order=1)[ind]
            dx = x_to_px(dx)
            dy = y_to_px(dy)
            if dx or dy:
                rot = np.rad2deg(np.arctan2(dy, dx))
                # ensure that the rotation is between -90 and 90 deg
                rot = (rot + 90) % 180 - 90
            else:
                rot = 0.0

            # position
            x, y = xx[idx2, hb_size], yy[idx2, hb_size]

            # check positioning
            if not (x_min_vis <= x <= x_max_vis) or not (y_min_vis <= y <= y_max_vis):
                continue
            elif any(too_close(x, y, x0, y0) for x0, y0, _ in positions + other_positions):
                continue

            # at this point, a good position was found
            break

        # apply a relative label offset
        if label_offset:
            s = label_height * label_offset
            x += s * math.sin(np.deg2rad(rot))
            y += s * math.cos(np.deg2rad(rot))

        # store when visible
        if (x_min_vis <= x <= x_max_vis) and (y_min_vis <= y <= y_max_vis):
            positions.append((x, y, rot))

    return positions
