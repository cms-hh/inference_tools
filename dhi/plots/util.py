# coding: utf-8

"""
Different helpers and ROOT style configurations to be used with plotlib.
"""

import math
import array
import uuid
import functools
import collections
import itertools
import contextlib

import six
import numpy as np
import scipy.interpolate

from dhi.config import poi_data, br_hh_names
from dhi.util import import_ROOT, try_int, to_root_latex, make_list


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


def create_random_name():
    return str(uuid.uuid4())


def draw_model_parameters(model_parameters, pad, grouped=False, x_offset=25, y_offset=40, dy=24,
        props=None):
    import plotlib.root as r
    from plotlib.util import merge_dicts

    # merge properties with defaults
    props = merge_dicts({"TextSize": 20}, props)

    labels = []

    if grouped:
        # group parameters by value
        groups = collections.OrderedDict()
        for p, v in model_parameters.items():
            # each parameter key can be a list
            for _p in make_list(p):
                p_label = poi_data.get(_p, {}).get("label", _p)
                groups.setdefault(v, []).append(p_label)

        # create labels
        for i, (v, ps) in enumerate(groups.items()):
            label = "{} = {}".format(" = ".join(map(str, ps)), try_int(v))
            label = r.routines.create_top_left_label(label, pad=pad, props=props, x_offset=x_offset,
                y_offset=y_offset + i * dy)
            labels.append(label)

    else:
        # create one label per parameter
        for i, (p, v) in enumerate(model_parameters.items()):
            # each parameter key can be a list
            p_labels = [poi_data.get(_p, {}).get("label", _p) for _p in make_list(p)]
            label = "{} = {}".format(" = ".join(p_labels), try_int(v))
            label = r.routines.create_top_left_label(label, pad=pad, props=props, x_offset=x_offset,
                y_offset=y_offset + i * dy)
            labels.append(label)

    return labels


def create_hh_process_label(poi="r", br=None):
    return "pp #rightarrow {}{}".format(
        {"r": "HH (incl.)", "r_gghh": "HH", "r_qqhh": "HHqq", "r_vhh": "VHH"}.get(poi, "HH"),
        " ({})".format(to_root_latex(br_hh_names[br])) if br in br_hh_names else "",
    )


def determine_limit_digits(limit, is_xsec=False):
    # TODO: adapt to publication style
    if is_xsec:
        if limit < 10:
            return 2
        elif limit < 200:
            return 1
        else:
            return 0
    else:
        if limit < 10:
            return 2
        elif limit < 300:
            return 1
        else:
            return 0


def get_y_range(y_min_value, y_max_value, y_min=None, y_max=None, log=False, top_margin=0.38,
        visible_margin=0.4):
    if log:
        if y_min is None:
            y_min = (0.75 * y_min_value) if y_min_value > 0 else 1e-3
        if y_max is None:
            y_max = y_min * 10**(math.log10(y_max_value / y_min) * (1. + top_margin))
        y_max_vis = y_min * 10**(math.log10(y_max / y_min) / (1. + visible_margin))
    else:
        if y_min is None:
            y_min = 0. if y_min_value is None else y_min_value
        if y_max is None:
            y_max = (y_max_value - y_min) * (1. + top_margin)
        y_max_vis = y_max / (1. + visible_margin) + y_min

    return y_min, y_max, y_max_vis


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
    hist_padded = hist.__class__(create_random_name(), "", len(x_edges) - 1,
        array.array("d", x_edges), len(y_edges) - 1, array.array("d", y_edges))
    hist_padded.SetDirectory(0)
    for by, _data in enumerate(data):
        for bx, z in enumerate(_data):
            hist_padded.SetBinContent(bx + 1, by + 1, z)

    return hist_padded


# helper to fill each bin in a 2D histogram from potentially sparse points via interpolation
def fill_hist_from_points(h, x_values, y_values, z_values, interpolation="tgraph2d"):
    ROOT = import_ROOT()

    # remove nans in z_values
    nan_indices = np.argwhere(np.isnan(np.array(z_values)))
    x_values = [x for i, x in enumerate(x_values) if i not in nan_indices]
    y_values = [y for i, y in enumerate(y_values) if i not in nan_indices]
    z_values = [z for i, z in enumerate(z_values) if i not in nan_indices]

    # create an interpolation function
    if interpolation == "tgraph2d":
        g = ROOT.TGraph2D(len(z_values))
        for i, (x, y, z) in enumerate(zip(x_values, y_values, z_values)):
            g.SetPoint(i, x, y, z)
        interp = lambda x, y: g.Interpolate(x, y)
    elif interpolation.startswith("interp2d_"):
        interp = scipy.interpolate.interp2d(x_values, y_values, z_values, kind=interpolation[9:])

    # then, fill histogram bins
    if isinstance(h, ROOT.TH2Poly):
        for poly_bin in h.GetBins():
            z = interp(*find_poly_bin_center(poly_bin))
            poly_bin.SetContent(z)
    else:
        # strictly rectangular bins
        for bx in range(1, h.GetNbinsX() + 1):
            for by in range(1, h.GetNbinsY() + 1):
                x = h.GetXaxis().GetBinCenter(bx)
                y = h.GetYaxis().GetBinCenter(by)
                h.SetBinContent(bx, by, interp(x, y))


def find_poly_bin_center(poly_bin, n=1000):
    # get an initial guess
    x = poly_bin.GetXMin() + 0.5 * (poly_bin.GetXMax() - poly_bin.GetXMin())
    y = poly_bin.GetYMin() + 0.5 * (poly_bin.GetYMax() - poly_bin.GetYMin())

    # vary until the point is inside the bin
    for _ in range(n):
        if poly_bin.IsInside(x, y):
            return x, y

        # vary
        raise NotImplementedError("center determination of complex poly bins not implemented yet")

    raise Exception("could not determine poly bin center after {} iterations".format(n))


# helper to extract contours
def get_contours(x_values, y_values, z_values, levels, frame_kwargs=None, min_points=10):
    ROOT = import_ROOT()

    # remove nans in z_values
    nan_indices = np.argwhere(np.isnan(np.array(z_values)))
    x_values = [x for i, x in enumerate(x_values) if i not in nan_indices]
    y_values = [y for i, y in enumerate(y_values) if i not in nan_indices]
    z_values = [z for i, z in enumerate(z_values) if i not in nan_indices]

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
    with temporary_canvas() as c:
        c.cd()
        h = ROOT.TH2F(create_random_name(), "", x_n, x_min, x_max, y_n, y_min, y_max)
        fill_hist_from_points(h, x_values, y_values, z_values)

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

    x, y = ROOT.Double(), ROOT.Double()
    for i in range(g.GetN()):
        g.GetPoint(i, x, y)
        x_values.append(float(x))
        y_values.append(float(y))
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
    elif errors:
        return x_values, y_values, x_errors, y_errors
    else:
        return x_values, y_values


def repeat_graph(g, n):
    points = sum((n * [p] for p in zip(*get_graph_points(g))), [])

    g_repeated = g.__class__(len(points))

    for i, (x, y) in enumerate(points):
        g_repeated.SetPoint(i, x, y)

    return g_repeated


def invert_graph(g, x_min=None, x_max=None, y_min=None, y_max=None, x_axis=None, y_axis=None,
        offset=0.):
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
    dist = lambda x, y: ((x - x_values[0])**2. + (y - y_values[0])**2.)**0.5
    start_index = min(list(range(4)), key=lambda i: dist(corners[i][0], corners[i][1]))

    # to invert the graph, create a new graph whose points start with an outer frame consisting of
    # the 4 corners, the graph points itself, the first point of the graph again to close it, and
    # ending with the closest corner again
    points = (2 * corners)[start_index:start_index + 5]
    points.extend(zip(x_values, y_values))
    points.append((x_values[0], y_values[0]))
    points.append(corners[start_index])

    # copy the graph and fill points
    g_inv = g.__class__(len(points))
    for i, (x, y) in enumerate(points):
        g_inv.SetPoint(i, x, y)

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


@contextlib.contextmanager
def temporary_canvas(*args, **kwargs):
    ROOT = import_ROOT()

    c = ROOT.TCanvas(*args, **kwargs)

    try:
        yield c
    finally:
        if c and not c.IsZombie():
            c.Close()
