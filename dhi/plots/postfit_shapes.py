# coding: utf-8

"""
Postfit shape plots using ROOT.
"""

import math
import array
from collections import defaultdict

import six
import uproot

from dhi.config import poi_data, campaign_labels, colors, cms_postfix
from dhi.util import (
    import_ROOT, DotDict, to_root_latex, linspace, try_int, poisson_asym_errors, make_list, warn,
)
from dhi.plots.util import use_style, create_model_parameters


colors = colors.root


@use_style("dhi_default")
def plot_s_over_b(
    paths,
    poi,
    fit_diagnostics_path,
    bins=8,
    signal_superimposed=False,
    signal_scale=1.,
    signal_scale_ratio=1.,
    show_best_fit=True,
    y1_min=None,
    y1_max=None,
    y2_min=None,
    y2_max=None,
    model_parameters=None,
    campaign=None,
    prefit=False,
    unblinded=False,
    paper=False,
):
    """
    Creates a postfit signal-over-background plot combined over all bins in the fit of a *poi* and
    saves it at *paths*. The plot is based on the fit diagnostics file *fit_diagnostics_path*
    produced by combine. *bins* can either be a single number of bins to use, or a list of n+1 bin
    edges.

    When *signal_superimposed* is *True*, the signal at the top pad
    is not drawn stacked on top of the background but as a separate histogram. For visualization
    purposes, the fitted signal can be scaled by *signal_scale*, and, when drawing the signal
    superimposed, by *signal_scale_ratio* at the bottom ratio pad. When *signal_superimposed* is
    *True*, the signal at the top pad is not drawn stacked on top of the background but as a
    separate histogram. When *show_best_fit* is *False*, the value of the signal scale is not shown
    in the legend labels.

    *y1_min*, *y1_max*, *y2_min* and *y2_max* define the ranges of the y-axes of the upper pad and
    ratio pad, respectively. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*. When *prefit* is *True*, signal, background and uncertainties are
    shown according to the prefit expectation. When *unblinded* is *True*, some legend labels are
    changed accordingly. When *paper* is *True*, certain plot configurations are adjusted for use in
    publications.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/postfit.html#combined-postfit-shapes
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # input checks
    assert signal_scale != 0
    assert signal_scale_ratio != 0
    if not signal_superimposed:
        signal_scale_ratio = signal_scale
    assert not (prefit and unblinded)

    # load the shape data from the fit diagnostics file
    bin_data = load_bin_data(fit_diagnostics_path)

    # warn about categories with missing signal or data contributions
    bad_signal_indices = set()
    bad_data_indices = set()
    missing_signals = defaultdict(int)
    missing_data = defaultdict(int)
    for i, b in enumerate(bin_data):
        if b.pre_signal is None:
            bad_signal_indices.add(i)
            missing_signals[b.category] += 1
        if b.data is None:
            bad_data_indices.add(i)
            missing_data[b.category] += 1
    if bad_signal_indices:
        bin_data = [b for i, b in enumerate(bin_data) if i not in bad_signal_indices]
        warn(
            "WARNING: detected {} categories without signal contributions that will not be "
            "considered in this plot; ignore this warning when they are pure control regions; "
            "categories:\n  - {}".format(
                len(missing_signals),
                "\n  - ".join("{} ({} bins)".format(c, n) for c, n in missing_signals.items()),
            )
        )
    if len(bad_data_indices) not in [0, len(bin_data)]:
        warn(
            "WARNING: detected {} categories without data contributions; this can affect the "
            "agreement in the produced plot; categories:\n  - {}".format(
                len(missing_data),
                "\n  - ".join("{} ({} bins)".format(c, n) for c, n in missing_data.items()),
            )
        )

    # compute prefit log(s/b) where possible
    x_min = 1.e5
    x_max = -1.e5
    for b in bin_data:
        if b.pre_signal > 0:
            b["pre_s_over_b"] = math.log(b.pre_signal / b.pre_background, 10.)
            x_min = min(x_min, b.pre_s_over_b)
            x_max = max(x_max, b.pre_s_over_b)

    # infer bin edges when only a number of bins is passed
    if isinstance(bins, six.integer_types):
        bins = linspace(x_min, x_max, bins + 1)
    else:
        x_min = bins[0]
        x_max = bins[-1]

    # load the signal strength modifier
    signal_strength = uproot.open(fit_diagnostics_path)["tree_fit_sb"].array(poi).tolist()[0]

    # start plotting
    r.setup_style()
    canvas, (pad1, pad2) = r.routines.create_canvas(divide=(1, 2))
    r.setup_pad(pad1, props={"Logy": True, "BottomMargin": 0.3})
    r.setup_pad(pad2, props={"TopMargin": 0.7})
    pad1.cd()
    draw_objs1 = []
    draw_objs2 = []
    legend_entries = []

    # dummy histograms for both pads to control axes
    h_dummy1 = ROOT.TH1F("dummy1", ";;Events", 1, x_min, x_max)
    h_dummy2 = ROOT.TH1F("dummy2", ";Pre-fit expected log_{10}(S/B);Data / Bkg.", 1, x_min, x_max)
    r.setup_hist(h_dummy1, pad=pad1, props={"LineWidth": 0})
    r.setup_hist(h_dummy2, pad=pad2, props={"LineWidth": 0})
    r.setup_y_axis(h_dummy2.GetYaxis(), pad2, props={"Ndivisions": 6, "CenterTitle": True})
    r.setup_x_axis(h_dummy2.GetXaxis(), pad2, props={"TitleOffset": 1.23})
    draw_objs1.append((h_dummy1, "HIST"))
    draw_objs2.append((h_dummy2, "HIST"))

    # helper to create a signal label
    def signal_label(scale):
        label = "Signal"
        if not show_best_fit:
            return label
        if scale == 1:
            label += " x {}".format(try_int(scale))
        label += " ({} = {:.2f})".format(to_root_latex(poi_data[poi].label), signal_strength)
        return label

    # superimposed signal histogram at the top
    hist_s1 = ROOT.TH1F("s_post1", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_s1, props={"LineColor": colors.blue_signal})
    if signal_superimposed:
        draw_objs1.append((hist_s1, "SAME,HIST"))
        legend_entries.append((hist_s1, signal_label(signal_scale), "L"))

    # postfit signal histogram at the top
    hist_sb1 = ROOT.TH1F("sb_post1", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_sb1, props={"FillColor": colors.blue_signal})
    if signal_superimposed:
        legend_entries.append((hist_sb1, signal_label(signal_scale_ratio), "AF"))
    else:
        legend_entries.append((hist_sb1, signal_label(signal_scale), "AF"))
        draw_objs1.append((hist_sb1, "SAME,HIST"))

    # postfit B histogram at the top
    fit_type = "pre" if prefit else "post"
    hist_b1 = ROOT.TH1F("b_post1", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_b1, props={"FillColor": colors.white})
    legend_entries.insert(0, (hist_b1, "Background ({}-fit)".format(fit_type), "L"))
    if signal_superimposed:
        draw_objs1.insert(-1, (hist_b1, "SAME,HIST"))
    else:
        draw_objs1.append((hist_b1, "SAME,HIST"))

    # dummy data histogram to handle binning
    hist_d1 = ROOT.TH1F("d1", "", len(bins) - 1, array.array("f", bins))

    # actual data graph at the top
    data_postfix = "" if unblinded else " (Asimov)"
    graph_d1 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_d1, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 1}, color=1)
    draw_objs1.append((graph_d1, "PEZ,SAME"))
    legend_entries.insert(0, (graph_d1, "Data" + data_postfix, "LP"))

    # postfit S+B ratio histogram and a mask histogram to mimic errors
    hist_sb2 = ROOT.TH1F("sb_post2", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_sb2, props={"FillColor": colors.blue_signal})
    draw_objs2.append((hist_sb2, "SAME,HIST"))
    hist_mask2 = ROOT.TH1F("mask_post2", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_mask2, props={"FillColor": colors.white})
    draw_objs2.append((hist_mask2, "SAME,HIST"))

    # dummy histograms to handle binning of background uncertainties
    hist_b_err_up1 = ROOT.TH1F("b_err_up1", "", len(bins) - 1, array.array("f", bins))
    hist_b_err_down1 = ROOT.TH1F("b_err_down1", "", len(bins) - 1, array.array("f", bins))

    # postfit background uncertainty in the ratio
    graph_b_err2 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_b_err2, props={"FillColor": colors.black, "FillStyle": 3345, "LineWidth": 0})
    draw_objs2.append((graph_b_err2, "SAME,2"))
    legend_entries.insert(2, (graph_b_err2, "Uncertainty ({}-fit)".format(fit_type), "F"))

    # data graph in the ratio
    graph_d2 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_d2, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 1}, color=1)
    draw_objs2.append((graph_d2, "SAME,PEZ"))

    # fill histograms by traversing bin data
    for _bin in bin_data:
        s_over_b = min(x_max - 1e-5, max(_bin.get("pre_s_over_b", x_min), x_min + 1e-5))
        # get values to fill
        s = _bin[fit_type + "_signal"]
        b = _bin[fit_type + "_background"]
        b_err_up = _bin[fit_type + "_background_err_up"]
        b_err_down = _bin[fit_type + "_background_err_down"]
        d = _bin.data
        # signal and background
        hist_b1.Fill(s_over_b, b)
        hist_s1.Fill(s_over_b, s * signal_scale)
        hist_sb1.Fill(s_over_b, b + s * signal_scale)
        # data histogram for binning
        hist_d1.Fill(s_over_b, d)
        # background uncertainty histogram for binning
        hist_b_err_up1.Fill(s_over_b, b_err_up)
        hist_b_err_down1.Fill(s_over_b, b_err_down)

    # fill remaining objects
    for i in range(hist_sb1.GetNbinsX()):
        # get values from top histograms
        s = hist_s1.GetBinContent(i + 1) / signal_scale
        x = hist_b1.GetBinCenter(i + 1)
        w = hist_b1.GetBinWidth(i + 1)
        b = hist_b1.GetBinContent(i + 1)
        b_err_up = hist_b_err_up1.GetBinContent(i + 1)
        b_err_down = hist_b_err_down1.GetBinContent(i + 1)
        d = hist_d1.GetBinContent(i + 1)
        d_err_up, d_err_down = poisson_asym_errors(d)
        # zero safe b value, leading to almost 0 when used as denominator
        b_safe = b or 1e15
        # bottom signal + background histogram and mask
        sb = b + s * signal_scale_ratio
        hist_sb2.SetBinContent(i + 1, max(sb / b_safe, 1.))
        hist_mask2.SetBinContent(i + 1, min(sb / b_safe, 1.))
        # data points at the top
        graph_d1.SetPoint(i, x, d)
        graph_d1.SetPointError(i, 0., 0., d_err_down, d_err_up)
        # data points in the ratio
        graph_d2.SetPoint(i, x, d / b_safe)
        graph_d2.SetPointError(i, 0., 0., d_err_down / b_safe, d_err_up / b_safe)
        # uncertainty in the ratio
        graph_b_err2.SetPoint(i, x, 1.)
        graph_b_err2.SetPointError(i, 0.5 * w, 0.5 * w, b_err_down / b_safe, b_err_up / b_safe)

    # set y ranges in both pads
    if y1_min is None:
        y1_min = 5.
    if y1_max is None:
        y1_max_value = hist_b1.GetMaximum()
        y1_max = y1_min * 10**(1.35 * math.log10(y1_max_value / y1_min))
    if y2_min is None:
        y2_min = 0.7
        y2_max = 1.7
    h_dummy1.SetMinimum(y1_min)
    h_dummy1.SetMaximum(y1_max)
    h_dummy2.SetMinimum(y2_min)
    h_dummy2.SetMaximum(y2_max)

    # legend
    legend = r.routines.create_legend(pad=pad1, width=250, n=len(legend_entries))
    r.fill_legend(legend, legend_entries)
    draw_objs1.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad1, "tr",
        props={"LineWidth": 0, "FillColor": colors.white_trans_70})
    draw_objs1.insert(-1, legend_box)

    # model parameter labels
    if model_parameters:
        draw_objs1.extend(create_model_parameters(model_parameters, pad1, x_offset=200))

    # cms label
    cms_labels = r.routines.create_cms_labels(postfix="" if paper else cms_postfix, pad=pad1)
    draw_objs1.extend(cms_labels)

    # campaign label
    if campaign:
        campaign_label = to_root_latex(campaign_labels.get(campaign, campaign))
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad1)
        draw_objs1.append(campaign_label)

    # draw all objects
    pad1.cd()
    r.routines.draw_objects(draw_objs1)
    pad2.cd()
    r.routines.draw_objects(draw_objs2)

    # save
    r.update_canvas(canvas)
    for path in make_list(paths):
        canvas.SaveAs(path)


def load_bin_data(fit_diagnostics_path):
    ROOT = import_ROOT()

    # the returned list contains information per bin in a dict
    bin_data = []

    # open the fit diagnostics file and load content
    tfile = ROOT.TFile(fit_diagnostics_path, "READ")
    tdir_pre = tfile.Get("shapes_prefit")
    tdir_post = tfile.Get("shapes_fit_s")

    # traverse categories
    for cat_key in tdir_pre.GetListOfKeys():
        cat_name = cat_key.GetName()
        cat_dir_pre = tdir_pre.Get(cat_name)
        cat_dir_post = tdir_post.Get(cat_name)
        if not isinstance(cat_dir_pre, ROOT.TDirectoryFile):
            continue
        elif not isinstance(cat_dir_post, ROOT.TDirectoryFile):
            raise Exception("category '{}' exists for pre- but not postfit".format(cat_name))

        # get histograms and the data graph
        s_pre = cat_dir_pre.Get("total_signal")
        b_pre = cat_dir_pre.Get("total_background")
        a_pre = cat_dir_pre.Get("total")
        d_pre = cat_dir_pre.Get("data")
        s_post = cat_dir_post.Get("total_signal")
        b_post = cat_dir_post.Get("total_background")
        a_post = cat_dir_post.Get("total")
        d_post = cat_dir_post.Get("data")

        # some dimension checks
        n = b_pre.GetNbinsX()
        assert b_pre.GetNbinsX() == a_pre.GetNbinsX() == n
        assert b_post.GetNbinsX() == a_post.GetNbinsX() == n
        # total_signal's might be missing when a pure control region was fitted
        if s_pre:
            assert s_pre.GetNbinsX() == s_post.GetNbinsX() == n
        # data might be missing when blinded
        if d_pre:
            assert d_pre.GetN() == d_post.GetN() == n

        # read bins one by one
        for i in range(n):
            # skip bins where data or background is zero in prefit
            epsilon = 1e-5
            if b_pre.GetBinContent(i + 1) < epsilon or (d_pre and d_pre.GetY()[i] < epsilon):
                continue

            # fill bin data
            bin_data.append(DotDict(
                category=cat_name,
                bin=i,
                pre_signal=s_pre.GetBinContent(i + 1) if s_pre else None,
                pre_signal_err_up=s_pre.GetBinErrorUp(i + 1) if s_pre else None,
                pre_signal_err_down=s_pre.GetBinErrorLow(i + 1) if s_pre else None,
                pre_background=b_pre.GetBinContent(i + 1),
                pre_background_err_up=b_pre.GetBinErrorUp(i + 1),
                pre_background_err_down=b_pre.GetBinErrorLow(i + 1),
                pre_all=a_pre.GetBinContent(i + 1),
                pre_all_err_up=a_pre.GetBinErrorUp(i + 1),
                pre_all_err_down=a_pre.GetBinErrorLow(i + 1),
                post_signal=s_post.GetBinContent(i + 1) if s_post else None,
                post_signal_err_up=s_post.GetBinErrorUp(i + 1) if s_post else None,
                post_signal_err_down=s_post.GetBinErrorLow(i + 1) if s_post else None,
                post_background=b_post.GetBinContent(i + 1),
                post_background_err_up=b_post.GetBinErrorUp(i + 1),
                post_background_err_down=b_post.GetBinErrorLow(i + 1),
                post_all=a_post.GetBinContent(i + 1),
                post_all_err_up=a_post.GetBinErrorUp(i + 1),
                post_all_err_down=a_post.GetBinErrorLow(i + 1),
                data=d_post.GetY()[i] if d_post else None,
            ))

    return bin_data
