# coding: utf-8

"""
Postfit shape plots using ROOT.
"""

import math
import array

import six
import uproot

from dhi.config import poi_data, campaign_labels, colors
from dhi.util import import_ROOT, DotDict, to_root_latex, linspace, try_int, poisson_asym_errors
from dhi.plots.util import use_style, create_model_parameters


colors = colors.root


@use_style("dhi_default")
def plot_s_over_b(
    path,
    poi,
    fit_diagnostics_path,
    bins=8,
    y1_min=None,
    y1_max=None,
    y2_min=None,
    y2_max=None,
    signal_scale=100.,
    model_parameters=None,
    campaign=None
):
    """
    Creates a postfit signal-over-background plot combined over all bins in the fit of a *poi* and
    saves it at *path*. The plot is based on the fit diagnostics file *fit_diagnostics_path*
    produced by combine. *bins* can either be a single number of bins to use, or a list of n+1 bin
    edges. *y1_min*, *y1_max*, *y2_min* and *y2_max* define the ranges of the y-axes of the upper
    pad and ratio pad, respectively. The signal can optionally be scaled by *signal_scale* for
    visualization purposes. *model_parameters* can be a dictionary of key-value pairs of model
    parameters. *campaign* should refer to the name of a campaign label defined in
    *dhi.config.campaign_labels*.

    Example: https://cms-hh.web.cern.ch/tools/inference/tasks/postfit.html#combined-postfit-shapes
    """
    import plotlib.root as r
    ROOT = import_ROOT()

    # load the shape data from the fit diagnostics file
    bin_data = load_bin_data(fit_diagnostics_path)

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

    # postfit signal histogram at the top
    hist_s_post1 = ROOT.TH1F("s_post1", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_s_post1, props={"FillColor": colors.blue_signal})
    draw_objs1.append((hist_s_post1, "SAME,HIST"))
    scale_text = "" if signal_scale == 1 else " x {}".format(try_int(signal_scale))
    signal_label = "Signal ({} = {:.2f}){}".format(to_root_latex(poi_data[poi].label),
        signal_strength, scale_text)
    legend_entries.append((hist_s_post1, signal_label, "AF"))

    # postfit background histogram at the top
    hist_b_post1 = ROOT.TH1F("b_post1", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_b_post1, props={"FillColor": colors.white})
    draw_objs1.append((hist_b_post1, "SAME,HIST"))
    legend_entries.insert(0, (hist_b_post1, "Background (postfit)", "L"))

    # dummy data histogram to handle binning
    hist_d1 = ROOT.TH1F("d1", "", len(bins) - 1, array.array("f", bins))

    # actual data graph at the top
    graph_d1 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_d1, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 1}, color=1)
    draw_objs1.append((graph_d1, "PEZ,SAME"))
    legend_entries.insert(0, (graph_d1, "Data", "LP"))

    # postfit signal histogram in the ratio
    hist_s_post2 = ROOT.TH1F("s_post2", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_s_post2, props={"FillColor": colors.blue_signal})
    draw_objs2.append((hist_s_post2, "SAME,HIST"))

    # postfit background histogram in the ratio (all ones)
    hist_b_post2 = ROOT.TH1F("b_post2", "", len(bins) - 1, array.array("f", bins))
    r.setup_hist(hist_b_post2, props={"FillColor": colors.white})
    draw_objs2.append((hist_b_post2, "SAME,HIST"))

    # dummy histograms to handle binning of background uncertainties
    hist_b_err_up1 = ROOT.TH1F("b_err_up1", "", len(bins) - 1, array.array("f", bins))
    hist_b_err_down1 = ROOT.TH1F("b_err_down1", "", len(bins) - 1, array.array("f", bins))

    # postfit background uncertainty in the ratio
    graph_b_err2 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_b_err2, props={"FillColor": colors.black, "FillStyle": 3345, "LineWidth": 0})
    draw_objs2.append((graph_b_err2, "SAME,2"))
    legend_entries.insert(-1, (graph_b_err2, "Uncertainty (postfit)", "F"))

    # data graph in the ratio
    graph_d2 = ROOT.TGraphAsymmErrors(len(bins) - 1)
    r.setup_hist(graph_d2, props={"LineWidth": 2, "MarkerStyle": 20, "MarkerSize": 1}, color=1)
    draw_objs2.append((graph_d2, "SAME,PEZ"))

    # fill histograms by traversing bin data
    for b in bin_data:
        s_over_b = min(x_max - 1e-5, max(b.get("pre_s_over_b", x_min), x_min + 1e-5))
        # signal and background
        hist_b_post1.Fill(s_over_b, b.post_background)
        hist_s_post1.Fill(s_over_b, b.post_background + b.post_signal * signal_scale)
        # data histogram for binning
        hist_d1.Fill(s_over_b, b.data)
        # background uncertainty histogram for binning
        hist_b_err_up1.Fill(s_over_b, b.post_background_err_up)
        hist_b_err_down1.Fill(s_over_b, b.post_background_err_down)

    # fill remaining objects
    for i in range(hist_s_post1.GetNbinsX()):
        # get values
        x = hist_s_post1.GetBinCenter(i + 1)
        w = hist_s_post1.GetBinWidth(i + 1)
        s = hist_s_post1.GetBinContent(i + 1)
        b = hist_b_post1.GetBinContent(i + 1)
        b_err_up = hist_b_err_up1.GetBinContent(i + 1)
        b_err_down = hist_b_err_down1.GetBinContent(i + 1)
        d = hist_d1.GetBinContent(i + 1)
        d_err = poisson_asym_errors(d)
        # signal and background
        hist_s_post2.SetBinContent(i + 1, s / b)
        hist_b_post2.SetBinContent(i + 1, 1.)
        # data points at the top
        graph_d1.SetPoint(i, x, d)
        graph_d1.SetPointError(i, 0., 0., d_err[1], d_err[0])
        # data points in the ratio
        graph_d2.SetPoint(i, x, d / b)
        graph_d2.SetPointError(i, 0., 0., d_err[1] / b, d_err[0] / b)
        # uncertainty in the ratio
        graph_b_err2.SetPoint(i, x, 1.)
        graph_b_err2.SetPointError(i, 0.5 * w, 0.5 * w, b_err_down / b, b_err_up / b)

    # set y ranges in both pads
    if y1_min is None:
        y1_min = 5.
    if y1_max is None:
        y1_max_value = hist_b_post1.GetMaximum()
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
        draw_objs1.extend(create_model_parameters(model_parameters, pad1))

    # cms label
    cms_labels = r.routines.create_cms_labels(pad=pad1)
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
        n = s_pre.GetNbinsX()
        assert(b_pre.GetNbinsX() == a_pre.GetNbinsX() == n)
        assert(s_post.GetNbinsX() == b_post.GetNbinsX() == a_post.GetNbinsX() == n)
        assert(not d_pre or d_pre.GetN() == n)
        assert(not d_post or d_post.GetN() == n)

        # read bins one by one
        for i in range(n):
            # skip bins where data or background is zero in prefit
            epsilon = 1e-5
            if b_pre.GetBinContent(i + 1) < epsilon or (d_pre and d_pre.GetY()[i] < epsilon):
                continue

            # fill bin data
            bin_data.append(DotDict(
                pre_signal=s_pre.GetBinContent(i + 1),
                pre_signal_err_up=s_pre.GetBinErrorUp(i + 1),
                pre_signal_err_down=s_pre.GetBinErrorLow(i + 1),
                pre_background=b_pre.GetBinContent(i + 1),
                pre_background_err_up=b_pre.GetBinErrorUp(i + 1),
                pre_background_err_down=b_pre.GetBinErrorLow(i + 1),
                pre_all=a_pre.GetBinContent(i + 1),
                pre_all_err_up=a_pre.GetBinErrorUp(i + 1),
                pre_all_err_down=a_pre.GetBinErrorLow(i + 1),
                post_signal=s_post.GetBinContent(i + 1),
                post_signal_err_up=s_post.GetBinErrorUp(i + 1),
                post_signal_err_down=s_post.GetBinErrorLow(i + 1),
                post_background=b_post.GetBinContent(i + 1),
                post_background_err_up=b_post.GetBinErrorUp(i + 1),
                post_background_err_down=b_post.GetBinErrorLow(i + 1),
                post_all=a_post.GetBinContent(i + 1),
                post_all_err_up=a_post.GetBinErrorUp(i + 1),
                post_all_err_down=a_post.GetBinErrorLow(i + 1),
                data=d_post.GetY()[i] if d_post else None,
            ))

    return bin_data
