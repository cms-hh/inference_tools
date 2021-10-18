#!/usr/bin/env python
# coding: utf-8

"""
Script to extract and plot shapes from a ROOT file create by combine's FitDiagnostics.
"""

import os
from collections import OrderedDict
import json
from shutil import copyfile
#import os.path
from dhi.util import import_ROOT

ROOT = import_ROOT()

def create_postfit_plots(
    path,
    fit_diagnostics_path,
    normalize_X_original,
    doPostFit,
    divideByBinWidth,
    bin,
    binToRead,
    unblind,
    options_dat,
    file_sig_options,
    file_bkg_options
):
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    minY = bin["minY"]
    maxY = bin["maxY"]
    minYerr = bin["minYerr"]
    maxYerr = bin["maxYerr"]
    if doPostFit :
        try: minYerr = bin["minYerr_postfit"]
        except: True
        try: maxYerr = bin["maxYerr_postfit"]
        except: True
    useLogPlot = bin["useLogPlot"]
    era = bin["era"]
    labelX = bin["labelX"]
    header_legend = bin["header_legend"]
    datacard_original = bin["datacard_original"]
    bin_name_original = bin["bin_name_original"]
    number_columns_legend = bin["number_columns_legend"]

    print("Reading %s for signal options/process" % file_sig_options)
    with open(file_sig_options) as ff : procs_plot_options_sig = json.load(ff, object_pairs_hook=OrderedDict)
    scale_GGF = 0
    scale_VBF = 0
    try: scale_GGF = bin["scale_ggf"]
    except:
        print("")
    try: scale_VBF = bin["scale_vbf"]
    except:
        print("")

    typeFit = None
    if doPostFit:
        folder = "shapes_fit_s"
        folder_data = "shapes_fit_s"
        typeFit = "postfit"
    else:
        folder = "shapes_prefit"
        folder_data = "shapes_prefit"
        typeFit = "prefit"

    name_total = "total_background"

    if normalize_X_original:
        fileOrig = datacard_original.replace("$DHI_DATACARDS_RUN2", os.getenv('DHI_DATACARDS_RUN2'))
        print("template on ", fileOrig)
    else:
        fileOrig = fit_diagnostics_path

    print("reading shapes from: ", fit_diagnostics_path)
    fin = ROOT.TFile(fit_diagnostics_path, "READ")
    print("read shapes from: ")

    labelY = "Events"
    if divideByBinWidth:
        labelY = "Events / bin width"

    if not doPostFit:
        header_legend = header_legend + ", \n" + typeFit
    else:
        header_legend = header_legend + ", #mu(t#bar{t}H)=#hat{#mu}"

    # list of folders to read from
    catcats = bin["align_cats"]
    yiels_list = dict.fromkeys(catcats, {})
    round_yiels_list = 4

    print("Reading %s for BKG options/process" % file_bkg_options)
    with open(file_bkg_options) as ff : dprocs = json.load(ff, object_pairs_hook=OrderedDict)

    # add stack of single H as second
    hprocs = ["ggH", "qqH", "bbH", "ttH", "WH", "ZH", "TH", "tHq", "tHW", "VH"]
    hdecays = ["hbb", "hgg", "hmm", "htt", "hww", "hzz", "hcc",]
    if bin["merged_eras_fit"] :
        singleH = [ "%s_%s_%s" % (proc, erastr, decay) for proc in hprocs for erastr in ["2016", "2017", "2018"] for decay in hdecays ]
    else :
        singleH = [ "%s_%s" % (proc, decay) for proc in hprocs for decay in hdecays ]
    # some channels do not separate some by decay, they should, however this is here by now
    singleH += ["TH", "VH", "TTH", "ttVH"]
    ## make a list without the major
    countOnce = 0
    for sh in singleH:
        if countOnce == 0:
            hist = fin.Get(str("%s/%s/%s" % (folder, catcats[0], sh)))
            try:
                hist.Integral()
            except:
                continue
            countOnce = 1
            label_singleH = "single H"
            print("Add single H legend (proc %s)" % sh)
        else:
            label_singleH = "none"
        ordered_dict_prepend(
            dprocs,
            sh,
            {"color": 226, "fillStype": 1001, "label": label_singleH, "make border": False},
        )

    print("will draw processes", list(dprocs.keys()))

    if normalize_X_original:
        fileorriginal = ROOT.TFile(fileOrig, "READ")
        FoundHist = False

        for histRead in list(dprocs.keys()) :
            if not FoundHist :
                readFromOriginal = str("%s/%s" % (bin_name_original, histRead)) if not bin_name_original == "none" else str(histRead)
                print("try original readFrom ", readFromOriginal)
                template = fileorriginal.Get(readFromOriginal)
                if template.Integral() > 0 : FoundHist = True

        print("Getting original readFrom ", readFromOriginal)
        template.GetYaxis().SetTitle(labelY)
        template.SetTitle(" ")
        nbinscatlist = [template.GetNbinsX()]
    else:
        print("Drawing: ", catcats)
        nbinstotal = 0
        nbinscatlist = []
        for catcat in catcats:
            readFromTot = str("%s/%s/%s" % (folder, catcat, name_total))
            hist = fin.Get(readFromTot)
            print("reading shapes", readFromTot)
            print(hist.Integral())
            nbinscat = GetNonZeroBins(hist)
            nbinscatlist += [nbinscat]
            print(readFromTot, nbinscat)
            nbinstotal += nbinscat
        template = ROOT.TH1F("my_hist", "", nbinstotal, 0 - 0.5, nbinstotal - 0.5)
        template.GetYaxis().SetTitle(labelY)
        print(nbinscatlist)

    #legend1 = ROOT.TLegend(0.2400, 0.645, 0.9450, 0.910)
    if "splitline" in header_legend :
        bottom_legend = 0.52
    else :
        bottom_legend = 0.64
    legend1 = ROOT.TLegend(0.2400, bottom_legend, 0.9450, 0.90)
    legend1.SetNColumns(number_columns_legend)
    legend1.SetFillStyle(0)
    legend1.SetBorderSize(0)
    legend1.SetFillColor(10)
    legend1.SetTextSize(0.040 if do_bottom else 0.03)
    legend1.SetHeader(header_legend)
    header = legend1.GetListOfPrimitives().First()
    header.SetTextSize(0.05 if do_bottom else 0.04)
    header.SetTextColor(1)
    header.SetTextFont(62)
    #header.SetEntrySeparation(1)

    dataTGraph1 = ROOT.TGraphAsymmErrors()
    if unblind:
        dataTGraph1.Set(template.GetXaxis().GetNbins())
        lastbin = 0
        for cc, catcat in enumerate(catcats):
            readFrom = str("%s/%s" % (folder, catcat))
            readFromTot = str("%s/%s/%s" % (folder, catcat, name_total))
            print(" histtotal ", readFromTot)
            histtotal = fin.Get(readFromTot)
            data_cat = 0
            info_bin = process_data_histo(
                template,
                data_cat,
                dataTGraph1,
                readFrom,
                fin,
                lastbin,
                histtotal,
                nbinscatlist[cc],
                minY,
                maxY,
                divideByBinWidth,
            )
            yiels_list[catcat]["Data"] = info_bin["data_cat"]
            lastbin += info_bin["allbins"]
        dataTGraph1.Draw()
        legend1.AddEntry(dataTGraph1, "Data", "p")

    lastbin = 0
    hist_total = template.Clone()
    for cc, catcat in enumerate(catcats):
        readFrom = str("%s/%s" % (folder, catcat))
        print("read the hist with total uncertainties", readFrom, catcat)
        info_bin = process_total_histo(
            hist_total,
            readFrom,
            fin,
            divideByBinWidth,
            name_total,
            lastbin,
            do_bottom,
            labelX,
            nbinscatlist[cc],
            minY,
            maxY,
            totalBand=True,
        )
        lastbin += info_bin["allbins"]
        yiels_list[catcat]["Total"] = {
        "central" : round(info_bin["yield_cat"], round_yiels_list),
        "err" : round(info_bin["yield_cat_err"], round_yiels_list)
        }
    print("hist_total", hist_total.Integral())

    ## declare canvases sizes accordingly
    WW = 600
    HH = 700
    TT = 0.08 * HH
    BB = 0.12 * HH
    RR = 0.04 * WW
    if do_bottom:
        LL = 0.13 * WW
        canvas = ROOT.TCanvas("canvas", "canvas", WW, HH)
        canvas.SetBorderMode(0)
        canvas.SetLeftMargin(LL / WW)
        canvas.SetRightMargin(RR / WW)
        canvas.SetTopMargin(TT / HH)
        canvas.SetBottomMargin(BB / HH)
        canvas.SetTickx(0)
        canvas.SetTicky(0)
        # canvas.SetGrid()
    else:
        LL = 0.14 * WW
        canvas = ROOT.TCanvas("canvas", "canvas", WW, WW)
        canvas.SetBorderMode(0)
        canvas.SetLeftMargin(LL / WW)
        canvas.SetRightMargin(RR / WW)
        canvas.SetTopMargin(TT / HH)
        canvas.SetBottomMargin(TT / HH)
        canvas.SetTickx(0)
    canvas.SetFillColor(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)

    if do_bottom:
        topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.34, 1.00, 0.995)
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.20)
        topPad.SetRightMargin(0.04)
        topPad.SetBottomMargin(0.053)

        bottomPad = ROOT.TPad("bottomPad", "bottomPad", 0.00, 0.05, 1.00, 0.34)
        bottomPad.SetFillColor(10)
        bottomPad.SetTopMargin(0.036)
        bottomPad.SetLeftMargin(0.20)
        bottomPad.SetBottomMargin(0.35)
        bottomPad.SetRightMargin(0.04)

        topPad.Draw()
        bottomPad.Draw()
    else:
        topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.0, 1.00, 0.995)
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.20)
        topPad.SetRightMargin(0.04)
        topPad.SetBottomMargin(0.1)
        topPad.Draw()

    oplin = "linear"
    if useLogPlot:
        topPad.SetLogy()
        oplin = "log"

    topPad.cd()
    dumb = hist_total.Draw()
    del dumb
    histogramStack_mc = ROOT.THStack()
    print("list of processes considered and their integrals")
    linebin = []
    linebinW = []
    poslinebinW_X = []
    pos_linebinW_Y = []
    y0 = bin["cats_labels_height"]

    hist_template = template.Clone()
    for ii in xrange(1, hist_template.GetNbinsX() + 1):
        hist_template.SetBinContent(ii, 0)

    for kk, key in enumerate(dprocs.keys()):
        hist_rebin = hist_template.Clone()
        lastbin = 0  # for putting histograms from different bins in same plot side by side
        addlegend = True
        for cc, catcat in enumerate(catcats):
            if not cc == 0:
                addlegend = False
            if kk == 0:
                firstHisto = ROOT.TH1F()
            readFrom = str("%s/%s" % (folder, catcat))
            info_hist = stack_histo(
                hist_rebin,
                fin,
                readFrom,
                key,
                dprocs[key],
                divideByBinWidth,
                addlegend,
                lastbin,
                nbinscatlist[cc],
                normalize_X_original,
                firstHisto,
                era,
                legend1
            )
            yiels_list[catcat][key] = {
            "central" : round(info_hist["yield_cat"], round_yiels_list),
            "err" : round(info_hist["yield_cat_err"], round_yiels_list)
            }
            #round(info_hist["yield_cat"], round_yiels_list)
            lastbin += info_hist["lastbin"]
            if kk == 0:
                print(info_hist)
                print("info_hist[binEdge]", info_hist["binEdge"])
                if info_hist["binEdge"] > 0:
                    linebin += [
                        ROOT.TLine(info_hist["binEdge"], 0.0, info_hist["binEdge"], y0 * 1.1)
                    ]  # (legend_y0 + 0.05)*maxY
                x0 = float(lastbin - info_hist["labelPos"] - 1)
                sum_inX = 0.1950
                if len(catcat) > 2:
                    if len(catcat) == 3:
                        sum_inX = 5.85
                    else:
                        sum_inX = 4.0
                if len(catcat) == 0:
                    poslinebinW_X += [x0 - sum_inX]
                else:
                    poslinebinW_X += [bin["align_cats_labelsX"][cc]]
                pos_linebinW_Y += [y0]
        if (
            hist_rebin == 0
            or not hist_rebin.Integral() > 0
            or (info_hist["labelPos"] == 0 and not normalize_X_original)
        ):
            continue
        print(key, 0 if hist_rebin == 0 else hist_rebin.Integral())
        print("Stacking proocess %s, with yield %s " % (key, str(round(hist_rebin.Integral(), 2))))
        dumb = histogramStack_mc.Add(hist_rebin)
        del dumb

    dumb = hist_total.Draw("same")
    dumb = histogramStack_mc.Draw("hist,same")
    del dumb
    dumb = hist_total.Draw("e2,same")
    del dumb
    legend1.AddEntry(hist_total, "Uncertainty", "f")

    for line1 in linebin:
        line1.SetLineColor(1)
        line1.SetLineStyle(3)
        line1.Draw()

    for cc, cat in enumerate(bin["align_cats_labels"]):
        print("Draw label cat", cat, cc)
        sumBottom = 0
        for ccf, cf in enumerate(cat):
            linebinW = ROOT.TLatex()
            linebinW.DrawLatex(poslinebinW_X[cc], pos_linebinW_Y[cc] + sumBottom, cf)
            linebinW.SetTextFont(50)
            linebinW.SetTextAlign(12)
            linebinW.SetTextSize(0.05)
            linebinW.SetTextColor(1)
            if era == 0:
                sumBottom += -4.4
            else:
                sumBottom += -2.4

    ## draw signal
    hist_sig = [ROOT.TH1F() for _ in range(len(procs_plot_options_sig.keys()))]
    for kk, key in enumerate(procs_plot_options_sig.keys()):
        hist_sig_part = template.Clone()
        for cc, catcat in enumerate(catcats):
            ### make the single H stack entry
            sigs_to_stack = []
            readFrom = str("%s/%s" % (folder, catcat))
            fin.cd(readFrom)
            for key0 in ROOT.gDirectory.GetListOfKeys():
                obj_name = key0.GetName()
                if key in obj_name:
                    sigs_to_stack += [obj_name]
            print(catcat, key, "sigs_to_stack ", sigs_to_stack)

        for sig in sigs_to_stack :
            lastbin = 0
            for cc, catcat in enumerate(catcats):
                readFrom = str("%s/%s" % (folder, catcat))
                info_bin = process_total_histo(
                    hist_sig_part,
                    readFrom,
                    fin,
                    divideByBinWidth,
                    sig,
                    lastbin,
                    do_bottom,
                    labelX,
                    nbinscatlist[cc],
                    minY,
                    maxY,
                    totalBand=False,
                )
                lastbin += info_bin["allbins"]
                yiels_list[catcat][key] = {
                "central" : round(info_bin["yield_cat"], round_yiels_list),
                "err" : round(info_bin["yield_cat_err"], round_yiels_list)
                }
                if not hist_sig[kk].Integral() > 0:
                    hist_sig[kk] = hist_sig_part.Clone()
                else:
                    hist_sig[kk].Add(hist_sig_part)
                # print(catcat, key,  sig, lastbin, hist_sig_part.Integral(), hist_sig[kk].Integral())
                hist_sig[kk].Scale(procs_plot_options_sig[key]["scaleBy"])
                if "ggHH" in key and scale_GGF > 0 :
                    print("Scaling GGF signal by %d" % scale_GGF )
                    hist_sig[kk].Scale(scale_GGF)
                if "qqHH" in key and scale_VBF > 0 :
                    print("Scaling VBF signal by %d" % scale_VBF )
                    hist_sig[kk].Scale(scale_VBF)

    for kk, key in enumerate(procs_plot_options_sig.keys()):
        try:
            hist_sig[kk].Integral()
        except:
            print("A full signal list doesn't exist for %s" % key)
            continue
        hist_sig[kk].SetMarkerSize(0)
        hist_sig[kk].SetLineColor(procs_plot_options_sig[key]["color"])
        hist_sig[kk].SetFillStyle(procs_plot_options_sig[key]["fillStype"])
        hist_sig[kk].SetFillColorAlpha(procs_plot_options_sig[key]["color"], 0.40)
        hist_sig[kk].SetLineWidth(2)
        dumb = hist_sig[kk].Draw("hist,same")
        del dumb
        legend_signal = procs_plot_options_sig[key]["label"]
        if "ggHH" in key and scale_GGF > 0 and not scale_GGF == 1 :
            legend_signal = "%s X %i" % (legend_signal, scale_GGF)
        if "qqHH" in key and scale_VBF > 0 and not scale_VBF == 1 :
            legend_signal = "%s X %i" % (legend_signal, scale_VBF)
        legend1.AddEntry(hist_sig[kk], legend_signal, "f")

    if unblind:
        dumb = dataTGraph1.Draw("e1P,same")
        del dumb
    dumb = hist_total.Draw("axis,same")
    del dumb

    dumb = legend1.Draw("same")
    del dumb

    labels = addLabel_CMS_preliminary(era, do_bottom)
    for ll, label in enumerate(labels):
        if ll == 0:
            dumb = label.Draw("same")
            del dumb
        else:
            dumb = label.Draw()
            del dumb

    #################################
    if do_bottom:
        bottomPad.cd()
        print("doing bottom pad")
        hist_total_err = template.Clone()
        lastbin = 0
        for cc, catcat in enumerate(catcats):
            readFrom = str("%s/%s" % (folder, catcat))
            histtotal = hist_total
            lastbin += do_hist_total_err(hist_total_err, labelX, histtotal, minYerr, maxYerr, era)
            print(readFrom, lastbin)
        dumb = hist_total_err.Draw("e2")
        del dumb
        if unblind:
            dataTGraph2 = ROOT.TGraphAsymmErrors()
            lastbin = 0
            for cc, catcat in enumerate(catcats):
                readFrom = str("%s/%s" % (folder, catcat))
                readFromTot = str("%s/%s/%s" % (folder, catcat, name_total))
                histtotal = fin.Get(readFromTot)
                lastbin += err_data(
                    dataTGraph2,
                    hist_total,
                    dataTGraph1,
                    hist_total,
                    readFrom,
                    fin,
                    divideByBinWidth,
                    lastbin,
                )
            dumb = dataTGraph2.Draw("e1P,same")
            del dumb
        line = ROOT.TF1(
            "line", "0", hist_total_err.GetXaxis().GetXmin(), hist_total_err.GetXaxis().GetXmax()
        )
        line.SetLineStyle(3)
        line.SetLineColor(1)
        dumb = line.Draw("same")
        del dumb
        print("done bottom pad")
    ##################################

    optbin = "plain"
    if divideByBinWidth:
        optbin = "divideByBinWidth"

    savepdf = path + "_%s_%s_unblind%s" % (typeFit, oplin, unblind)
    if not do_bottom:
        savepdf = savepdf + "_noBottom"
    print("saving...", savepdf)
    dumb = canvas.SaveAs(savepdf + ".pdf")
    print("saved", savepdf + ".pdf")
    del dumb
    dumb = canvas.SaveAs(savepdf + ".png")
    print("saved", savepdf + ".png")
    del dumb
    canvas.IsA().Destructor(canvas)

    with open(savepdf + "_yield.json", 'w') as outfile : json.dump(yiels_list, outfile, sort_keys=True, indent=2)
    print("saved", savepdf + "_yield.json")

def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
    root = dct._OrderedDict__root
    first = root[1]

    if key in dct:
        link = dct._OrderedDict__map[key]
        link_prev, link_next, _ = link
        link_prev[1] = link_next
        link_next[0] = link_prev
        link[0] = root
        link[1] = first
        root[1] = first[0] = link
    else:
        root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
        dict_setitem(dct, key, value)


def GetNonZeroBins(template):
    nbins = 0
    for ii in xrange(1, template.GetXaxis().GetNbins() + 1):
        binContent_original = template.GetBinContent(ii)
        if binContent_original > 0:
            nbins += 1
    return nbins


def process_data_histo(
    template, data_cat, dataTGraph1, folder, fin, lastbin, histtotal, catbin, minY, maxY, divideByBinWidth
):
    readFrom = str("%s/data" % folder)
    dataTGraph = fin.Get(readFrom)
    #try :
    #    dataTGraph.Integral()
    #except :
    #    readFrom = str("%s/data_obs" % folder)
    #    dataTGraph = fin.Get(readFrom)
    print("adding", readFrom)
    allbins = catbin
    for ii in xrange(0, allbins):
        bin_width = 1.0
        if divideByBinWidth:
            bin_width = histtotal.GetXaxis().GetBinWidth(ii + 1)
        xp = ROOT.Double()
        yp = ROOT.Double()
        dataTGraph.GetPoint(ii, xp, yp)

        # do noot draw erroor bars on empty bins
        if yp == 0.0 :
            yp = ROOT.Double(-100)
            errYhigh = ROOT.Double(0)
            errYlow = ROOT.Double(0)
        else :
            errYhigh = dataTGraph.GetErrorYhigh(ii)
            errYlow = dataTGraph.GetErrorYlow(ii)

        dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin + 1), yp / bin_width)
        dataTGraph1.SetPointEYlow(ii + lastbin, errYlow / bin_width)
        dataTGraph1.SetPointEYhigh(ii + lastbin, errYhigh / bin_width)
        dataTGraph1.SetPointEXlow(ii + lastbin, template.GetBinWidth(ii + 1) / 2.0)
        dataTGraph1.SetPointEXhigh(ii + lastbin, template.GetBinWidth(ii + 1) / 2.0)

        data_cat += yp
    del dataTGraph
    dataTGraph1.SetMarkerColor(1)
    dataTGraph1.SetMarkerStyle(20)
    dataTGraph1.SetMarkerSize(0.8)
    dataTGraph1.SetLineColor(1)
    dataTGraph1.SetLineWidth(1)
    dataTGraph1.SetLineStyle(1)
    dataTGraph1.SetMinimum(minY)
    dataTGraph1.SetMaximum(maxY)
    #print("Data integral in %s = %d" % (folder, data_cat))
    return {
    "allbins" : allbins,
    "data_cat" : data_cat
    }


def process_total_histo(
    hist,
    folder,
    fin,
    divideByBinWidth,
    name_total,
    lastbin,
    do_bottom,
    labelX,
    catbins,
    minY,
    maxY,
    totalBand,
):
    total_hist_name = str("%s/%s" % (folder, name_total))
    total_hist = fin.Get(total_hist_name)
    allbins = catbins
    try:
        total_hist.Integral()
    except:
        print("Doesn't exist %s" % total_hist_name)
        return allbins

    hist.SetMarkerSize(0)
    hist.SetMarkerColor(16)
    hist.SetFillColorAlpha(12, 0.40)
    hist.SetLineWidth(0)
    if totalBand:
        print("Total band taken from %s" % total_hist_name)
        hist.SetMinimum(minY)
        hist.SetMaximum(maxY)
    for ii in xrange(1, allbins + 1):
        bin_width = 1.0
        if divideByBinWidth:
            bin_width = total_hist.GetXaxis().GetBinWidth(ii)
        hist.SetBinContent(ii + lastbin, 0.0003 + total_hist.GetBinContent(ii) / bin_width)
        hist.SetBinError(ii + lastbin, 0.0003 + total_hist.GetBinError(ii) / bin_width)
    if not hist.GetSumw2N():
        hist.Sumw2()
    if not do_bottom:
        hist.GetXaxis().SetTitle(labelX)
        hist.GetXaxis().SetTitleOffset(0.85)
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetXaxis().SetLabelSize(0.05)
        hist.GetYaxis().SetTitleOffset(1.5)
        hist.GetXaxis().SetLabelColor(1)
    else:
        hist.GetXaxis().SetTitleOffset(0.7)
        hist.GetYaxis().SetTitleOffset(1.2)
        hist.GetXaxis().SetLabelColor(10)
    hist.GetXaxis().SetTickLength(0.04)
    hist.GetYaxis().SetTitleSize(0.055)
    hist.GetYaxis().SetTickLength(0.04)
    hist.GetYaxis().SetLabelSize(0.050)
    #return allbins

    error_hist =  ROOT.Double()
    integral_histo = total_hist.IntegralAndError(0, total_hist.GetNbinsX()+1, error_hist, "")
    return {
        "allbins"   : allbins,
        "yield_cat" : integral_histo,
        "yield_cat_err" : error_hist
    }


def addLabel_CMS_preliminary(era, do_bottom):
    x0 = 0.2
    y0 = 0.953 if do_bottom else 0.935
    ypreliminary = 0.95 if do_bottom else 0.935
    xpreliminary = 0.12 if do_bottom else 0.085
    ylumi = 0.95 if do_bottom else 0.965
    xlumi = 0.65 if do_bottom else 0.73
    title_size_CMS = 0.0575 if do_bottom else 0.04
    title_size_Preliminary = 0.048 if do_bottom else 0.03
    title_size_lumi = 0.045 if do_bottom else 0.03
    label_cms = ROOT.TPaveText(x0, y0, x0 + 0.0950, y0 + 0.0600, "NDC")
    label_cms.AddText("CMS")
    label_cms.SetTextFont(61)
    label_cms.SetTextAlign(13)
    label_cms.SetTextSize(title_size_CMS)
    label_cms.SetTextColor(1)
    label_cms.SetFillStyle(0)
    label_cms.SetBorderSize(0)
    label_preliminary = ROOT.TPaveText(
        x0 + xpreliminary, y0 - 0.005, x0 + 0.0980 + 0.12, y0 + 0.0600 - 0.005, "NDC"
    )
    label_preliminary.AddText("Preliminary")
    label_preliminary.SetTextFont(50)
    label_preliminary.SetTextAlign(13)
    label_preliminary.SetTextSize(title_size_Preliminary)
    label_preliminary.SetTextColor(1)
    label_preliminary.SetFillStyle(0)
    label_preliminary.SetBorderSize(0)
    label_luminosity = ROOT.TPaveText(xlumi, y0 + 0.0035, xlumi + 0.0900, y0 + 0.040, "NDC")
    if era == 2016:
        lumi = "35.92"
    if era == 2017:
        lumi = "41.53"
    if era == 2018:
        lumi = "59.74"
    if era == 20172018:
        lumi = "101.27"
    if era == 0:
        lumi = "137"
    label_luminosity.AddText(lumi + " fb^{-1} (13 TeV)")
    label_luminosity.SetTextFont(42)
    label_luminosity.SetTextAlign(13)
    label_luminosity.SetTextSize(title_size_lumi)
    label_luminosity.SetTextColor(1)
    label_luminosity.SetFillStyle(0)
    label_luminosity.SetBorderSize(0)

    return [label_cms, label_preliminary, label_luminosity]


def stack_histo(
    hist_rebin_local,
    fin,
    folder,
    name,
    itemDict,
    divideByBinWidth,
    addlegend,
    lastbin,
    catbin,
    original,
    firstHisto,
    era,
    legend
):
    histo_name = str("%s/%s" % (folder, name))
    print("try find %s" % histo_name)
    hist = fin.Get(histo_name)
    allbins = catbin
    try:
        hist.Integral()
    except:
        print("Doesn't exist %s" % histo_name)
        return {
            "lastbin": allbins,
            "binEdge": lastbin - 0.5,
            "labelPos": 0 if not original == "none" else float(allbins / 2),
            "yield_cat" : 0.0,
            "yield_cat_err" : 0.0
        }
    if not firstHisto.Integral() > 0:
        firstHisto = hist.Clone()
        for ii in xrange(1, firstHisto.GetNbinsX() + 1):
            firstHisto.SetBinError(ii, 0.001)
            firstHisto.SetBinContent(ii, 0.001)
    hist_rebin_local.SetMarkerSize(0)
    hist_rebin_local.SetFillColor(itemDict["color"])
    if not itemDict["fillStype"] == 0:
        hist_rebin_local.SetFillStyle(itemDict["fillStype"])

    if "none" not in itemDict["label"] and addlegend:
        legend.AddEntry(hist_rebin_local, itemDict["label"], "l" if itemDict["color"] == 0 else "f")
    if itemDict["make border"] == True:
        hist_rebin_local.SetLineColor(1 if itemDict["color"] == 0 else itemDict["color"])
        hist_rebin_local.SetLineWidth(3 if itemDict["color"] == 0 else 1)

    else:
        hist_rebin_local.SetLineColor(itemDict["color"])
    for ii in xrange(1, allbins + 1):
        bin_width = 1.0
        if divideByBinWidth:
            bin_width = hist.GetXaxis().GetBinWidth(ii)
        ### remove and point bins with negative entry
        binContent_original = hist.GetBinContent(ii)
        binError2_original = hist.GetBinError(ii) ** 2
        if binContent_original < 0.0:
            binContent_modified = 0.0
            print("bin with negative entry: ", ii, "\t", binContent_original)
            binError2_modified = binError2_original + math.pow(
                (binContent_original - binContent_modified), 2
            )
            if not binError2_modified >= 0.0:
                print"Bin error negative!"
            hist_rebin_local.SetBinError(ii + lastbin, math.sqrt(binError2_modified) / bin_width)
            hist_rebin_local.SetBinContent(ii + lastbin, 0.0)
            print"binerror_original= ", binError2_original, "\t", "bincontent_original", "\t", binContent_original, "\t", "bincontent_modified", "\t", binContent_modified, "\t", "binerror= ", hist_rebin.GetBinError(
                ii
            )
        else:
            hist_rebin_local.SetBinError(ii + lastbin, hist.GetBinError(ii) / bin_width)
            hist_rebin_local.SetBinContent(ii + lastbin, hist.GetBinContent(ii) / bin_width)
    if not hist.GetSumw2N():
        hist.Sumw2()

    error_hist =  ROOT.Double()
    integral_histo = hist.IntegralAndError(0, hist.GetNbinsX()+1, error_hist, "")

    return {
        "lastbin": allbins,
        "binEdge": hist.GetXaxis().GetBinLowEdge(lastbin)
        + hist.GetXaxis().GetBinWidth(lastbin)
        - 0.5,  # if lastbin > 0 else 0
        "labelPos": float(allbins / 2),
        "yield_cat" : integral_histo,
        "yield_cat_err" : error_hist
    }


def do_hist_total_err(hist_total_err, labelX, total_hist, minBottom, maxBottom, era):
    allbins = total_hist.GetNbinsX()  # GetNonZeroBins(total_hist)
    hist_total_err.GetYaxis().SetTitle("#frac{Data - Expectation}{Expectation}")
    hist_total_err.GetXaxis().SetTitleOffset(1.25)
    hist_total_err.GetYaxis().SetTitleOffset(1.0)
    hist_total_err.GetXaxis().SetTitleSize(0.14)
    hist_total_err.GetYaxis().SetTitleSize(0.075)
    hist_total_err.GetYaxis().SetLabelSize(0.105)
    hist_total_err.GetXaxis().SetLabelSize(0.10)
    hist_total_err.GetYaxis().SetTickLength(0.04)
    hist_total_err.GetXaxis().SetLabelColor(1)
    hist_total_err.GetXaxis().SetTitle(labelX)
    hist_total_err.SetMarkerSize(0)
    hist_total_err.SetFillColorAlpha(12, 0.40)
    hist_total_err.SetLineWidth(0)
    if era == 0:
        minBottom = minBottom  # *3/2
        maxBottom = maxBottom
    hist_total_err.SetMinimum(minBottom)
    hist_total_err.SetMaximum(maxBottom)
    for bin in xrange(0, allbins):
        hist_total_err.SetBinContent(bin + 1, 0)
        if total_hist.GetBinContent(bin + 1) > 0.0:
            hist_total_err.SetBinError(
                bin + 1, total_hist.GetBinError(bin + 1) / total_hist.GetBinContent(bin + 1)
            )
    return allbins


def err_data(dataTGraph1, template, dataTGraph, histtotal, folder, fin, divideByBinWidth, lastbin):
    print(" do unblided bottom pad")
    allbins = histtotal.GetXaxis().GetNbins()  # GetNonZeroBins(histtotal)
    print("allbins", allbins)
    for ii in xrange(0, allbins):
        bin_width = 1.0
        if divideByBinWidth:
            bin_width = histtotal.GetXaxis().GetBinWidth(ii + 1)
        if histtotal.GetBinContent(ii + 1) == 0:
            continue
        dividend = histtotal.GetBinContent(ii + 1) * bin_width
        xp = ROOT.Double()
        yp = ROOT.Double()
        dataTGraph.GetPoint(ii, xp, yp)
        if yp > 0:
            if dividend > 0:
                dataTGraph1.SetPoint(
                    ii + lastbin, template.GetBinCenter(ii + lastbin + 1), yp / dividend - 1
                )
            else:
                dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin + 1), -1.0)
        else:
            dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin + 1), -100.0)
        dataTGraph1.SetPointEYlow(ii + lastbin, dataTGraph.GetErrorYlow(ii) / dividend)
        dataTGraph1.SetPointEYhigh(ii + lastbin, dataTGraph.GetErrorYhigh(ii) / dividend)
        dataTGraph1.SetPointEXlow(ii + lastbin, template.GetBinWidth(ii + 1) / 2.0)
        dataTGraph1.SetPointEXhigh(ii + lastbin, template.GetBinWidth(ii + 1) / 2.0)
    dataTGraph1.SetMarkerColor(1)
    dataTGraph1.SetMarkerStyle(20)
    dataTGraph1.SetMarkerSize(0.8)
    dataTGraph1.SetLineColor(1)
    dataTGraph1.SetLineWidth(1)
    dataTGraph1.SetLineStyle(1)
    return allbins


if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--plot_options_dict",
        dest="plot_options_dict",
        help="Dictionary with list of bins to plot and general options",
        #default="no",
    )
    #parser.add_argument(
    #    "--plot_options_full",
    #    dest="plot_options_full",
    #    help="Dictionary with list of channels with its respective bins to plot and general options",
    #    default="no",
    #)
    parser.add_argument(
        "--output_folder", dest="output_folder", help="Where the plots will be saved"
    )
    parser.add_argument(
        "--unblind", action="store_true", dest="unblind", help="Draw data", default=False
    )
    parser.add_argument(
        "--doPostFit",
        action="store_true",
        dest="doPostFit",
        help="Take shapes from postfit, if not added will take prefit shapes.",
        default=False,
    )
    parser.add_argument(
        "--not_do_bottom",
        action="store_true",
        dest="not_do_bottom",
        help="Do not do bottom pad.",
        default=False,
    )
    parser.add_argument(
        "--overwrite_era",
        dest="overwrite_era",
        help="If a value is given it will replace all instances of 'ERA' in the dictionary with the given value. Values can be 2016, 2017, 2018, 20172018 or all. If the value is all it will loop on {2016, 2017, 2018}",
        default="no",
    )
    parser.add_argument(
        "--overwrite_fitdiag",
        dest="overwrite_fitdiag",
        help="If a value is given it will replace all instances of 'PATH_FITDIAGNOSIS' in the dictionary with the given value.",
        default="no",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        help="If a value is given in the form \"{'TEXT':'newtext'}\" it will replace all instances of 'TEXT' in the dictionary with 'newtext'. Cases are separated by virgula, eg \"{'TEXT':'newtext','TEXT2':'newtext2'}\". A .json file with multiple entries can also be entered. It superseeds the option overwrite_era = all",
        default="no",
    )
    args = parser.parse_args()

    unblind = args.unblind
    doPostFit = args.doPostFit
    do_bottom = not args.not_do_bottom
    divideByBinWidth = False
    output_folder = args.output_folder
    overwrite_era = args.overwrite_era
    overwrite_fitdiag = args.overwrite_fitdiag
    overwrite = args.overwrite
    plot_options_dict = args.plot_options_dict

    options_dat = os.path.normpath(plot_options_dict)
    print("Reading plot options from %s" % options_dat)

    if not overwrite=="no" :
        if not ".json" in overwrite :
            # interpret the command line
            modifications = {"plot1" :  eval(str(overwrite)) }
        else :
            with open(overwrite) as ff : modifications = json.load(ff)
    else :
        if overwrite_era=="all" :
            modifications = {"plot1" : {'ERA' : 2016}, "plot2" : {'ERA' : 2017}, "plot3" : {'ERA' : 2018}}
        else :
            modifications = {"plots1" : {'NONE' : 'NONE'}}

    for item_modify in modifications :
        this_plot = modifications[item_modify]
        print(this_plot)

        if overwrite_era=="no" and overwrite_fitdiag=="no" and overwrite=="no":
            with open(options_dat) as ff : info_bin = json.load(ff)
        else :
            dict_for_era = options_dat.replace(os.path.dirname(options_dat), output_folder).replace(os.path.basename(options_dat), "temp_" + os.path.basename(options_dat))

            print("Replacing strings in a new dictionary %s" % dict_for_era )
            for key_modify in this_plot :
                print(key_modify, this_plot[key_modify])
            fin = open(options_dat, "rt")
            fout = open(dict_for_era, "wt")

            for line in fin :
                if not overwrite_era=="no" and not overwrite_era=="all" :
                    line = line.replace('ERA', overwrite_era)
                if not overwrite_fitdiag=="no"  :
                    line = line.replace('PATH_FITDIAGNOSIS', overwrite_fitdiag)
                for key_modify in this_plot :
                    line = line.replace(key_modify, str(this_plot[key_modify]))
                fout.write(line)
            fin.close()
            fout.close()

            with open(dict_for_era) as ff : info_bin = json.load(ff)

            # rename the file to the plot name for safekeeping
            name_plot = "plot_" + list(info_bin.keys())[0]
            # copy the dict files for processes options for each plot to the output_folder for keepsafe
            copyfile(dict_for_era, dict_for_era.replace("temp", name_plot))
            for key_bin in info_bin :
                for pp in [str(info_bin[key_bin]["procs_plot_options_bkg"]), str(info_bin[key_bin]["procs_plot_options_sig"])] :
                    pp = pp.replace("$DHI_DATACARDS_RUN2", os.getenv('DHI_DATACARDS_RUN2'))
                    if str(pp).startswith("/") :
                        copyfile(pp, os.path.join(output_folder, os.path.basename(pp)))
                    else :
                        copyfile( os.path.join(os.path.dirname(options_dat), pp),  os.path.join(output_folder, pp))

        for key, bin in info_bin.iteritems():

            normalize_X_original = True
            if bin["datacard_original"] == "none":
                normalize_X_original = False

            procs_plot = str(bin["procs_plot_options_sig"]).replace("$DHI_DATACARDS_RUN2", os.getenv('DHI_DATACARDS_RUN2'))
            file_sig_options = procs_plot if procs_plot.startswith("/") else options_dat.replace(os.path.basename(options_dat), procs_plot)

            procs_plot = str(bin["procs_plot_options_bkg"]).replace("$DHI_DATACARDS_RUN2", os.getenv('DHI_DATACARDS_RUN2'))
            file_bkg_options = procs_plot if procs_plot.startswith("/") else options_dat.replace(os.path.basename(options_dat), procs_plot)

            data_dir = bin["fitdiagnosis"]
            print("Drawing %s" % key)
            create_postfit_plots(
                path="%s/plot_%s" % (output_folder, key),
                fit_diagnostics_path=data_dir,
                normalize_X_original=normalize_X_original,
                doPostFit=doPostFit,
                divideByBinWidth=divideByBinWidth,
                bin=bin,
                binToRead=key,
                unblind=unblind,
                options_dat=options_dat,
                file_sig_options=file_sig_options,
                file_bkg_options=file_bkg_options
            )
