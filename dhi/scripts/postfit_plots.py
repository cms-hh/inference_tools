#!/usr/bin/env python
# coding: utf-8

"""
Script to extract and plot shapes from a ROOT file create by combine's FitDiagnostics.
"""

import os
from collections import OrderedDict

from dhi.util import import_ROOT
from dhi.util_shapes_plot import test_print, GetNonZeroBins, process_data_histo, process_total_histo, addLabel_CMS_preliminary, stack_histo, do_hist_total_err, err_data, ordered_dict_prepend


def get_parser():
    """
    Creates a new argument parser.
    """
    #from optparse import OptionParser
    #parser = OptionParser()
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--plot_options_dict",
        #type="string",
        metavar="FILE",
        dest="plot_options_dict",
        help="Dictionary with list of bins to plot and options",
        )
    parser.add_argument(
        "--output_folder",
        #type="string",
        metavar="FILE",
        dest="output_folder",
        help="Where the plots will be saved"
        )
    parser.add_argument(
        "--unblind",
        action="store_true",
        dest="unblind",
        help="Draw data",
        default=False
        )
    parser.add_argument(
        "--doPostFit",
        action="store_true",
        dest="doPostFit",
        help="Take shapes from postfit, if not added will take prefit shapes.",
        default=False
        )
    parser.add_argument(
        "--not_do_bottom",
        action="store_true",
        dest="not_do_bottom",
        help="Do not do bottom pad.",
        default=False)
    return parser

def create_postfit_plots(
    path,
    fit_diagnostics_path,
    normalize_X_original,
    doPostFit,
    divideByBinWidth,
    bin,
    binToRead,
    unblind
    ):
    ROOT = import_ROOT()
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    minY=bin["minY"]
    maxY=bin["maxY"]
    minYerr=bin["minYerr"]
    maxYerr=bin["maxYerr"]
    useLogPlot=bin["useLogPlot"]
    era=bin["era"]
    labelX=bin["labelX"]
    header_legend=bin["header_legend"]
    datacard_original=bin["datacard_original"]
    bin_name_original=bin["bin_name_original"]
    number_columns_legend=bin["number_columns_legend"]
    procs_plot_options_sig=bin["procs_plot_options_sig"]

    typeFit = None
    if doPostFit :
        folder = "shapes_fit_s"
        folder_data = "shapes_fit_s"
        typeFit = "postfit"
    else :
        folder = "shapes_prefit"
        folder_data = "shapes_prefit"
        typeFit = "prefit"

    name_total = "total" # total_background ? TOFIX

    if normalize_X_original :
        fileOrig = datacard_original
        print ("template on ", fileOrig)
    else :
        fileOrig = fit_diagnostics_path

    print("reading shapes from: ", fit_diagnostics_path)
    fin = ROOT.TFile(fit_diagnostics_path, "READ")
    print("read shapes from: ")

    labelY = "Events"
    if divideByBinWidth : labelY = "Events / bin width"

    if not doPostFit :
        header_legend = header_legend+", \n"+typeFit
    else :
        header_legend = header_legend+", #mu(t#bar{t}H)=#hat{#mu}"

    # list of folders to read from
    catcats =     bin["align_cats"]

    dprocs=bin["procs_plot_options_bkg"]
    # add stack of single H as second
    singleH = [
        "ggH_hbb", "ggH_hgg", "ggH_hmm", "ggH_htt", "ggH_hww", "ggH_hzz",
        "qqH_hbb", "qqH_hgg", "qqH_hmm", "qqH_htt", "qqH_hww", "qqH_hzz",
        "ttH_hbb", "ttH_hgg", "ttH_hmm", "ttH_htt", "ttH_hww", "ttH_hzz",
        "WH_hbb", "WH_hgg", "WH_hmm", "WH_htt", "WH_hww", "WH_hzz",
        "ZH_hbb", "ZH_hgg", "ZH_hmm", "ZH_htt", "ZH_hww", "ZH_hzz",
        "TH_hbb", "TH_hgg", "TH_hmm", "TH_htt", "TH_hww", "TH_hzz",
        "tHq_hbb", "tHq_hgg", "tHq_hmm", "tHq_htt", "tHq_hww", "tHq_hzz",
        "tHW_hbb", "tHW_hgg", "tHW_hmm", "tHW_htt", "tHW_hww", "tHW_hzz",
        "VH_hww", "VH_hgg", "VH_hmm", "VH_htt", "VH_hww", "VH_hzz",
        "TH", "VH", "TTH", "ttVH"
        ]
    ## make a list without the major
    countOnce = 0
    for sh in singleH :
        if countOnce == 0 :
            hist = fin.Get(folder + "/" + catcats[0] + "/" + sh )
            try :
                hist.Integral()
            except :
                continue
            countOnce = 1
            label_singleH = "single H"
            print("Add single H legend (proc %s)" % sh)
        else :
            label_singleH = "none"
        ordered_dict_prepend(dprocs, sh, {"color" : 226, "fillStype"   : 1001, "label" : label_singleH          , "make border" : False})


    print ("will draw processes", list(dprocs.keys()))

    if normalize_X_original :
        fileorriginal = ROOT.TFile(fileOrig, "READ")
        histRead = list(dprocs.keys())[0] # "TTH"
        readFromOriginal = "%s/%s" % (bin_name_original, histRead) if not bin_name_original == "none" else histRead
        print ("original readFrom ", readFromOriginal)
        template = fileorriginal.Get( readFromOriginal )
        template.GetYaxis().SetTitle(labelY)
        template.SetTitle(" ")
        nbinscatlist = [template.GetNbinsX()]
    else :
        print("Drawing: ", catcats)
        nbinstotal = 0
        nbinscatlist = []
        for catcat in catcats :
            readFrom = folder + "/" + catcat
            hist = fin.Get(readFrom + "/" + name_total )
            print ("reading shapes", readFrom + "/" + name_total)
            print (hist.Integral())
            nbinscat =  GetNonZeroBins(hist)
            nbinscatlist += [nbinscat]
            print (readFrom, nbinscat)
            nbinstotal += nbinscat
        template = ROOT.TH1F("my_hist", "", nbinstotal, 0 - 0.5 , nbinstotal - 0.5)
        template.GetYaxis().SetTitle(labelY)
        print (nbinscatlist)

    legend1 = ROOT.TLegend(0.2400, 0.645, 0.9450, 0.910)
    legend1.SetNColumns(number_columns_legend)
    legend1.SetFillStyle(0)
    legend1.SetBorderSize(0)
    legend1.SetFillColor(10)
    legend1.SetTextSize(0.040 if do_bottom else 0.03)
    legend1.SetHeader(header_legend)
    header = legend1.GetListOfPrimitives().First()
    header.SetTextSize(.05 if do_bottom else 0.04)
    header.SetTextColor(1)
    header.SetTextFont(62)

    dataTGraph1 = ROOT.TGraphAsymmErrors()
    if unblind :
        dataTGraph1.Set(template.GetXaxis().GetNbins())
        lastbin = 0
        for cc, catcat in enumerate(catcats) :
            readFrom = folder + "/" + catcat
            print( " histtotal ", readFrom + "/" + name_total )
            histtotal = fin.Get(readFrom + "/" + name_total )
            lastbin += process_data_histo(
                template,
                dataTGraph1,
                readFrom,
                fin,
                lastbin,
                histtotal,
                nbinscatlist[cc],
                minY, maxY,
                divideByBinWidth
                )
        dataTGraph1.Draw()
        legend1.AddEntry(dataTGraph1, "Data", "p")

    lastbin = 0
    hist_total = template.Clone()
    for cc, catcat in enumerate(catcats) :
        readFrom = folder + "/" + catcat
        print ("read the hist with total uncertainties", readFrom, catcat)
        lastbin += process_total_histo(
            hist_total,
            readFrom,
            fin,
            divideByBinWidth,
            name_total,
            lastbin,
            do_bottom,
            labelX,
            nbinscatlist[cc],
            minY, maxY,
            totalBand=True
            )
    print("hist_total", hist_total.Integral())

    ## declare canvases sizes accordingly
    WW = 600
    HH  = 700
    TT = 0.08*HH
    BB = 0.12*HH
    RR = 0.04*WW
    if do_bottom :
        LL = 0.13*WW
        canvas = ROOT.TCanvas("canvas", "canvas", WW, HH)
        canvas.SetBorderMode(0)
        canvas.SetLeftMargin( LL/WW )
        canvas.SetRightMargin( RR/WW )
        canvas.SetTopMargin( TT/HH )
        canvas.SetBottomMargin( BB/HH )
        canvas.SetTickx(0)
        canvas.SetTicky(0)
        #canvas.SetGrid()
    else :
        LL = 0.14*WW
        canvas = ROOT.TCanvas("canvas", "canvas", WW, WW)
        canvas.SetBorderMode(0)
        canvas.SetLeftMargin( LL/WW )
        canvas.SetRightMargin( RR/WW )
        canvas.SetTopMargin( TT/HH )
        canvas.SetBottomMargin( TT/HH )
        canvas.SetTickx(0)
    canvas.SetFillColor(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)

    if do_bottom :
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
    else :
        topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.0, 1.00, 0.995)
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.20)
        topPad.SetRightMargin(0.04)
        topPad.SetBottomMargin(0.1)
        topPad.Draw()

    oplin = "linear"
    if useLogPlot :
        topPad.SetLogy()
        oplin = "log"

    topPad.cd()
    dumb = hist_total.Draw()
    del dumb
    histogramStack_mc = ROOT.THStack()
    print ("list of processes considered and their integrals")
    linebin = []
    linebinW = []
    poslinebinW_X = []
    pos_linebinW_Y = []
    y0 = bin["cats_labels_height"]
    for kk, key in  enumerate(dprocs.keys()) :
        hist_rebin = template.Clone()
        lastbin = 0 # for putting histograms from different bins in same plot side by side
        addlegend = True
        for cc, catcat in enumerate(catcats) :
            if not cc == 0 :
                addlegend = False
            if kk == 0 :
                firstHisto = ROOT.TH1F()
            readFrom = folder + "/" + catcat
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
            lastbin += info_hist["lastbin"]
            if kk == 0 :
                print (info_hist)
                print ("info_hist[binEdge]", info_hist["binEdge"])
                if info_hist["binEdge"] > 0 :
                    linebin += [ROOT.TLine(info_hist["binEdge"], 0., info_hist["binEdge"], y0*1.1)] # (legend_y0 + 0.05)*maxY
                x0 = float(lastbin - info_hist["labelPos"] - 1)
                sum_inX = 0.1950
                if len(catcat) > 2 :
                    if len(catcat) == 3 :
                        sum_inX = 5.85
                    else :
                        sum_inX = 4.0
                if len(catcat) == 0 :
                    poslinebinW_X += [x0 - sum_inX]
                else :
                    poslinebinW_X += [bin["align_cats_labelsX"][cc]]
                pos_linebinW_Y += [y0]
        if hist_rebin == 0 or not hist_rebin.Integral() > 0 or (info_hist["labelPos"] == 0 and not normalize_X_original  )  :
            continue
        print (key,  0 if hist_rebin == 0 else hist_rebin.Integral() )
        print("Stacking proocess %s, with yield %s " % (key, str(round(hist_rebin.Integral(),2))))
        dumb = histogramStack_mc.Add(hist_rebin)
        del dumb

    dumb = hist_total.Draw("same")
    dumb = histogramStack_mc.Draw("hist,same")
    del dumb
    dumb = hist_total.Draw("e2,same")
    del dumb

    for line1 in linebin :
        line1.SetLineColor(1)
        line1.SetLineStyle(3)
        line1.Draw()

    for cc, cat in enumerate(bin["align_cats_labels"]) :
        print ("Draw label cat", cat, cc)
        sumBottom = 0
        for ccf, cf in enumerate(cat) :
            linebinW = ROOT.TLatex()
            linebinW.DrawLatex(poslinebinW_X[cc], pos_linebinW_Y[cc] + sumBottom, cf)
            linebinW.SetTextFont(50)
            linebinW.SetTextAlign(12)
            linebinW.SetTextSize(0.05)
            linebinW.SetTextColor(1)
            if era == 0 :
                sumBottom += -4.4
            else :
                sumBottom += -2.4

    ## draw signal
    hist_sig = [ROOT.TH1F() for _ in range(len(procs_plot_options_sig.keys()))]
    for kk, key in  enumerate(procs_plot_options_sig.keys()) :
        hist_sig_part = template.Clone()
        for cc, catcat in enumerate(catcats) :
            ### make the single H stack entry
            sigs_to_stack = []
            fin.cd(folder + "/" + catcat )
            for key0 in ROOT.gDirectory.GetListOfKeys() :
                obj_name = key0.GetName()
                if key in obj_name :
                    sigs_to_stack += [obj_name]
            print(catcat, key,  "sigs_to_stack ", sigs_to_stack)


        for sig in sigs_to_stack : #procs_plot_options_sig[key]["processes"] :
            lastbin = 0
            for cc, catcat in enumerate(catcats) :
                readFrom = folder + "/" + catcat
                lastbin += process_total_histo(
                    hist_sig_part,
                    readFrom,
                    fin,
                    divideByBinWidth,
                    sig,
                    lastbin,
                    do_bottom,
                    labelX,
                    nbinscatlist[cc],
                    minY, maxY,
                    totalBand=False
                    )
                if not hist_sig[kk].Integral() > 0 :
                    hist_sig[kk] = hist_sig_part.Clone()
                else :
                    hist_sig[kk].Add(hist_sig_part)
                #print(catcat, key,  sig, lastbin, hist_sig_part.Integral(), hist_sig[kk].Integral())
                hist_sig[kk].Scale(procs_plot_options_sig[key]["scaleBy"])

    for kk, key in  enumerate(procs_plot_options_sig.keys()) :
        try  :
            hist_sig[kk].Integral()
        except :
            print ("A full signal list doesn't exist for %s" % key)
            continue
        hist_sig[kk].SetMarkerSize(0)
        hist_sig[kk].SetLineColor(procs_plot_options_sig[key]["color"])
        hist_sig[kk].SetFillStyle(procs_plot_options_sig[key]["fillStype"])
        hist_sig[kk].SetFillColorAlpha(procs_plot_options_sig[key]["color"], 0.40)
        hist_sig[kk].SetLineWidth(2)
        dumb = hist_sig[kk].Draw("hist,same")
        del dumb
        legend1.AddEntry(hist_sig[kk], procs_plot_options_sig[key]["label"], "f")

    if unblind :
        dumb = dataTGraph1.Draw("e1P,same")
        del dumb
    dumb = hist_total.Draw("axis,same")
    del dumb

    dumb = legend1.Draw("same")
    del dumb

    labels = addLabel_CMS_preliminary(era, do_bottom)
    for ll, label in enumerate(labels) :
        if ll == 0 :
            dumb = label.Draw("same")
            del dumb
        else :
            dumb = label.Draw()
            del dumb

    #################################
    if do_bottom :
        bottomPad.cd()
        print ("doing bottom pad")
        hist_total_err = template.Clone()
        lastbin = 0
        for cc, catcat in enumerate(catcats) :
            readFrom = folder + "/" + catcat
            histtotal = hist_total
            lastbin += do_hist_total_err(
                hist_total_err,
                labelX, histtotal  ,
                minYerr,
                maxYerr,
                era
                )
            print (readFrom, lastbin)
        dumb = hist_total_err.Draw("e2")
        del dumb
        if unblind :
            dataTGraph2 = ROOT.TGraphAsymmErrors()
            lastbin = 0
            for cc, catcat in enumerate(catcats) :
                readFrom = folder + "/" + catcat
                histtotal = fin.Get(readFrom + "/" + name_total)
                lastbin += err_data(
                    dataTGraph2,
                    hist_total,
                    dataTGraph1,
                    hist_total,
                    readFrom,
                    fin,
                    divideByBinWidth,
                    lastbin
                    )
            dumb = dataTGraph2.Draw("e1P,same")
            del dumb
        line = ROOT.TF1("line", "0", hist_total_err.GetXaxis().GetXmin(), hist_total_err.GetXaxis().GetXmax())
        line.SetLineStyle(3)
        line.SetLineColor(1)
        dumb = line.Draw("same")
        del dumb
        print ("done bottom pad")
    ##################################

    optbin = "plain"
    if divideByBinWidth :
        optbin = "divideByBinWidth"

    savepdf = path +  "_%s_%s_unblind%s" % (typeFit, oplin, unblind)
    if not do_bottom :
        savepdf = savepdf + "_noBottom"
    print ("saving...", savepdf )
    dumb = canvas.SaveAs(savepdf + ".pdf")
    print ("saved", savepdf + ".pdf")
    del dumb
    dumb = canvas.SaveAs(savepdf + ".png")
    print ("saved", savepdf + ".png")
    del dumb
    canvas.IsA().Destructor(canvas)

if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--plot_options_dict",
        dest="plot_options_dict",
        help="Dictionary with list of bins to plot and options",
        )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        help="Where the plots will be saved"
        )
    parser.add_argument(
        "--unblind",
        action="store_true",
        dest="unblind",
        help="Draw data",
        default=False
        )
    parser.add_argument(
        "--doPostFit",
        action="store_true",
        dest="doPostFit",
        help="Take shapes from postfit, if not added will take prefit shapes.",
        default=False
        )
    parser.add_argument(
        "--not_do_bottom",
        action="store_true",
        dest="not_do_bottom",
        help="Do not do bottom pad.",
        default=False)
    args = parser.parse_args()

    unblind =  args.unblind
    doPostFit = args.doPostFit
    do_bottom = not args.not_do_bottom
    divideByBinWidth = False
    output_folder = args.output_folder

    options_dat = args.plot_options_dict
    print("Reading plot options from %s" % options_dat)
    info_bin = eval(open(options_dat, 'r').read())

    for key, bin in info_bin.iteritems() :

        normalize_X_original = True
        if bin["bin_name_original"] == "none" :
            normalize_X_original = False

        data_dir = bin["fitdiagnosis"]
        print("Drawing %s" % key)
        create_postfit_plots(
            path="%s/plot_%s" % (output_folder, key) ,
            fit_diagnostics_path=data_dir,
            normalize_X_original=normalize_X_original,
            doPostFit=doPostFit,
            divideByBinWidth=divideByBinWidth,
            bin=bin,
            binToRead=key,
            unblind=unblind
        )
