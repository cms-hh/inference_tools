#!/usr/bin/env python
# coding: utf-8

"""
Script to extract and plot shapes from a ROOT file create by combine's FitDiagnostics.
"""

import os
from collections import OrderedDict

from dhi.util import import_ROOT
from dhi.util_shapes_plot import test_print, GetNonZeroBins, rebin_data, rebin_total, addLabel_CMS_preliminary, rebin_hist


## input: fitDiagnosis + the original datacard.root
## step 1: subcategories should be merged using proper naming convention for bins, eg
"""
combineCards.py \
bbWW_SL=datacard_hh_bb1l_hh_bb1l_cat_jet_2BDT_Wjj_simple_SM_Res_allReco_bbWW_nonresNLO_none_75_multisig_2017.txt \
bbWW_DL=datacard_hh_bb2l_hh_bb2l_OS_SM_plainVars_inclusive_bbWW_nonresNLO_none_45_noSL_2017.txt>\
datacard_SL_DL_bbWW_nonresNLO_none_45_75_allSig_2017_noSingleH_naming.txt
"""
## --> do use the era in the bin name
## step 2: user enter dictionaries with options

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
    #test_print()
    ## dictionary with subcategories / processes /
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
    nameLabel=bin["nameLabel"]
    datacard_original=bin["datacard_original"]
    bin_name_original=bin["bin_name_original"]

    typeFit = None
    if doPostFit :
        folder = "shapes_fit_s"
        folder_data = "shapes_fit_s" # it is a TGraphAsymmErrors, not histogram
        typeFit = "postfit"
    else :
        folder = "shapes_prefit"
        folder_data = "shapes_prefit" # it is a TGraphAsymmErrors, not histogram
        typeFit = "prefit"

    name_total = "total" # total_background ?

    binToReadOriginal=bin_name_original # XandaFix
    label_head=nameLabel
    shapes_input=fit_diagnostics_path
    fromHavester=False

    if normalize_X_original :
        fileOrig = datacard_original
        print ("template on ", fileOrig)
    else :
        fileOrig = shapes_input

    print("reading shapes from: ", shapes_input)
    fin = [ROOT.TFile(shapes_input, "READ")]
    ## decide if sum years or do by year

    labelY = "Events"
    if divideByBinWidth :
        labelY = "Events / bin width"

    if not doPostFit :
        label_head = label_head+", \n"+typeFit
    else :
        label_head = label_head+", #mu(t#bar{t}H)=#hat{#mu}"

    dprocs=bin["procs_plot_options"]
    print ("will draw processes", list(dprocs.keys()))

    if not binToRead == "none" :
        catcats =  [binToRead]
    else :
        #catcats = getCats(folder, fin[0], options.fromHavester)
        catcats = getCats(folder, fin[0], False)

    if normalize_X_original :

        readFrom = binToReadOriginal
        print ("original readFrom ", readFrom)
        fileorriginal = ROOT.TFile(fileOrig, "READ")

        histRead = list(dprocs.keys())[0] # "TTH"
        template = fileorriginal.Get( "%s/%s" % (readFrom, histRead) )
        template.GetYaxis().SetTitle(labelY)
        template.SetTitle(" ")
        nbinscatlist = [template.GetNbinsX()]
        datahist = fileorriginal.Get(readFrom + "data_obs")
    else :
        print("Drawing: ", catcats)
        nbinstotal = 0
        nbinscatlist = []
        for catcat in catcats :
            readFrom = folder + "/" + catcat
            hist = fin[0].Get(readFrom + "/" + name_total )
            print ("reading shapes", readFrom + "/" + name_total)
            print (hist.Integral())
            nbinscat =  GetNonZeroBins(hist)
            nbinscatlist += [nbinscat]
            print (readFrom, nbinscat)
            nbinstotal += nbinscat
            datahist = fin[0].Get(readFrom + "/data")
        template = ROOT.TH1F("my_hist", "", nbinstotal, 0 - 0.5 , nbinstotal - 0.5)
        template.GetYaxis().SetTitle(labelY)
        print (nbinscatlist)

    legend_y0 = 0.645
    legend1 = ROOT.TLegend(0.2400, legend_y0, 0.9450, 0.910)
    legend1.SetNColumns(3)
    legend1.SetFillStyle(0)
    legend1.SetBorderSize(0)
    legend1.SetFillColor(10)
    legend1.SetTextSize(0.040)
    legend1.SetHeader(label_head)
    header = legend1.GetListOfPrimitives().First()
    header.SetTextSize(.05)
    header.SetTextColor(1)
    header.SetTextFont(62)

    dataTGraph1 = ROOT.TGraphAsymmErrors()
    """if unblind :
        dataTGraph1.Set(template.GetXaxis().GetNbins())
        lastbin = 0
        for cc, catcat in enumerate(catcats) :
            readFrom = folder + "/" + catcat
            print( " histtotal ", readFrom + "/" + name_total )
            histtotal = fin[0].Get(readFrom + "/" + name_total )
            lastbin += rebin_data(
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
        legend1.AddEntry(dataTGraph1, "Data", "p")"""

    lastbin = 0
    hist_total = template.Clone()
    for cc, catcat in enumerate(catcats) :
        readFrom = folder + "/" + catcat
        print ("read the hist with total uncertainties", readFrom, catcat)
        lastbin += rebin_total(
            hist_total,
            readFrom,
            fin,
            divideByBinWidth,
            name_total,
            lastbin,
            do_bottom,
            labelX,
            nbinscatlist[cc],
            minY, maxY
            )
    print("hist_total", hist_total.Integral())

    ## declare canvases sizes accordingly
    canvas = ROOT.TCanvas("canvas", "canvas", 900, 900)
    """
    if do_bottom :
        canvas = ROOT.TCanvas("canvas", "canvas", 600, 1500)
    else :
        canvas = ROOT.TCanvas("canvas", "canvas", 900, 900)"""
    canvas.SetFillColor(10)
    canvas.SetBorderSize(2)
    dumb = canvas.Draw()
    del dumb

    """if do_bottom :
        topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.34, 1.00, 0.995)
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.20)
        topPad.SetBottomMargin(0.053)
        topPad.SetRightMargin(0.04)
        if bin["useLogPlot"]:
            topPad.SetLogy()

        bottomPad = ROOT.TPad("bottomPad", "bottomPad", 0.00, 0.05, 1.00, 0.34)
        bottomPad.SetFillColor(10)
        bottomPad.SetTopMargin(0.036)
        bottomPad.SetLeftMargin(0.20)
        bottomPad.SetBottomMargin(0.35)
        bottomPad.SetRightMargin(0.04)
    else :
        topPad = ROOT.TPad("topPad", "topPad", 0.00, 0.05, 1.00, 0.995)
        topPad.SetFillColor(10)
        topPad.SetTopMargin(0.075)
        topPad.SetLeftMargin(0.20)
        topPad.SetBottomMargin(0.1)
        topPad.SetRightMargin(0.04)
        if bin["useLogPlot"]:
            topPad.SetLogy()"""

    canvas.cd()
    if useLogPlot :
        canvas.SetLogy()


    """dumb = topPad.Draw()
    del dumb
    topPad.cd()
    del topPad"""
    #dumb = hist_total.Draw("axis")
    dumb = hist_total.Draw()

    del dumb
    histogramStack_mc = ROOT.THStack()
    print ("list of processes considered and their integrals")

    linebin = []
    linebinW = []
    poslinebinW_X = []
    pos_linebinW_Y = []
    y0 = 100. #options_plot_ranges("ttH")[typeCat]["position_cats"]
    if era == 0 :
        y0 = 2 * y0
    #"""
    for kk, key in  enumerate(dprocs.keys()) :
        hist_rebin = template.Clone()
        lastbin = 0
        addlegend = True
        print("Stacking ", key)
        for cc, catcat in enumerate(catcats) :
            if not cc == 0 :
                addlegend = False
            if kk == 0 :
                firstHisto = ROOT.TH1F()
            readFrom = folder + "/" + catcat
            info_hist = rebin_hist(
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
            print (info_hist["lastbin"] , lastbin, nbinscatlist[cc] )
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
                    poslinebinW_X += [10] #[options_plot_ranges("ttH")[typeCat]["catsX"][cc]]
                pos_linebinW_Y += [y0]
        if hist_rebin == 0 or not hist_rebin.Integral() > 0 or (info_hist["labelPos"] == 0 and not normalize_X_original == "none" )  :
            continue
        print (key,  0 if hist_rebin == 0 else hist_rebin.Integral() )
        dumb = histogramStack_mc.Add(hist_rebin)
        del dumb
    #"""

    ## create signal histogram
    ################################

    #for line1 in linebin :
    #    line1.SetLineColor(1)
    #    line1.SetLineStyle(3)
    #    line1.Draw()

    dumb = hist_total.Draw("same")
    dumb = histogramStack_mc.Draw("hist,same")
    del dumb
    dumb = hist_total.Draw("e2,same")
    del dumb
    ## draw signal
    ################################
    if unblind :
        dumb = dataTGraph1.Draw("e1P,same")
        del dumb
    dumb = hist_total.Draw("axis,same")
    del dumb

    dumb = legend1.Draw("same")
    del dumb

    labels = addLabel_CMS_preliminary(era)
    for ll, label in enumerate(labels) :
        if ll == 0 :
            dumb = label.Draw("same")
            del dumb
        else :
            dumb = label.Draw()
            del dumb

    #################################
    """if do_bottom :
        canvas.cd()
        dumb = bottomPad.Draw()
        del dumb"""

    ##################################
    oplin = "linear"
    if useLogPlot :
        oplin = "log"
        print ("made log")
    optbin = "plain"
    if divideByBinWidth :
        optbin = "divideByBinWidth"

    savepdf = path.replace(".pdf", "%s_%s_unblind%s" % (typeFit, oplin, unblind))
    #options.odir+category+"_"+typeFit+"_"+optbin+"_"+options.nameOut+"_unblind"+str(options.unblind)+"_"+oplin + "_" + options.typeCat
    print ("saving...", savepdf )
    dumb = canvas.SaveAs(savepdf + ".pdf")
    print ("saved", savepdf + ".pdf")
    del dumb
    canvas.IsA().Destructor(canvas)



if __name__ == "__main__":
    # test
    data_dir = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017/fitDiagnostics.root"
    options_dat = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017/datacard_SL_DL_bbWW_nonresNLO_none_45_75_allSig_2017_noSingleH_naming_plot_options.dat"
    output_folder = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017"
    normalize_X_original = True
    unblind = True
    doPostFit = False
    do_bottom = False
    divideByBinWidth = False

    labelY = "Events"
    if divideByBinWidth : labelY = "Events / bin width"

    info_bin = eval(open(options_dat, 'r').read())

    for key, bin in info_bin.iteritems() :
        print("Drawing %s" % key)
        create_postfit_plots(
            path="%s/plot_%s.pdf" % (output_folder, key) ,
            fit_diagnostics_path=data_dir,
            normalize_X_original=normalize_X_original,
            doPostFit=doPostFit,
            divideByBinWidth=divideByBinWidth,
            bin=bin,
            binToRead=key,
            unblind=unblind
        )
