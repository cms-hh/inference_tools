#!/usr/bin/env python
# coding: utf-8

"""
Script to extract and plot shapes from a ROOT file create by combine's FitDiagnostics.
"""

import os
from collections import OrderedDict

from dhi.util import import_ROOT
from dhi.util_shapes_plot import test_print, GetNonZeroBins, process_data_histo, process_total_histo, addLabel_CMS_preliminary, stack_histo, do_hist_total_err, err_data

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
    nameLabel=bin["nameLabel"]
    datacard_original=bin["datacard_original"]
    bin_name_original=bin["bin_name_original"]
    number_columns_legend=bin["number_columns_legend"]

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
    if divideByBinWidth : labelY = "Events / bin width"

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

    legend1 = ROOT.TLegend(0.2400, 0.645, 0.9450, 0.910)
    legend1.SetNColumns(number_columns_legend)
    legend1.SetFillStyle(0)
    legend1.SetBorderSize(0)
    legend1.SetFillColor(10)
    legend1.SetTextSize(0.040 if do_bottom else 0.03)
    legend1.SetHeader(label_head)
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
            histtotal = fin[0].Get(readFrom + "/" + name_total )
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
            minY, maxY
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
    for kk, key in  enumerate(dprocs.keys()) :
        hist_rebin = template.Clone()
        lastbin = 0 # reminiscent of putting histograms from different bins in same plot side by side
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
        print("Stacking proocess %s, with yield %s " % (key, str(round(hist_rebin.Integral(),2))))
        dumb = histogramStack_mc.Add(hist_rebin)
        del dumb

    ## create signal histogram
    ################################

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
                histtotal = fin[0].Get((readFrom + "/" + name_total).replace("2018", str(era)) )
                lastbin += err_data(
                    dataTGraph2,
                    hist_total,
                    dataTGraph1,
                    hist_total,
                    readFrom,
                    fin[0],
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
    #    #canvas.cd()
    #    #dumb = bottomPad.Draw()
    #    #del dumb

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
    canvas.IsA().Destructor(canvas)

if __name__ == "__main__":
    # test
    data_dir = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017/fitDiagnostics.root"
    options_dat = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017/datacard_SL_DL_bbWW_nonresNLO_none_45_75_allSig_2017_noSingleH_naming_plot_options.dat"
    output_folder = "/afs/cern.ch/work/a/acarvalh/public/to_HH_bbWW/combo_test_plots_2017/plots/"
    normalize_X_original = True
    unblind = True
    doPostFit = False
    do_bottom = True
    divideByBinWidth = False

    info_bin = eval(open(options_dat, 'r').read())

    for key, bin in info_bin.iteritems() :
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
