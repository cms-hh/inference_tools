from dhi.util import import_ROOT
ROOT = import_ROOT()

def test_print() :
    print("it works!")

def GetNonZeroBins(template) :
    nbins = 0
    for ii in xrange(1, template.GetXaxis().GetNbins()+1) :
        binContent_original = template.GetBinContent(ii)
        if binContent_original > 0 : nbins += 1
    return nbins

def process_data_histo(template, dataTGraph1, folder, fin, lastbin, histtotal, catbin, minY, maxY, divideByBinWidth) :
    dataTGraph = fin.Get(folder + "/data")
    print("adding", folder + "/data")
    allbins = catbin
    for ii in xrange(0, allbins) :
        bin_width = 1.
        if divideByBinWidth :
            bin_width = histtotal.GetXaxis().GetBinWidth(ii+1)
        xp = ROOT.Double()
        yp = ROOT.Double()
        dataTGraph.GetPoint(ii, xp, yp)
        dataTGraph1.SetPoint(      ii + lastbin,  template.GetBinCenter(ii + lastbin + 1) , yp/bin_width)
        dataTGraph1.SetPointEYlow( ii + lastbin,  dataTGraph.GetErrorYlow(ii)/bin_width)
        dataTGraph1.SetPointEYhigh(ii + lastbin,  dataTGraph.GetErrorYhigh(ii)/bin_width)
        dataTGraph1.SetPointEXlow( ii + lastbin,  template.GetBinWidth(ii+1)/2.)
        dataTGraph1.SetPointEXhigh(ii + lastbin,  template.GetBinWidth(ii+1)/2.)
    del dataTGraph
    dataTGraph1.SetMarkerColor(1)
    dataTGraph1.SetMarkerStyle(20)
    dataTGraph1.SetMarkerSize(0.8)
    dataTGraph1.SetLineColor(1)
    dataTGraph1.SetLineWidth(1)
    dataTGraph1.SetLineStyle(1)
    dataTGraph1.SetMinimum(minY)
    dataTGraph1.SetMaximum(maxY)
    return allbins

def process_total_histo(hist, folder, fin, divideByBinWidth, name_total, lastbin, do_bottom, labelX, catbins, minY, maxY) :
    total_hist_name = folder + "/" + name_total
    total_hist = fin.Get(total_hist_name)
    print ("Total band taken from %s" % total_hist_name)
    allbins = catbins
    hist.SetMarkerSize(0)
    hist.SetMarkerColor(16)
    hist.SetFillColorAlpha(12, 0.40)
    hist.SetLineWidth(0)
    hist.SetMinimum(minY)
    hist.SetMaximum(maxY)
    for ii in xrange(1, allbins + 1) :
        bin_width = 1.
        if divideByBinWidth : bin_width = total_hist.GetXaxis().GetBinWidth(ii)
        hist.SetBinContent(ii + lastbin, 0.03 + total_hist.GetBinContent(ii)/bin_width)
        hist.SetBinError(  ii + lastbin, 0.03 + total_hist.GetBinError(ii)/bin_width)
    if not hist.GetSumw2N() : hist.Sumw2()
    if not do_bottom :
        hist.GetXaxis().SetTitle(labelX)
        hist.GetXaxis().SetTitleOffset(0.85)
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetXaxis().SetLabelSize(0.05)
        hist.GetYaxis().SetTitleOffset(1.5)
        hist.GetXaxis().SetLabelColor(1)
    else :
        hist.GetXaxis().SetTitleOffset(0.7)
        hist.GetYaxis().SetTitleOffset(1.2)
        hist.GetXaxis().SetLabelColor(10)
    hist.GetXaxis().SetTickLength(0.04)
    hist.GetYaxis().SetTitleSize(0.055)
    hist.GetYaxis().SetTickLength(0.04)
    hist.GetYaxis().SetLabelSize(0.050)
    return allbins

def addLabel_CMS_preliminary(era, do_bottom) :
    x0 = 0.2
    y0 = 0.953 if do_bottom else 0.935
    ypreliminary = 0.95 if do_bottom else 0.935
    xpreliminary = 0.12 if do_bottom else 0.085
    ylumi = 0.95 if do_bottom else 0.965
    xlumi = 0.7 if do_bottom else 0.73
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
    label_preliminary = ROOT.TPaveText(x0 + xpreliminary, y0 - 0.005, x0 + 0.0980 + 0.12, y0 + 0.0600 - 0.005, "NDC")
    label_preliminary.AddText("Preliminary")
    label_preliminary.SetTextFont(50)
    label_preliminary.SetTextAlign(13)
    label_preliminary.SetTextSize(title_size_Preliminary)
    label_preliminary.SetTextColor(1)
    label_preliminary.SetFillStyle(0)
    label_preliminary.SetBorderSize(0)
    label_luminosity = ROOT.TPaveText(xlumi, y0 + 0.0035, xlumi + 0.0900, y0 + 0.040, "NDC")
    if era == 2016 : lumi = "35.92"
    if era == 2017 : lumi = "41.53"
    if era == 2018 : lumi = "59.74"
    if era == 0    : lumi = "137"
    label_luminosity.AddText(lumi + " fb^{-1} (13 TeV)")
    label_luminosity.SetTextFont(42)
    label_luminosity.SetTextAlign(13)
    label_luminosity.SetTextSize(title_size_lumi)
    label_luminosity.SetTextColor(1)
    label_luminosity.SetFillStyle(0)
    label_luminosity.SetBorderSize(0)

    return [label_cms, label_preliminary, label_luminosity]

def stack_histo(hist_rebin_local, fin, folder, name, itemDict, divideByBinWidth, addlegend, lastbin, catbin, original, firstHisto, era, legend) :
    histo_name = folder+"/"+name
    print ("try find %s" % histo_name)
    hist = fin.Get(histo_name)
    allbins = catbin
    try  :
        hist.Integral()
    except :
        print ("Doesn't exist %s" % histo_name)
        return {
            "lastbin" : allbins,
            "binEdge" : lastbin - 0.5 , 
            "labelPos" : 0 if not original == "none" else float(allbins/2)
            }
    if not firstHisto.Integral() > 0 :
        firstHisto = hist.Clone()
        for ii in xrange(1, firstHisto.GetNbinsX() + 1) :
            firstHisto.SetBinError(  ii, 0.001)
            firstHisto.SetBinContent(ii, 0.001)
    hist_rebin_local.SetMarkerSize(0)
    hist_rebin_local.SetFillColor(itemDict["color"])
    if not itemDict["fillStype"] == 0 :
        hist_rebin_local.SetFillStyle(itemDict["fillStype"])

    if "none" not in itemDict["label"] and addlegend :
        legend.AddEntry(hist_rebin_local, itemDict["label"], "f")
    if itemDict["make border"] == True :  hist_rebin_local.SetLineColor(1)
    else : hist_rebin_local.SetLineColor(itemDict["color"])
    for ii in xrange(1, allbins + 1) :
        bin_width = 1.
        if divideByBinWidth : bin_width = hist.GetXaxis().GetBinWidth(ii)
        ### remove and point bins with negative entry
        binContent_original = hist.GetBinContent(ii)
        binError2_original = hist.GetBinError(ii)**2
        if binContent_original < 0. :
            binContent_modified = 0.
            print ("bin with negative entry: ", ii, '\t', binContent_original)
            binError2_modified = binError2_original + math.pow((binContent_original-binContent_modified),2)
            if not binError2_modified >= 0. : print "Bin error negative!"
            hist_rebin_local.SetBinError(  ii + lastbin, math.sqrt(binError2_modified)/bin_width)
            hist_rebin_local.SetBinContent(ii + lastbin, 0.)
            print 'binerror_original= ', binError2_original, '\t',  'bincontent_original', '\t', binContent_original,'\t', 'bincontent_modified', '\t', binContent_modified, '\t', 'binerror= ', hist_rebin.GetBinError(ii)
        else :
            hist_rebin_local.SetBinError(  ii + lastbin,   hist.GetBinError(ii)/bin_width)
            hist_rebin_local.SetBinContent(ii + lastbin, hist.GetBinContent(ii)/bin_width)
    if not hist.GetSumw2N() : hist.Sumw2()
    return {
        "lastbin" : allbins,
        "binEdge" : hist.GetXaxis().GetBinLowEdge(lastbin) + hist.GetXaxis().GetBinWidth(lastbin) - 0.5 , # if lastbin > 0 else 0
        "labelPos" : float(allbins/2)
        }

def do_hist_total_err(hist_total_err, labelX, total_hist, minBottom, maxBottom, era) :
    allbins = total_hist.GetNbinsX() #GetNonZeroBins(total_hist)
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
    if era == 0 :
        minBottom = minBottom #*3/2
        maxBottom = maxBottom
    hist_total_err.SetMinimum(minBottom)
    hist_total_err.SetMaximum(maxBottom)
    for bin in xrange(0, allbins) :
        hist_total_err.SetBinContent(bin+1, 0)
        if total_hist.GetBinContent(bin+1) > 0. :
            hist_total_err.SetBinError(bin + 1, total_hist.GetBinError(bin+1)/total_hist.GetBinContent(bin+1))
    return allbins

def err_data(dataTGraph1, template, dataTGraph, histtotal, folder, fin, divideByBinWidth, lastbin) :
    print(" do unblided bottom pad")
    allbins = histtotal.GetXaxis().GetNbins() #GetNonZeroBins(histtotal)
    print("allbins", allbins)
    for ii in xrange(0, allbins) :
        bin_width = 1.
        if divideByBinWidth :
            bin_width = histtotal.GetXaxis().GetBinWidth(ii+1)
        if histtotal.GetBinContent(ii+1) == 0 : continue
        dividend = histtotal.GetBinContent(ii+1)*bin_width
        xp = ROOT.Double()
        yp = ROOT.Double()
        dataTGraph.GetPoint(ii,xp,yp)
        if yp > 0 :
            if dividend > 0 :
                dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin + 1) , yp/dividend-1)
            else :
                dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin + 1) , -1.)
        else :
            dataTGraph1.SetPoint(ii + lastbin, template.GetBinCenter(ii + lastbin +1) , -1.)
        dataTGraph1.SetPointEYlow(ii + lastbin,  dataTGraph.GetErrorYlow(ii)/dividend)
        dataTGraph1.SetPointEYhigh(ii + lastbin, dataTGraph.GetErrorYhigh(ii)/dividend)
        dataTGraph1.SetPointEXlow(ii + lastbin,  template.GetBinWidth(ii+1)/2.)
        dataTGraph1.SetPointEXhigh(ii + lastbin, template.GetBinWidth(ii+1)/2.)
    dataTGraph1.SetMarkerColor(1)
    dataTGraph1.SetMarkerStyle(20)
    dataTGraph1.SetMarkerSize(0.8)
    dataTGraph1.SetLineColor(1)
    dataTGraph1.SetLineWidth(1)
    dataTGraph1.SetLineStyle(1)
    return allbins
