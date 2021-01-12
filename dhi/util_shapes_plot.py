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

def rebin_data(template, dataTGraph1, folder, fin, lastbin, histtotal, catbin, minY, maxY, divideByBinWidth) :
    dataTGraph = fin[0].Get(folder + "/data")
    print("adding", folder + "/data")
    allbins = catbin
    if len(fin) == 3 :
        for eraa in [1,2] :
            if eraa == 1 : folderRead = folder.replace("2018", "2017")
            if eraa == 2 : folderRead = folder.replace("2018", "2016")
            dataTGraph2 = fin[eraa].Get(folderRead + "/data")
            print("adding", folderRead + "/data", lastbin)
            for ii in xrange(0, allbins) :
                xp = ROOT.Double()
                yp = ROOT.Double()
                dataTGraph.GetPoint(ii, xp, yp)
                xp2 = ROOT.Double()
                yp2 = ROOT.Double()
                dataTGraph2.GetPoint(ii, xp2, yp2)
                dataTGraph.SetPoint(      ii ,  template.GetBinCenter(ii + 1) , (yp + yp2))
                dataTGraph.SetPointEYlow( ii ,  math.sqrt(dataTGraph.GetErrorYlow(ii)*dataTGraph.GetErrorYlow(ii) + dataTGraph2.GetErrorYlow(ii)*dataTGraph2.GetErrorYlow(ii)))
                dataTGraph.SetPointEYhigh(ii ,  math.sqrt(dataTGraph.GetErrorYhigh(ii)*dataTGraph.GetErrorYhigh(ii) + dataTGraph2.GetErrorYhigh(ii)*dataTGraph2.GetErrorYhigh(ii)))
                dataTGraph.SetPointEXlow( ii ,  template.GetBinWidth(ii+1)/2.)
                dataTGraph.SetPointEXhigh(ii ,  template.GetBinWidth(ii+1)/2.)
            del dataTGraph2
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

def rebin_total(hist, folder, fin, divideByBinWidth, name_total, lastbin, do_bottom, labelX, catbins, minY, maxY) :
    total_hist = fin[0].Get(folder + "/" + name_total)
    print(len(fin))
    ## if we want to sum eras; XandaFix
    if len(fin) == 3 :
        for eraa in [1,2] :
            if eraa == 1 : folderRead = folder.replace("2018", "2017")
            if eraa == 2 : folderRead = folder.replace("2018", "2016")
            print ("reading ", eraa, folderRead + "/" + name_total)
            print(folderRead + "/" + name_total)
            total_hist.Add(fin[eraa].Get(folderRead + "/" + name_total))
    print (folder + "/" + name_total)
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
        hist.GetXaxis().SetTitleOffset(1.2)
        hist.GetXaxis().SetTitleSize(0.05)
        hist.GetXaxis().SetLabelSize(0.05)
        hist.GetXaxis().SetLabelColor(1)
    else :
        hist.GetXaxis().SetLabelColor(10)
        hist.GetXaxis().SetTitleOffset(0.7)
        hist.GetXaxis().SetTickLength(0.04)
        hist.GetYaxis().SetTitleOffset(1.2)
    hist.GetYaxis().SetTitleSize(0.055)
    hist.GetYaxis().SetTickLength(0.04)
    hist.GetYaxis().SetLabelSize(0.050)
    return allbins

def addLabel_CMS_preliminary(era) :
    x0 = 0.2
    y0 = 0.953
    ypreliminary = 0.95
    xlumi = 0.67
    label_cms = ROOT.TPaveText(x0, y0, x0 + 0.0950, y0 + 0.0600, "NDC")
    label_cms.AddText("CMS")
    label_cms.SetTextFont(61)
    label_cms.SetTextAlign(13)
    label_cms.SetTextSize(0.0575)
    label_cms.SetTextColor(1)
    label_cms.SetFillStyle(0)
    label_cms.SetBorderSize(0)
    label_preliminary = ROOT.TPaveText(x0 + 0.12, y0 - 0.005, x0 + 0.0980 + 0.12, y0 + 0.0600 - 0.005, "NDC")
    label_preliminary.AddText("Preliminary")
    label_preliminary.SetTextFont(50)
    label_preliminary.SetTextAlign(13)
    label_preliminary.SetTextSize(0.048)
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
    label_luminosity.SetTextSize(0.045)
    label_luminosity.SetTextColor(1)
    label_luminosity.SetFillStyle(0)
    label_luminosity.SetBorderSize(0)

    return [label_cms, label_preliminary, label_luminosity]

def rebin_hist(hist_rebin_local, fin, folder, name, itemDict, divideByBinWidth, addlegend, lastbin, catbin, original, firstHisto, era, legend) :
    histo_name = folder+"/"+name
    print ("try find %s" % histo_name)
    hist = fin[0].Get(histo_name) #era
    allbins = catbin #hist.GetNbinsX() #GetNonZeroBins(hist)
    try  :
        hist.Integral()
    except :
        print ("Doesn't exist %s in %s" % (histo_name, era))
        if len(fin) > 1 :
            hist = firstHisto.Clone()
            ## find the histogram in any of the three eras file (it shall be an smarter way to write)
            """hist = fin[1].Get(folder.replace("2018", "2017")+"/"+name)
            try  : hist.Integral()
            except :
                print ("Doesn't exist " + folder.replace("2018", "2017")+"/"+name, "in 2017")
                if len(fin) > 2 :
                    hist = fin[2].Get(folder.replace("2018", "2016")+"/"+name)
                    try  : hist.Integral()
                    except :
                        print ("Doesn't exist " + folder.replace("2018", "2016")+"/"+name, "in 2016")
                        hist = firstHisto.Clone()
                    hist = firstHisto.Clone()"""
        else :
            print ("Doesn't exist %s" % histo_name)
            return {
                "lastbin" : allbins,
                "binEdge" : lastbin - 0.5 , # if lastbin > 0 else 0
                "labelPos" : 0 if not original == "none" else float(allbins/2)
                }
    if not firstHisto.Integral() > 0 :
        firstHisto = hist.Clone()
        for ii in xrange(1, firstHisto.GetNbinsX() + 1) :
            firstHisto.SetBinError(  ii, 0.001)
            firstHisto.SetBinContent(ii, 0.001)
    if len(fin) == 3 :
        for eraa in [1,2] :
            if eraa == 1 : folderRead = folder.replace("2018", "2017")
            if eraa == 2 : folderRead = folder.replace("2018", "2016")
            try :
                print ("era" + str(eraa))
                hist.Add(fin[eraa].Get(folderRead+"/"+name))
            except :
                if name == "data_fakes" :
                    try :
                        hist.Add(fin[eraa].Get(folderRead+"/fakes_mc"))
                    except :
                        print ("did not found fakes")
                        continue
                if name == "fakes_mc" :
                    try :
                        hist.Add(fin[eraa].Get(folderRead+"/data_fakes"))
                    except :
                        print ("did not found fakes")
                        continue
                else :
                    continue
    hist_rebin_local.SetMarkerSize(0)
    hist_rebin_local.SetFillColor(itemDict["color"])
    if not itemDict["fillStype"] == 0 :
        hist_rebin_local.SetFillStyle(itemDict["fillStype"])
    #print("signal HH stack inside", key, hist_rebin_local.Integral())


    if "none" not in itemDict["label"] and addlegend :
        legend.AddEntry(hist_rebin_local, itemDict["label"], "f")
    if itemDict["make border"] == True :  hist_rebin_local.SetLineColor(1)
    else : hist_rebin_local.SetLineColor(itemDict["color"])
    for ii in xrange(1, allbins + 1) :
        bin_width = 1.
        if divideByBinWidth : bin_width = hist.GetXaxis().GetBinWidth(ii)
        ### remove negatives
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
