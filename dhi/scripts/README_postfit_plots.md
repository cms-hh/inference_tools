-- to pass some info to online documentation in a proper format

# inputs

The list of user inputs to the plotter-only is:
- fitDiagnosis (result of the PostFitShapes task)
  - Made from a cards combination in which subcategories should be merged using a fixed naming convention for bins, eg
  """
  combineCards.py \
  bbWW_SL_2017=datacard_hh_bb1l_hh_bb1l_cat_jet_2BDT_Wjj_simple_SM_Res_allReco_bbWW_nonresNLO_none_75_multisig_2017.txt \
  bbWW_DL_2017=datacard_hh_bb2l_hh_bb2l_OS_SM_plainVars_inclusive_bbWW_nonresNLO_none_45_noSL_2017.txt>\
  datacard_SL_DL_bbWW_nonresNLO_none_45_75_allSig_2017_noSingleH_naming.txt
  """
- dictionaries with plot options, minding the above-mentioned fixed naming convention for bins

## Explaining the dictionary


This is an example of dictionary booking one plot:

```
{
  "plotXX" : {
      "fitdiagnosis"       : "/where/is/the/fitdiagnosis.root",
      "bin_name_original"  : "none",
      "datacard_original"  : "none",
      "minY"               : 0.1,
      "maxY"               : 100000000000.,
      "minYerr"            : -0.28,
      "maxYerr"            : 0.28,
      "useLogPlot"         : True,
      "era"                : 2016 ,
      "labelX"             : "bla bla",
      "header_legend"          : "bla bla can be latex",
      "number_columns_legend" : 3,
      "align_cats" : ["ch1","ch2",],
      "align_cats_labels" : [["ch1 bla", "ch1 more details"], ["ch2 bla", "ch2 more details"]],
      "align_cats_labelsX" : [3, 13],
      "cats_labels_height" : 1000000.,
      "merged_eras_fit" : False,
      "procs_plot_options_bkg" : OrderedDict(
          [
          ("Other_bbWW",       {"color" : 205, "fillStype"   : 1001, "label" : "others"           , "make border" : True}),
          ("VV",          {"color" : 823, "fillStype"   : 1001, "label" : "ZZ + WZ"          , "make border" : True}),
          ("ST",         {"color" : 822, "fillStype" : 1001, "label" : "single top"         , "make border" : True}),
          ("Fakes",  {"color" :  12, "fillStype" : 3345, "label" : "Fakes"  , "make border" : True}),
          ("DY",          {"color" : 221, "fillStype" : 1001, "label" : "DY"         , "make border" : True}),
          ("WJets",           {"color" : 5, "fillStype" : 1001, "label" : 'W + jets'   , "make border" : True}),
          ("TT",          {"color" : 17, "fillStype"  : 1001, "label" : 't#bar{t} + jets'   , "make border" : True})
          ]
      ),
      'procs_plot_options_sig' : OrderedDict(
       [
       ('ggHH_kl_1_kt_1' ,      {'color' : 5, 'fillStype'  : 3351, 'label' : 'GGF HH SM', 'scaleBy' : 1.}),
       ('ggHH_kl_5_kt_1',       {'color' : 221, 'fillStype'  : 3351, 'label' : 'GGF HH #kappa#lambda = 5', 'scaleBy' : 1.}),
       ('ggHH_kl_2p45_kt_1',    {'color' : 2, 'fillStype'  : 3351, 'label' : 'GGF HH #kappa#lambda = 2.45', 'scaleBy' : 1.}),
       ('qqHH_CV_1_C2V_1_kl_1', {'color' : 8, 'fillStype'  : 3351, 'label' : 'VBF HH SM', 'scaleBy' : 1.}),
       ]
       ),
  },
}
```

The keys of the dictionary are the names of the bins to plot distributions for (declared on the combineCards command). The entries are bellow:

- "datacard_original": the datacard.root file with shapes for that bin (what goes along with datacard.txt)
- "bin_name_original" : For some analyses the bin can be in a folder inside the datacard.root , if this is the case put the name of this folder. If there is no internal folder, just put "none".
- Y-axis of the shapes distributions (top pad) : "minY" / "maxY"
- Y-axis of the bottom pad : "minYerr", "maxYerr"
- "useLogPlot", for the shapes distributions (top pad)
- "era", to decide which lumi put on plot header
- "labelX" is the variable being plotted
- options for legends "header_legend", "number_columns_legend"
- "procs_plot_options_bkg" is the list of processes to be stacked in the plot, with options
  - the keys are the exact names of the processes on datacard.root
  - the order you write there is going to be the order of the stack and of the legend entries
  - if you put a process that is not on the datacard it will skip it (not break)
  - if you do not put a process that is on the datacard, it will not add it to the stack (useful to negligible processes)
  - How to merge processes: put them subsequently. Put the desired  "label" and "make border"==True only on the last of the list, on the others put "label"="none" and "make border"=False. Example here [link from datacards_run2 uploaded example].
- "procs_plot_options_sig" is the list of signals (processes to not be added to the stack)
  - each entry will be added as one overlaid histogram, summing up the processes listed in "processes" and scaled by "scaleBy"
-  "align_cats" is a list of bins/channels that we want drawn one after another in the same plot
-   "align_cats_labels" are the labels for "align_cats"
-  "align_cats_labelsX" : the X positions for the labels "align_cats_labels"
-   "cats_labels_height" : the Y positions for the labels "align_cats_labels"
- "merged_eras_fit" : if true it will try to read the single H processes with era in name (e.g. "ttH_2017_hbb" instead of "ttH_hbb")

- TODO: make the dictionary example on datacards_run2 repo, and make the path to the original datacard.root or relative to the datacards_run2 when I do the example with cards from datacards_run2 repo

## Common options that we may like to change

Right now:
- single H processes are added to the stack of backgrounds, as first entry
- the total band is the sum of BKGs only

We would like:
- use the fact that there is a naming convention to signal and avoiod the need of "procs_plot_options_sig" on the dictionary (right nonow we did not agreed on the suffixes, so that is not so eazy)

## Pending/suggestions

- make the options flow in law scheme

- in the shapes plot lumi header (addLabel_CMS_preliminary(era) function [here](https://gitlab.cern.ch/hh/tools/inference/-/blob/postfit_plots/dhi/util_shapes_plot.py#L109-112) take the numbers from an central place (the same that is written for other plots)

- save the log of the plot (what the script prints) along with the plot.pdf/root/png, so the person running can check it and the person implementing a new dictionary can debug mistakes/lists of processes

- I still did not removed completely the functionality to plot different categories one after the other in a same canvas -- I will remove when I am sure nobody will use == after some tests

## Task missing:
- make prefit/postfit table of yields with uncertainties
