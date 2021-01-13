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

- TODO: make the dictionary example on datacards_run2 repo, and make the path to the original datacard.root or relative to the datacards_run2 when I do the example with cards from datacards_run2 repo

## Common options that we may like to change

- single H processes are added to the stack of backgrounds
- the total band is the sum of BKGs only

## Pending/suggestions

- make the options flow in law scheme

- in the shapes plot lumi header (addLabel_CMS_preliminary(era) function here) take the numbers from an central place (the same that is written for other eras)

- save the log of the plot (what the script ) along with the plot.pdf/root/png, so the person running can check it and the person implementing a new dictionary can debug mistakes/lists of processes

- I still did not removed completely the functionality to plot different categories one after the other in a same canvas -- I will remove when I am sure nobody will use == after some tests

## Task missing:
- make prefit/postfit table of yields with uncertainties
