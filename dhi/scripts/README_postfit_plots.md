-- to pass some info to online documentation in a proper format

We use the plotter in standalone mode as:

```
python dhi/scripts/postfit_plots.py \
--output_folder /where/save/the/plots/ \
--plot_options_dict /where/is/plot_options.json
```

There are additional options, like eg  `--unblind` (the default is blinded). Please use `--help` to see further options.

# Explaining the dictionary

This is an example of dictionary booking one plot, that should be saved in a .json file/format as `/where/is/plot_options.json` (on the example comand line above):

```
{
  "plotXX_2018":
  {
    "fitdiagnosis": "/where/is/the/fitdiagnosis.root",
    "datacard_original": "none",
    "bin_name_original": "none",
    "procs_plot_options_bkg": "plot_options_BKG_colors.json",
    "procs_plot_options_sig": "plot_options_sig_colors.json",
    "header_legend": "bla bla can be latex",
    "labelX": "bla bla can be latex",
    "maxY": 10000.0,
    "minY": 0.001,
    "maxYerr": 20.01,
    "minYerr": -5.01,
    "maxYerr_postfit": 15.01,
    "minYerr_postfit": -15.01,
    "cats_labels_height": 8.0,
    "number_columns_legend": 3,
    "useLogPlot": true,
    "era": 2018,
    "align_cats": ["ch1","ch2",],
    "align_cats_labelsX": [6, 20],
    "align_cats_labels": [["ch1 bla", "ch1 more details"], ["ch2 bla", "ch2 more details"]],
    "merged_eras_fit": false
    }
  }
```

## Plot options

The keys of the dictionary are the names of the bins to plot distributions for (declared on the combineCards command). The entries are bellow:

- "datacard_original": the datacard.root file with shapes for that bin (what goes along with datacard.txt)
- "bin_name_original" : For some analyses the bin can be in a folder inside the datacard.root , if this is the case put the name of this folder. If there is no internal folder, just put "none".
- Y-axis of the shapes distributions (top pad) : "minY" / "maxY"
- Y-axis of the bottom pad for prefit plot: "minYerr", "maxYerr"
- Y-axis of the bottom pad for postfit plot: "minYerr_postfit", "maxYerr_postfit" (if it is not given it will use the ones for prefit, defined above)
- "useLogPlot", for the shapes distributions (top pad)
- "era", to decide which lumi put on plot header
- "labelX" is the variable being plotted
- options for legends "header_legend", "number_columns_legend"
- "procs_plot_options_bkg" is the name of the json file containing the list of BKG processes to be drawn and options for plotting. See point bellow.
- "procs_plot_options_sig" is the name of the json file containing the list of signal processes to be drawn and options for plotting. See point bellow.
-  "align_cats" is a list of bins/channels that we want drawn one after another in the same plot
  - It can be one, that is the most common use
-   "align_cats_labels" are the labels for "align_cats"
-  "align_cats_labelsX" : the X positions for the labels "align_cats_labels"
-   "cats_labels_height" : the Y positions for the labels "align_cats_labels"
- "merged_eras_fit" : if true it will try to read the single H processes with era in name (e.g. "ttH_2017_hbb" instead of "ttH_hbb")


- TODO: make the dictionary example on datacards_run2 repo, and make the path to the original datacard.root or relative to the datacards_run2 when I do the example with cards from datacards_run2 repo

### Processes dictionaries

There are mentions for two files json files, if just the plain name of a file it will assume that they are in the same folder than `/where/is/plot_options.json`, if it is a path it will use the string as an absolute path.

#### Explaining procs_plot_options_bkg

- "procs_plot_options_bkg" is the list of processes to be stacked in the plot, with options
  - the keys are the exact names of the processes on datacard.root
  - the order you write there is going to be the order of the stack and of the legend entries
  - if you put a process that is not on the datacard it will skip it (not break)
  - if you do not put a process that is on the datacard, it will not add it to the stack (useful to negligible processes)
  - How to merge processes: put them subsequently. Put the desired  "label" and "make border"==True only on the last of the list, on the others put "label"="none" and "make border"=False. Example here [link from datacards_run2 uploaded example].

An explicit example is bellow:

```
{
  "bbzz4l_others" : {"color": 205, "fillStype": 1001, "make border": true, "label": "others"},
  "ZX"            : {"color": 208, "fillStype": 1001, "make border": true, "label": "ZX"},
  "ttZ"           : {"color": 9, "fillStype": 1001, "make border": true, "label": "ttZ"},
  "ggZZ"          : {"color": 822, "fillStype": 1001, "make border": true, "label": "ggZZ"},
  "qqZZ"          : {"color": 221, "fillStype": 1001, "make border": true, "label": "qqZZ"}
}
```

#### Explaining procs_plot_options_sig

- "procs_plot_options_sig" is the list of signals (processes to be drawn but not added to the stack)
  - each entry will be added as one overlaid histogram, summing up the processes listed in "processes" and scaled by "scaleBy"
  - the order you write there is going to be the legend entries

An explicit example is bellow:

```
{
  "ggHH_kl_1_kt_1"       : {"color": 5, "fillStype": 3351, "scaleBy": 1.0, "label": "GGF SM"},
  "ggHH_kl_5_kt_1"       : {"color": 221, "fillStype": 3351, "scaleBy": 1.0, "label": "GGF #kappa#lambda = 5"},
  "ggHH_kl_2p45_kt_1"    : {"color": 2, "fillStype": 3351, "scaleBy": 1.0, "label": "GGF #kappa#lambda = 2.45"},
  "qqHH_CV_1_C2V_1_kl_1" : {"color": 8, "fillStype": 3351, "scaleBy": 1.0, "label": "VBF SM"},
  "qqHH_CV_1_C2V_2_kl_1" : {"color": 6, "fillStype": 3351, "scaleBy": 1.0, "label": "VBF c2V = 2"}
}
```

### Common options that we may like to change

Right now:
- the total band is the sum of BKGs only

### Pending/suggestions

- make the options flow in law scheme

- in the shapes plot lumi header (addLabel_CMS_preliminary(era) function [here](https://gitlab.cern.ch/hh/tools/inference/-/blob/postfit_plots/dhi/util_shapes_plot.py#L109-112) take the numbers from an central place (the same that is written for other plots)

- save the log of the plot (what the script prints) along with the plot.pdf/root/png, so the person running can check it and the person implementing a new dictionary can debug mistakes/lists of processes



## On the shapes input (FitDiagnostics)

The list of user inputs to the plotter-only is:
- fitDiagnosis (result of the PostFitShapes task)
  - Made from a cards combination in which subcategories should be merged using a fixed naming convention for bins, that you will enter on the dictionary as "align_cats", eg (following the example dictionary above)
  """
  combineCards.py \
  ch1=datacard_ch1.txt \
  ch2=datacard_ch2.txt>\
  datacard_combo.txt
  """
- dictionaries with plot options, minding the above-mentioned fixed naming convention for bins


# Task missing:
- make prefit/postfit table of yields with uncertainties
