
We use the plotter as a task in this example as:

```
law run  PlotDistributionsAndTables \
--version plotting_FAKE \
--datacards $DHI_BASE/docs/example_distribution_plotter/datacard.txt \
--draw_for_channels $DHI_BASE/docs/example_distribution_plotter/list_pairs.json
```

Note that the signals in the result are the ones rotated by the physics model,
therefore if you will wanna draw the VBF sognal with C2V=2 you must ask that of the fit, eg, add to the fitdiagnosis making: `--parameter-values C2V=2.0`. By default it does prefit quantities, see options of `--type-fit` to change that. \\

- The `list_pairs.json` is a list of a template dictionary `` and a list of substituons of keywords of it ``.
This second is optional, if it is not given it will use the template literally to make the result.
- In the way it is implemented we can use more than one pair (template dictionary, keywords to substitute). Eg. using very different options on the template for doing tables and plots, making easier to make a second pair, and/or adding different channels that make part of a same fit.

The result will be:
- A folder with plots as booked on the dictionaries, for each plot a json an tex file with the yields.
  - Not all the options are contemplated on this example, see [here](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/scripts/README_postfit_plots.md) for descriptions of the dictionaries and all options.
- For each plot it saves also the dictionary to tweak and reproduce with the standalone script
  - In the same folder a log file with all commands for reruning the plots (that is what control if the task is done)
