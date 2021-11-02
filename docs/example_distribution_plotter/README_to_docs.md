
We use the plotter as a task in this example as:

```
law run  PlotDistributionsAndTables \
--version plotting_FAKE \
--datacards $DHI_BASE/docs/example_distribution_plotter/datacard.txt \
--draw_for_channels $DHI_BASE/docs/example_distribution_plotter/list_pairs.json
```

Note that the signals in the result are the ones rotated by the physics model,
therefore if you will wanna draw the VBF sognal with C2V=2 you must ask that of the fit, eg, add to the fitdiagnosis making: `--parameter-values C2V=2.0`.

The result will be:
- A folder with plots as booked on the dictionaries
  - Not all the options are contemplated on this example, see [here](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/scripts/README_postfit_plots.md) for descriptions of the dictionaries and all options.
- In the same folder a log file with all the (that is what control if the task is done)
