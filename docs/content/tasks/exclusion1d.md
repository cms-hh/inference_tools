### Combined plot: fit and exclusion

The task `PlotBestFitAndExclusion` creates a plot showing the best fit values of a POI as well as its excluded region for multiple *configurations* - these can be different versions of datacards, or even channels or analyses as shown in the following plot.


```shell hl_lines="1-8"
law run PlotExclusionAndBestFit \
    --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/*/datacard.txt \
    --datacard-names ee,emu,mumu,Combined \
    --pois r_gghh \
    --campaign FAKE
```

![Best fit values and exlusions](../images/bestfitexclusion__r_gghh__kl_n51_-25.0_25.0.png)

The best fit values are extracted via likelihood profiling done by the [`LikelihoodScan1D`](likelihood1d.md) task. Excluded regions are inferred from the [`UpperLimits`](limits.md) task which scans limits in steps of the same POI as measured on a `--pois`, i.e., `r` itself, `r_gghh`or `r_qqhh`.


??? hint "Click to expand"

    ```mermaid
    graph TD;
        A(PlotExclusionAndBestFit) --> B(MergeUpperLimits);
        A(PlotExclusionAndBestFit) --> C(MergeLikelihoodScan1D);
        B --> D(UpperLimits);
        C --> E(LikelihoodScan1D);
        D --> F(CreateWorkspace);
        E --> F;
        F --> G(CombineDatacards);
    ```

    === "PlotExclusionAndBestFit"

        The `PlotExclusionAndBestFit` task collects the fit results from the `MergeUpperLimits` and `MergeLikelihoodScan1D` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-11,20-22"

        The description of the other tasks can be found in [Upper Limits](limits.md) and [Likelihood Scans](likelihood.md).
