### Limits vs. POI

The `PlotUpperLimits` task takes the outputs from `UpperLimits` and `MergeUpperLimits` and plots them the usual way:

```shell hl_lines="1-5"
law run PlotUpperLimits --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --xsec fb \
    --br bbwwllvv \
    --y-log \
```

![Upper limits](../images/limits__r__kl_n51_-25.0_25.0__fb_bbwwllvv_log.png)

??? hint "Click to expand"

    The example assumes the following directory structure:

    ```mermaid
    graph TD;
        A(PlotUpperLimits) --> B(MergeUpperLimits);
        B --> C(UpperLimits);
        C --> D(CreateWorkspace);
        D --> E(CombineDatacards);
    ```

    === "PlotUpperLimits"

        The `PlotUpperLimits` task collects the fit results from the `MergeUpperLimits` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-11"


    === "MergeUpperLimits"

        The `MergeUpperLimits` task collects the fit results from each of the `UpperLimits` and merges them.


    === "UpperLimits"

        The `UpperLimits` runs the fits for each point in the defined range.
        It provides some handy cli parameters to manipulate POIs, ranges and other options:

        --8<-- "content/parameters.md@-1,12-17"


    === "CreateWorkspace"

        The `CreateWorkspace` task takes the combined datacard and the PhysicsModel as input and creates a workspace for the fit.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,18"    


    === "CombineDatacards"

        The `CombineDatacards` task takes multiple datacards as input and simply combines them for further processing.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,19"  




### Multiple limits vs. POI

There are two plots that provide a visual comparison between multiple *configurations* - these can be different versions of datacards, or even channels or analyses.

The first one, `PlotMultipleUpperLimits`, draws multiple limits just as done with `PlotUpperLimits`.
The only difference is that only the median limit is shown to provide a better visual aid.

Instead of a parameter `--datacards`, this task introduces a `--multi-datacards` parameter.
It takes several CSV sequences of datacard paths, separated with a colon, e.g. `--multi-datacards card_ee_1.txt,card_ee_2.txt:card_mumu_1.txt,card_mumu_2.txt`.
In this example, the two `card_ee_*.txt` and the two `card_mumu_*.txt` cards will result in two dedicated measurmments, following the same task requirements, i.e., `UpperLimits` and `MergeUpperLimits`, as described above.

```shell hl_lines="1-3"
law run PlotMultipleUpperLimits --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt \
    --datacard-names ee,emu,mumu --xsec fb --y-log
```

![Upper limit comparison](../images/multilimits__r__kl_n51_-25.0_25.0__fb_log.png)

??? hint "Click to expand"

    ```mermaid
    graph TD;
        A(PlotMultipleUpperLimits) --> B1(MergeUpperLimits);
        A(PlotMultipleUpperLimits) --> B2(MergeUpperLimits);
        A(PlotMultipleUpperLimits) --> B3(MergeUpperLimits);
        A(PlotMultipleUpperLimits) --> ...;
        B1 --> C1(UpperLimits);
        B2 --> C2(UpperLimits);
        B3 --> C3(UpperLimits);
        C1 --> D1(CreateWorkspace);
        C2 --> D2(CreateWorkspace);
        C3 --> D3(CreateWorkspace);
        D1 --> E1(CombineDatacards);
        D2 --> E2(CombineDatacards);
        D3 --> E3(CombineDatacards);
    ```

    === "PlotMultipleUpperLimits"

        The `PlotMultipleUpperLimits` task collects the fit results from multiple `MergeUpperLimits` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-11,20-22"


    === "MergeUpperLimits"

        The `MergeUpperLimits` task collects the fit results from each of the `UpperLimits` and merges them.


    === "UpperLimits"

        The `UpperLimits` runs the fits for each point in the defined range.
        It provides some handy cli parameters to manipulate POIs, ranges and other options:

        --8<-- "content/parameters.md@-1,12-17"


    === "CreateWorkspace"

        The `CreateWorkspace` task takes the combined datacard and the PhysicsModel as input and creates a workspace for the fit.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,18"    


    === "CombineDatacards"

        The `CombineDatacards` task takes multiple datacards as input and simply combines them for further processing.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,19" 



### Multiple limits at a certain POI value

The second task, `PlotUpperLimitsAtPOI`, creates a plot comparing upper limits at a certain POI value.
Just as for the `PlotMultipleUpperLimits` task above, it includes a parameter `--multi-datacards` that accepts multiple CSV sequences of datacard paths, searated with a colon to denote multiple measurements.

Example:

```shell hl_lines="1-3"
law run PlotUpperLimitsAtPOI --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt \
    --datacard-names ee,emu,mumu
```

![Upper limits at POI](../images/limitatpoi__r__kl_1.0.png)

??? hint "Click to expand"

    ```mermaid
    graph TD;
        A(PlotMultipleUpperLimits) --> B1(UpperLimits);
        A(PlotMultipleUpperLimits) --> B2(UpperLimits);
        A(PlotMultipleUpperLimits) --> B3(UpperLimits);
        A(PlotMultipleUpperLimits) --> ...;
        B1 --> C1(CreateWorkspace);
        B2 --> C2(CreateWorkspace);
        B3 --> C3(CreateWorkspace);
        C1 --> D1(CombineDatacards);
        C2 --> D2(CombineDatacards);
        C3 --> D3(CombineDatacards);
    ```

    === "PlotMultipleUpperLimits"

        The `PlotMultipleUpperLimits` task collects the fit results from multiple `MergeUpperLimits` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-11,20-22"


    === "UpperLimits"

        The `UpperLimits` runs the fits for each point in the defined range.
        It provides some handy cli parameters to manipulate POIs, ranges and other options:

        --8<-- "content/parameters.md@-1,12-17"


    === "CreateWorkspace"

        The `CreateWorkspace` task takes the combined datacard and the PhysicsModel as input and creates a workspace for the fit.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,18"    


    === "CombineDatacards"

        The `CombineDatacards` task takes multiple datacards as input and simply combines them for further processing.
        The parameters are described as follows:

        --8<-- "content/parameters.md@-1,19" 



### Multiprocessing

As you can see there will be one output file in the `SiblingFileCollection` for each point. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow local --workers 4
```


### HTCondor submission

For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```