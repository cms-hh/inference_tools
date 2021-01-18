### 1D/2D Likelihood scans

1D:
```shell hl_lines="1"
law run PlotLikelihoodScan --version dev --datacards $DHI_EXAMPLE_CARDS --pois "kl" --scan-parameters "kl,-25,25"
```

![1D likelihood scan](../images/nll1d__kl_n51_-25.0_25.0.png) 


2D:
```shell hl_lines="1"
law run PlotLikelihoodScan --version dev --datacards $DHI_EXAMPLE_CARDS --pois "kl,kt" --scan-parameters "kl,-30,30:kt,-10,10"
```

![2D likelihood scan](../images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log.png)

??? hint "Click to expand"

    The example assumes the following directory structure:

    ```mermaid
    graph TD;
        A(PlotLikelihoodScan) --> B(MergeLikelihoodScan);
        B --> C(LikelihoodScan);
        C --> D(CreateWorkspace);
        D --> E(CombineDatacards);
    ```

    === "PlotLikelihoodScan"

        The `PlotLikelihoodScan` task collects the fit results from the `MergeUpperLimits` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-1,2-8,23-25,9-11"


    === "MergeLikelihoodScan"

        The `MergeLikelihoodScan` task collects the fit results from each of the `LikelihoodScan` and merges them.


    === "LikelihoodScan"

        The `LikelihoodScan` runs the fits for each point in the defined range.
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

As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan --version dev --datacards $DHI_EXAMPLE_CARDS --workflow local --workers 4
```


### HTCondor submission

For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:

```shell hl_lines="1"
law run LikelihoodScan --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```
