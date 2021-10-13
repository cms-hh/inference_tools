The exclusion plots combine results from scans of [upper limits](./limits.md) and [likelihood profiles](./likelihood.md).


### Comparison of exclusion performance

This chain of tasks leads to a plot that shows the excluded regions of a scanned parameter and its best fit value for multiple *configurations*.
These can be different versions of datacards, or even channels or analyses as shown in the following.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotExclusionAndBestFit \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF \
    --pois r_gghh \
    --campaign FAKE
```

Note that `kl,-25,25` is the default scan parameter.

Output:

![Best fit values and exlusions](../images/exclusionbestfit__poi_r_gghh__scan_kl_-30.0_30.0_n61__params_r1.0_r_qqhh1.0_kt1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotExclusionAndBestFit) --> B1(MergeUpperLimits);
    A -. optional .-> C1(MergeLikelihoodScan);
    B1 --> D1([UpperLimits]);
    D1 --> F1(CreateWorkspace);
    F1 --> G1(CombineDatacards);
    C1 --> E1([LikelihoodScan]);
    E1 --> F1;
    A --> B2(MergeUpperLimits);
    A -. optional .-> C2(MergeLikelihoodScan);
    B2 --> D2([UpperLimits]);
    D2 --> F2(CreateWorkspace);
    F2 --> G2(CombineDatacards);
    C2 --> E2([LikelihoodScan]);
    E2 --> F2;
    A --> ...;
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.
Please note that dependencies to [snapshots](snapshot.md) are omitted in this diagram for visual clearness.


#### Parameters

=== "PlotExclusionAndBestFit"

    --8<-- "content/snippets/plotexclusionandbestfit_param_tab.md"

=== "MergeUpperLimits"

    --8<-- "content/snippets/mergeupperlimits_param_tab.md"

=== "UpperLimits"

    --8<-- "content/snippets/upperlimits_param_tab.md"

=== "MergeLikelihoodScan"

    --8<-- "content/snippets/mergelikelihoodscan_param_tab.md"

=== "LikelihoodScan"

    --8<-- "content/snippets/likelihoodscan_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Executing `PlotExclusionAndBestFit` of `C2V` from `-5..5` with 4 local cores and changing the labels:

```shell hl_lines="6-9"
law run PlotExclusionAndBestFit \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF \
    --pois r \
    --scan-parameters C2V,-5,5 \
    --datacard-names All,ggF,VBF \
    --LikelihoodScan-workflow local \
    --UpperLimits-workflow local \
    --workers 4
```


**2.** Executing `PlotExclusionAndBestFit` tasks on HTCondor, managed by 4 local workers, and changing the labels:

```shell hl_lines="6-9"
law run PlotExclusionAndBestFit \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF \
    --pois r \
    --scan-parameters C2V,-5,5 \
    --datacard-names All,ggF,VBF \
    --LikelihoodScan-workflow htcondor \
    --UpperLimits-workflow htcondor \
    --workers 4
```


### 2D parameter exclusion

The `PlotExclusionAndBestFit2D` gathers data from [upper limit](./limits.md) and [likelihood profiling](./likelihood.md) tasks to create a plot showing the excluded region of two scan parameters as well as the position and errors of their best fit values.

- [Quick example](#quick-example_1)
- [Dependencies](#dependencies_1)
- [Parameters](#parameters_1)
- [Example commands](#example-commands_1)


#### Quick example

```shell
law run PlotExclusionAndBestFit2D \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --scan-parameters kl,-30,30,61:kt,-6,9,31
```

Output:

![2D best fit values and exlusions](../images/exclusionbestfit2d__poi_r__scan_kl_-30.0_30.0_n61_kt_-6.0_9.0_n31__params_r_qqhh1.0_r_gghh1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotExclusionAndBestFit2D) --> B(MergeUpperLimits);
    A -. optional .-> C(MergeLikelihoodScan);
    B --> D([UpperLimits]);
    D --> F(CreateWorkspace);
    F --> G(CombineDatacards);
    C --> E([LikelihoodScan]);
    E --> F;
    D -. optional .-> H([Snapshot]);
    E -. optional .-> I([Snapshot]);
    H --> F;
    I --> F;
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotExclusionAndBestFit2D"

    --8<-- "content/snippets/plotexclusionandbestfit2d_param_tab.md"

=== "MergeUpperLimits"

    --8<-- "content/snippets/mergeupperlimits_param_tab.md"

=== "UpperLimits"

    --8<-- "content/snippets/upperlimits_param_tab.md"

=== "MergeLikelihoodScan"

    --8<-- "content/snippets/mergelikelihoodscan_param_tab.md"

=== "LikelihoodScan"

    --8<-- "content/snippets/likelihoodscan_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Executing `PlotExclusionAndBestFit2D` of `kl` and `C2V` on HTCondor with each job processing 3 tasks, managed by 2 local processes:

```shell hl_lines="5-10"
law run PlotExclusionAndBestFit \
    --version dev \
    ---datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --scan-parameters C2V,-5,5:CV,-5,5 \
    --LikelihoodScan-workflow htcondor \
    --LikelihoodScan-tasks-per-job 3 \
    --UpperLimits-workflow htcondor \
    --UpperLimits-tasks-per-job 3 \
    --workers 2 \
    --campaign FAKE
```
