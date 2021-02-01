The `PlotPullsAndImpacts` task performs fits and shows both nuisance pulls and their impact on the POI by fixing parameters to their post fit value and extracting the resulting change of the POI.
Nuisance parameters to evaluate are extracted dynamically from the workspace.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotPullsAndImpacts \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r
```

Output:

![Pulls and impacts](../images/pulls_impacts__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotPullsAndImpacts) --> B(MergePullsAndImpacts);
    B --> C([PullsAndImpacts]);
    C --> D(CreateWorkspace);
    D --> E(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotPullsAndImpacts"

    --8<-- "content/snippets/plotpullsandimpacts_param_tab.md"

=== "MergePullsAndImpacts"

    --8<-- "content/snippets/mergepullsandimpacts_param_tab.md"

=== "PullsAndImpacts"

    --8<-- "content/snippets/pullsandimpacts_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `PullsAndImpacts` including all MC stats nuisances on htcondor.

```shell hl_lines="5-6"
law run PlotPullsAndImpacts \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --mc-stats \
    --PullsAndImpacts-workflow htcondor
```
