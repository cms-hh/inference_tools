### Limit on POI vs. scan parameter

The `PlotUpperLimits` task shows the upper limits on a POI computed over a range of values of a scan parameter.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotUpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --xsec fb \
    --y-log
```

Note that the above command uses `r` as the default POI and `kl,-25,25` as the default scan parameter and range.
See the task parameters below for more info.

Output:

![Upper limits](../images/limits__r__kl_n51_-25.0_25.0__fb_bbwwllvv_log.png)


#### Dependencies

```mermaid
graph LR;
    A{{PlotUpperLimits}} --> B(MergeUpperLimits);
    B --> C([UpperLimits]);
    C -. either .-> D([CreateWorkspace]);
    D --> E(CombineDatacards);
    C -. or .-> F([Snapshot]);
    F --> D;
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.
Hexagonal boxes mark tasks that can produce [HEPData](https://hepdata-submission.readthedocs.io/en/latest/) compatible yaml files.


#### Parameters

=== "PlotUpperLimits"

    --8<-- "content/snippets/plotupperlimits_param_tab.md"

=== "MergeUpperLimits"

    --8<-- "content/snippets/mergeupperlimits_param_tab.md"

=== "UpperLimits"

    --8<-- "content/snippets/upperlimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Limit on `r_qqhh` vs. `C2V` with 4 local cores:

```shell hl_lines="4-6"
law run PlotUpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r_qqhh \
    --scan-parameters C2V,-10,10,21 \
    --workers 4
```

**2.** Execute `UpperLimit` tasks on HTCondor, with one job handling two tasks sequentially:

```shell hl_lines="4-5"
law run PlotUpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --UpperLimits-workflow htcondor \
    --UpperLimits-tasks-per-job 2
```


### Multiple limits on POI vs. scan parameter

There are two plots that provide a visual comparison between multiple *configurations* - these can be different versions of datacards, or even channels or analyses.

The first one, `PlotMultipleUpperLimits`, draws multiple limits just as done with `PlotUpperLimits`.
The only difference is that only the median limit is shown to provide a better visual aid.

Instead of a parameter `--datacards`, this task introduces a `--multi-datacards` parameter.
It takes several CSV sequences of datacard paths, separated by a colon, e.g. `--multi-datacards card_ee_1.txt,card_ee_2.txt:card_mumu_1.txt,card_mumu_2.txt`.
In this example, the two `card_ee_*.txt` and the two `card_mumu_*.txt` cards will result in two dedicated measurmments, following the same task requirements, i.e., `UpperLimits` and `MergeUpperLimits`, as described above.

- [Quick example](#quick-example_1)
- [Dependencies](#dependencies_1)
- [Parameters](#parameters_1)
- [Example commands](#example-commands_1)


#### Quick example

```shell
law run PlotMultipleUpperLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF \
    --xsec fb \
    --y-log
```

Note that the above command uses `r` as the default POI and `kl,-25,25` as the default scan parameter and range.
See the task parameters below for more info.

Output:

![Upper limit comparison](../images/multilimits__poi_r__scan_kl_-25.0_25.0_n51__params_r_qqhh1.0_r_gghh1.0_kt1.0_CV1.0_C2V1.0__fb_log.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotMultipleUpperLimits) --> B1(MergeUpperLimits);
    A --> B2(MergeUpperLimits);
    A --> ...;
    B1 --> C1([UpperLimits]);
    B2 --> C2([UpperLimits]);
    C1 -. either .-> D1([CreateWorkspace]);
    C2 -. either .-> D2([CreateWorkspace]);
    D1 --> E1(CombineDatacards);
    D2 --> E2(CombineDatacards);
    C1 -. or .-> F1([Snapshot]);
    C2 -. or .-> F2([Snapshot]);
    F1 --> D1;
    F2 --> D2;
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotMultipleUpperLimits"

    --8<-- "content/snippets/plotmultipleupperlimits_param_tab.md"

=== "MergeUpperLimits"

    --8<-- "content/snippets/mergeupperlimits_param_tab.md"

=== "UpperLimits"

    --8<-- "content/snippets/upperlimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `UpperLimit` tasks on HTCondor, with one job handling two tasks sequentially:

```shell hl_lines="5-6"
law run PlotUpperLimits \
    --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt \
    --datacard-names ee,emu,mumu \
    --UpperLimits-workflow htcondor \
    --UpperLimits-tasks-per-job 2
```


### Multiple limits at a certain point of parameters

The second task, `PlotUpperLimitsAtPoint`, creates a plot comparing upper limits at a certain point of model parameters.
Just as for the `PlotMultipleUpperLimits` task above, it includes a parameter `--multi-datacards` that accepts multiple CSV sequences of datacard paths, separated with a colon to denote multiple measurements.

- [Quick example](#quick-example_2)
- [Dependencies](#dependencies_2)
- [Parameters](#parameters_2)
- [Example commands](#example-commands_2)


#### Quick example

```shell
law run PlotUpperLimitsAtPoint \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF
```

Note that the above command uses `r` as the default POI.
See the task parameters below for more info.

Output:

![Upper limits at POI](../images/limitsatpoint__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
graph LR;
    A{{PlotUpperLimitsAtPoint}} --> B1([UpperLimits]);
    A --> B2([UpperLimits]);
    A --> ...;
    B1 -. either .-> C1([CreateWorkspace]);
    B2 -. either .-> C2([CreateWorkspace]);
    C1 --> D1(CombineDatacards);
    C2 --> D2(CombineDatacards);
    B1 -. or .-> E1([Snapshot]);
    B2 -. or .-> E2([Snapshot]);
    E1 --> C1;
    E2 --> C2;
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.
Hexagonal boxes mark tasks that can produce [HEPData](https://hepdata-submission.readthedocs.io/en/latest/) compatible yaml files.


#### Parameters

=== "PlotUpperLimitsAtPoint"

    --8<-- "content/snippets/plotupperlimitsatpoint_param_tab.md"

=== "UpperLimits"

    --8<-- "content/snippets/upperlimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Change the order of limits in the plot without changing `--multi-datacards` and update labels:

```shell hl_lines="4-5"
law run PlotUpperLimitsAtPoint \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS:$DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF \
    --datacard-names All,ggF,VBF \
    --datacard-order 1,0,2
```
