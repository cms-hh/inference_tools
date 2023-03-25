The tasks documented below can be used to produce and visualize the limits corresponding to certain discrete EFT benchmark.
Compared to the [other tasks for obtaining limits](limits.md) which rely on the HH physics model for inter- and extrapolating the effect of variations of the *kappa* values, the EFT benchmark tasks extract information of the particular benchmarks directly from the name of the used datacard files.
This entails two major differences in the preparation of datacards and the steering of the tasks via parameters.

**Datacards**

The datacards for the various EFT benchmarks should be prepared according to the central [naming and datacard conventions](https://gitlab.cern.ch/hh/naming-conventions#hh-signals-for-eft-results) as well as the [EFT documentation](https://gitlab.cern.ch/hh/eft-benchmarks).
In particular, please make sure that

- your ggF signal is normalized to a hypothetical cross section of 1 **pb** (❗️) times the **branching ratio of your channel**, and
- that VBF processes (`qqHH_*`) are dropped except for the SM VBF signal, which should be marked as background by attributing a positive process id to it.

Names of EFT benchmark datacard files should have the format

```
datacard_<NAME>.txt
```

where `NAME` is the name of the particular benchmark.

For automatic benchmark grouping (e.g. into JHEP03 or JHEP04) within plots, you should use `NAME`'s such as `JHEP04BM1`, `JHEP04BM2`, etc, which are special names recognized by the plot routines. See `bm_labels` in [dhi/config.py](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/config.py) for a list of all known `NAME`'s.


==If your datacards contribute to the HH combination==, please make sure to use the ==exact same== naming scheme for processes, bins and parameters as the other datacards provided by your channel.

**Task parameters**

==Unlike==, for instance, the [upper limit](limits.md#limit-on-poi-vs-scan-parameter) or [likelihood scan](likelihood.md#single-likelihood-profiles) tasks where a single set of combined cards is used to extract results over a range of scan parameter values, each benchmark point requires its own datacard.
To allow that cards of different channels can still be combined per benchmark scenario, the usual `--datacards` parameter cannot be used as it targets only a single sequence of cards.

Instead, the tasks below use the `--multi-datacards` parameter that allows to define multiple sequences of files, separated by `:`, to be passed in the format

```
ch1/datacard_A.txt,ch1/datacard_B.txt:ch2/datacard_A.txt,ch2/datacard_B.txt:...
```

In this example, the different sequences `ch1/datacard_A.txt,ch1/datacard_B.txt` and `ch2/datacard_A.txt,ch2/datacard_B.txt` could correspond to different analysis channels.
**Datacards with the same `NAME` (`A` and `B`) across sequences will be combined** by means of the `CombineDatacards` task.
Therefore, a valid example is

```shell
--multi-datacards 'my_cards/datacard_JHEP04*.txt'
```

for a **single channel**, and

```shell
--multi-datacards 'bbbb/datacard_JHEP04*.txt:bbgg/datacard_JHEP04*.txt'
```

for **multiple channels**, where datacards corresponding to the same benchmark will be combined across the channels.
For obvious reasons, the number of files matched by `bbbb/datacard_JHEP04*.txt` and `bbgg/datacard_JHEP04*.txt` must be identical.


### Benchmark limits

The `PlotEFTBenchmarkLimits` task shows the upper limits on the rate of HH production via gluon-gluon fusion (POI `r_gghh`) obtained for several EFT benchmarks.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotEFTBenchmarkLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_EFT_BM \
    --xsec fb
```

As described above, the `--multi-datacards` parameter should be used to identify different sequences of datacards.

Output:

![EFT benchmark limits](../images/limits__eft__benchmarks.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotEFTBenchmarkLimits) --> B(MergeEFTBenchmarkLimits);
    B --> C([EFTBenchmarkLimits]);
    C --> D1([CreateWorkspace]);
    C --> D2([CreateWorkspace]);
    C --> ...;
    D1 --> E1(CombineDatacards);
    D2 --> E2(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotEFTBenchmarkLimits"

    --8<-- "content/snippets/ploteftbenchmarklimits_param_tab.md"

=== "MergeEFTBenchmarkLimits"

    --8<-- "content/snippets/mergeeftbenchmarklimits_param_tab.md"

=== "EFTBenchmarkLimits"

    --8<-- "content/snippets/eftbenchmarklimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `EFTBenchmarkLimits` tasks on HTCondor and apply the branching ratio of the `bbgg` channel to extracted limits:

```shell hl_lines="5-6"
law run PlotEFTBenchmarkLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_EFT_BM \
    --xsec fb \
    --br bbgg \
    --EFTBenchmarkLimits-workflow htcondor
```


### Multiple benchmark limits

The `PlotMultipleEFTBenchmarkLimits` task shows the upper limits on the rate of HH production via gluon-gluon fusion (POI `r_gghh`) obtained for several EFT benchmarks, but unlike `PlotEFTBenchmarkLimits` described [above](#benchmark-limits), results of datacards are not combined per benchmark, but separately shown.

- [Quick example](#quick-example_1)
- [Dependencies](#dependencies_1)
- [Parameters](#parameters_1)
- [Example commands](#example-commands_1)


#### Quick example

```shell
law run PlotMultipleEFTBenchmarkLimits \
    --version dev \
    --multi-datacards $$DHI_EXAMPLE_CARDS_EFT_BM_1:$DHI_EXAMPLE_CARDS_EFT_BM_2 \
    --xsec fb
```

As described above, the `--multi-datacards` parameter should be used to identify different sequences of datacards.

Output:

![Multiple EFT benchmark limits](../images/multilimits__eft__benchmarks.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotEFTBenchmarkLimits) --> B1(MergeEFTBenchmarkLimits);
    A --> B2(MergeEFTBenchmarkLimits);
    A --> X1(...);
    B1 --> C1([EFTBenchmarkLimits]);
    B2 --> C2([EFTBenchmarkLimits]);
    C1 --> D1([CreateWorkspace]);
    C1 --> D2([CreateWorkspace]);
    C1 --> X2(...);
    C2 --> D3([CreateWorkspace]);
    C2 --> D4([CreateWorkspace]);
    C2 --> X3(...);
    D1 --> E1(CombineDatacards);
    D2 --> E2(CombineDatacards);
    D3 --> E3(CombineDatacards);
    D4 --> E4(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotMultipleEFTBenchmarkLimits"

    --8<-- "content/snippets/plotmultipleeftbenchmarklimits_param_tab.md"

=== "MergeEFTBenchmarkLimits"

    --8<-- "content/snippets/mergeeftbenchmarklimits_param_tab.md"

=== "EFTBenchmarkLimits"

    --8<-- "content/snippets/eftbenchmarklimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `EFTBenchmarkLimits` tasks on HTCondor:

```shell hl_lines="5"
law run PlotMultipleEFTBenchmarkLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_EFT_BM_1:$DHI_EXAMPLE_CARDS_EFT_BM_2 \
    --xsec fb \
    --EFTBenchmarkLimits-workflow htcondor
```
