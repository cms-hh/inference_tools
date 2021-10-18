The tasks documented below can be used to produce and visualize the limits corresponding to certain discrete EFT benchmark.
Compared to the [other tasks for obtaining limits](limits.md) which rely on the HH physics model for inter- and extrapolating the effect of variations of the *kappa* values, the EFT benchmark tasks extract information of the particular benchmarks directly from the name of the used datacard files.
This entails two major differences in the preparation of datacards and the steering of the tasks via parameters.

**Datacards**

The datacards for the various EFT benchmarks should be prepared according to the central [EFT documentation](https://gitlab.cern.ch/hh/eft-benchmarks).
In particular, please make sure that your ggF signal is normalized to a hypothetical cross section of 1fb times the branching ratio of your channel, and that VBF processes (`qqHH_*`) are dropped except for the SM VBF signal, which should be marked as background by attributing a positive process id.
Names of EFT benchmark datacard files should have the format

`datacard_<NAME>.txt`,

where `NAME` is the name of the particular benchmark.

When working with the provided law tasks, the accepted datacard naming scheme can be slightly adapted to cover scenarios where several datacards of the same EFT benchmark are located in the same directory.
See the **task parameters** section below for more information.

==If your datacards contribute to the HH combination==, please make sure to use the ==exact same== naming scheme for processes, bins and parameters as the other datacards provided by your channel.

**Task parameters**

As benchmark names are extracted from names of the datacard files, the usual `--datacards` parameter cannot be used as it would not support the combination of cards across multiple channels.

The tasks below use the `--multi-datacards` parameter instead, allowing multiple sequences of files, separated by `:`, to be passed in the format `ch1/cardA,ch1/cardB:ch2/cardA,ch2/cardB`.
In this example, the different sequences `ch1/cardA,ch1/cardB` and `ch2/cardA,ch2/cardB` could correspond to different analysis channels.
**Files with the same (base)name across sequences will be combined** by means of the `CombineDatacards` task.
Therefore, a valid example is

```shell
--multi-datacards 'my_cards/datacard_bm*.txt'
```

for a **single channel**, and

```shell
--multi-datacards 'bbbb/datacard_bm*.txt:bbgg/datacard_bm*.txt'
```

for **multiple channels**, where datacards corresponding to the same benchmark (name) will be combined across the channels.

When datacards of the same EFT benchmark are located in the same directory (unlike the example above which assumes that files are placed in different subdirectories), you can use the `--datacard-pattern` parameter to select the datacards per sequence and to extract either the benchmark name.

Consider a directory that contains 6 files

```
datacard_bm1_A.txt
datacard_bm2_A.txt
datacard_bm3_A.txt
datacard_bm1_B.txt
datacard_bm2_B.txt
datacard_bm3_B.txt
```

and you want to compute the benchmark limits **only** for datacards `A`.
In this case, one can use

```shell
--multi-datacards 'datacard_bm*.txt' --datacard-pattern 'datacard_bm(.*)_A.txt'
```

where the pattern `datacard_bm(.*)_A.txt` is used both to select files from all matches of `--multi-datacards` and to extract the corresponding benchmark name via the regex group `(.*)`.

If you like to perform the scan for `A` **and* `B`, with datacards of the same benchmark being combined first, you can add another pattern separated by comma,

```shell
--multi-datacards 'datacard_bm*.txt' --datacard-pattern 'datacard_bm(.*)_A.txt,datacard_bm(.*)_B.txt'
```


### Benchmark limits

The `PlotEFTBenchmarkLimits` task shows the upper limits on the rate of HH production via gluon-gluon fusion (POI `r_gghh`) obtained for several EFT benchmarks.
As described above, datacard names should have the format `datacard_<NAME>.txt`.

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
    C --> D1(CreateWorkspace);
    C --> D2(CreateWorkspace);
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
