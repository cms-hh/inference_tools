### Testing a datacard

The `PlotGoodnessOfFit` task shows the test statistic value of a goodness-of-fit test between data and simulation as well as for a configurable number of toys.
The fit model is extracted from a single set of datacards.
A gaussian fit is performed to model the toy distribution and to infer a distance to the data test statistic in units of standard deviations.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotGoodnessOfFit \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --toys 1000 \
    --toys-per-task 20
```

Output:

![Goodness-of-fit test](../images/gofs__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotGoodnessOfFit) --> B(MergeGoodnessOfFit);
    B --> C([GoodnessOfFit]);
    C --> D(CreateWorkspace);
    D --> E(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotGoodnessOfFit"

    --8<-- "content/snippets/plotgoodnessoffit_param_tab.md"

=== "MergeGoodnessOfFit"

    --8<-- "content/snippets/mergegoodnessoffit_param_tab.md"

=== "GoodnessOfFit"

    --8<-- "content/snippets/goodnessoffit_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Run the test with the `KS` algorithm, executing tasks on htcondor.

```shell hl_lines="6-7"
law run PlotGoodnessOfFit \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS
    --toys 1000 \
    --toys-per-task 20 \
    --algorithm KS \
    --GoodnessOfFit-workflow htcondor
```


### Testing multiple datacards

To show the results goodness-of-fit tests of multiple sequences of datacards (e.g. to compare different categories or event analysis channels), use the `PlotMultiplGoodnessOfFits` task.

Instead of a parameter `--datacards`, this task introduces a `--multi-datacards` parameter.
It takes several CSV sequences of datacard paths, separated by a colon, e.g. `--multi-datacards card_ee_1.txt,card_ee_2.txt:card_mumu_1.txt,card_mumu_2.txt`.
In this example, the two `card_ee_*.txt` and the two `card_mumu_*.txt` cards will result in two dedicated measurmments, following the same task requirements, i.e., `GoodnessOfFit` and `MergeGoodnessOfFit`, as described above.

- [Quick example](#quick-example_1)
- [Dependencies](#dependencies_1)
- [Parameters](#parameters_1)
- [Example commands](#example-commands_1)


#### Quick example

```shell
law run PlotMultipleGoodnessOfFits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF:$DHI_EXAMPLE_CARDS \
    --datacard-names ggF,VBF,Combined \
    --toys 1000 \
    --toys-per-task 20
```

Output:

![Mutliple goodness-of-fit tests](../images/multigofs__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0__t1000_pt20.png)


#### Dependencies

```mermaid
graph LR;
    A(PlotMultiplGoodnessOfFits) --> B1(MergeGoodnessOfFit);
    A --> B2(MergeGoodnessOfFit);
    A --> ...;
    B1 --> C1([GoodnessOfFit]);
    B2 --> C2([GoodnessOfFit]);
    C1 --> D1(CreateWorkspace);
    C2 --> D2(CreateWorkspace);
    D1 --> E1(CombineDatacards);
    D2 --> E2(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotMultipleGoodnessOfFits"

    --8<-- "content/snippets/plotmultiplegoodnessoffits_param_tab.md"

=== "MergeGoodnessOfFit"

    --8<-- "content/snippets/mergegoodnessoffit_param_tab.md"

=== "GoodnessOfFit"

    --8<-- "content/snippets/goodnessoffit_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Run the test with the `KS` algorithm, executing tasks on htcondor.

```shell hl_lines="7-8"
law run PlotMultipleGoodnessOfFits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_GGF:$DHI_EXAMPLE_CARDS_VBF:$DHI_EXAMPLE_CARDS \
    --datacard-names ggF,VBF,Combined \
    --toys 1000 \
    --toys-per-task 20 \
    --algorithm KS \
    --GoodnessOfFit-workflow htcondor
```
