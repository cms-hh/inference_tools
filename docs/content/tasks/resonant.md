The tasks documented below can be used to produce and visualize the limits corresponding to certain resonance mass hypotheses.
Compared to the [other tasks for obtaining limits](limits.md) which rely on the HH physics model for inter- and extrapolating the effect of variations of the *kappa* values, the resonant limit tasks extract information of the particular hypothesis directly ==from the name of the used datacard files==.
This entails two major differences in the preparation of datacards and the steering of the tasks via parameters.

**Datacards**

The datacards for the various mass hypotheses should be prepared according to the central [naming and datacard conventions](https://gitlab.cern.ch/hh/naming-conventions#hh-signals-for-resonant-results).
In particular, please make sure that

- your ggF signal is normalized to a hypothetical cross section of 1 **pb** (❗️) times the **branching ratio of your channel**, and
- that VBF processes (`qqHH_*`) are dropped except for the SM VBF signal, which should be marked as background by attributing a positive process id to it.

Names of datacard files should have the format

```
datacard_<some_other_info>_<MASS>.txt
```

where `MASS` is the integer mass value of the resonance.

==If your datacards contribute to the HH combination==, please make sure to use the ==exact same== naming scheme for processes, bins and parameters as the other datacards provided by your channel.

**Task parameters**

==Unlike==, for instance, the [upper limit](limits.md#limit-on-poi-vs-scan-parameter) or [likelihood scan](likelihood.md#single-likelihood-profiles) tasks where a single set of combined cards is used to extract results over a range of scan parameter values, each mass point requires its own datacard.

Therefore, the cards passed to `--datacards` are actually parsed using the regular expression configured by the `--datacard-pattern` parameter, which defaults to `^.*_(\d+)\.txt$` (meaning `<any_text>_<mass_integer_value>.txt`).
The mass value is extracted using this expression (matching everything that within the brackets, i.e., the mass value `\d+`) and datacards with the same mass are combined.
For instance, when passing

```
--datacards ch1/datacard_250.txt,ch1/datacard_300.txt,ch2/datacard_250.txt
```

the resonant limit tasks will combine the first and last datacards as they are both corresponding to a mass value of `250`, and then compute two limits for `250` and `300`.

As usual, showing limit scans for multiple sequences (see [below](#multiple-resonant-limits)) requires the `--multi-datacards` parameter to be set instead.
Multiple comma-separeted sequences are themselves separated by a color character, so

```
--multi-datacards ch1/datacard_250.txt,ch1/datacard_300.txt:ch2/datacard_250.txt
```

will perform two separate scans, one at mass points `250` and `300` for cards in `ch1/`, and one scan at a single mass point `250` for the card in `ch2/`.


### Resonant limits

The `PlotResonantLimits` task shows the upper limits on the rate of HH production via gluon-gluon fusion (POI `r_gghh`) obtained for several resonance mass hypotheses.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotResonantLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS_RES \
    --xsec fb
```

Output:

![Resonant limits](../images/limits__res.png)


#### Dependencies

```mermaid
graph LR;
    A{{PlotResonantLimits}} --> B(MergeResonantLimits);
    B --> C([ResonantLimits]);
    C --> D1([CreateWorkspace]);
    C --> D2([CreateWorkspace]);
    C --> ...;
    D1 --> E1(CombineDatacards);
    D2 --> E2(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.
Hexagonal boxes mark tasks that can produce [HEPData](https://hepdata-submission.readthedocs.io/en/latest/) compatible yaml files.


#### Parameters

=== "PlotResonantLimits"

    --8<-- "content/snippets/plotresonantlimits_param_tab.md"

=== "MergeResonantLimits"

    --8<-- "content/snippets/mergeresonantlimits_param_tab.md"

=== "ResonantLimits"

    --8<-- "content/snippets/resonantlimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `ResonantLimits` tasks on HTCondor and apply the branching ratio of the `bbgg` channel to extracted limits:

```shell hl_lines="5-6"
law run PlotResonantLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS_RES \
    --xsec fb \
    --br bbgg \
    --ResonantLimits-workflow htcondor
```


### Multiple resonant limits

The `PlotMultipleResonantLimits` task shows the upper limits on the rate of HH production via gluon-gluon fusion obtained for several resonance mass hypotheses, but unlike `PlotResonantLimits` described [above](#benchmark-limits), results of datacards are not combined per mass point, but shown separately.

- [Quick example](#quick-example1)
- [Dependencies](#dependencies_1)
- [Parameters](#parameters_1)
- [Example commands](#example-commands_1)


#### Quick example

```shell
law run PlotMultipleResonantLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_RES_1:$DHI_EXAMPLE_CARDS_RES_2 \
    --xsec fb
```

As described above, the `--multi-datacards` parameter should be used to identify different sequences of datacards.

Output:

![Multiple resonant limits](../images/multilimits__res.png)


#### Dependencies

```mermaid
graph LR;
    A{{PlotResonantLimits}} --> B1(MergeResonantLimits);
    A --> B2(MergeResonantLimits);
    A --> X1(...);
    B1 --> C1([ResonantLimits]);
    B2 --> C2([ResonantLimits]);
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
Hexagonal boxes mark tasks that can produce [HEPData](https://hepdata-submission.readthedocs.io/en/latest/) compatible yaml files.


#### Parameters

=== "PlotMultipleResonantLimits"

    --8<-- "content/snippets/plotmultipleresonantlimits_param_tab.md"

=== "MergeResonantLimits"

    --8<-- "content/snippets/mergeresonantlimits_param_tab.md"

=== "ResonantLimits"

    --8<-- "content/snippets/resonantlimits_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Execute `ResonantLimits` tasks on HTCondor:

```shell hl_lines="5"
law run PlotMultipleResonantLimits \
    --version dev \
    --multi-datacards $DHI_EXAMPLE_CARDS_RES_1:$DHI_EXAMPLE_CARDS_RES_2 \
    --xsec fb \
    --ResonantLimits-workflow htcondor
```
