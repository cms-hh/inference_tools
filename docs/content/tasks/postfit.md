The `PlotPostfitSOverB` task reads prefit and postfit shapes from combine's `FitDiagnostics` output, orders their bins by their prefit S-over-B ratio, merges them in a configurable way, and shows the bin contents separately for background-only, the fitted signal and recorded data.

- [Quick example](#quick-example)
- [Dependencies](#dependencies)
- [Parameters](#parameters)
- [Example commands](#example-commands)


#### Quick example

```shell
law run PlotPostfitSOverB \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS
```

Note that the above command uses `r` as the default POI

Output:

![Postfit S-over-B](../images/postfitsoverb__poi_r__params_r_qqhh1.0_r_gghh1.0_kl1.0_kt1.0_CV1.0_C2V1.0.png)


#### Dependencies

```mermaid
    graph LR;
    A(PlotPostfitSOverB) --> B([PostFitShapes]);
    B --> C(CreateWorkspace);
    C --> D(CombineDatacards);
```

Rounded boxes mark [workflows](practices.md#workflows) with the option to run tasks as HTCondor jobs.


#### Parameters

=== "PlotPostfitSOverB"

    --8<-- "content/snippets/plotpostfitsoverb_param_tab.md"

=== "PostFitShapes"

    --8<-- "content/snippets/postfitshapes_param_tab.md"

=== "CreateWorkspace"

    --8<-- "content/snippets/createworkspace_param_tab.md"

=== "CombineDatacards"

    --8<-- "content/snippets/combinedatacards_param_tab.md"


#### Example commands

**1.** Configure custom bin edges and set the minimum y-axis value of the ratio plot.

```shell hl_lines="4-5"
law run PlotPostfitSOverB \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --bins "-5,-3,-2,-1.5,-1" \
    --ratio-min 0.5
```

Note the quotes around the bin edges which are required since the first value starts with a dash ("-") which is misinterpreted by the `ArgumentParser` in Python 2.
