### Pulls and Impacts

The default command to run the entire task chain is

```shell hl_lines="1"
law run PlotPullsAndImpacts --version dev --datacards $DHI_EXAMPLE_CARDS
```

![Pulls and impacts](../images/pulls_impacts__kl.png)

??? hint "Click to expand"

    ```mermaid
    graph TD;
        A(PlotPullsAndImpacts) --> B(MergePullsAndImpacts);
        B --> C(PullsAndImpacts);
        C --> D(CreateWorkspace);
        D --> E(CombineDatacards);
    ```

    === "PlotPullsAndImpacts"

        The `PlotPullsAndImpacts` task collects the fit results from `MergePullsAndImpacts` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/parameters.md@-3,23-24,26-31"

    === "PullsAndImpacts"

        The `PullsAndImpacts` runs the fits for each nuisance.
        It provides some handy cli parameters to manipulate POIs, ranges and other options:

        --8<-- "content/parameters.md@-1,12-13,15-17,32"



### Running on HTCondor

When working with large workspaces, each particular branch can take quite a while to process.
To run the 14 tasks as 14 jobs over HTCondor, just add `--workflow htcondor` to the command.
To control the *workflow type* when executing an upstream task, use full parameter location `--PullsAndImpacts-workflow htcondor` instead.

When configured to run on HTCondor, a few additional **parameters** are enabled.

- `--poll-interval INT/STRING`: The time between status polls in minutes. Allows verbose duration strings such as e.g. `45s`. Defaults to `1min`.
- `--retries INT`: The number of retries per job. Defaults to `3`.
- `--max-runtime INT/STRING`: The maximum job runtime in hours. Allows verbose duration strings such as e.g. `45mins`. Defaults to `2h`.
- `--parallel-jobs`: The maximum number of parallel jobs being processed. Defaults to `-1` which means that all jobs are run in parallel.
- `--tasks-per-job`: The number of tasks that each job is processing. Defaults to `1`, meaning that each job runs exactly one task.
