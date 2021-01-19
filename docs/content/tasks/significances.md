## Significance scans

The default command to run the entire task chain up to `PlotSignificanceScan` is

```shell hl_lines="1"
law run PlotSignificanceScan --version dev --datacards $DHI_EXAMPLE_CARDS
```

![Significance scan](../images/significances__r__kl_n17_-2.0_6.0.png)


??? hint "Click to expand"

    The example assumes the following directory structure:

    ```mermaid
    graph TD;
        A(PlotSignificanceScan) --> B(MergeSignificanceScan);
        B --> C(SignificanceScan);
        C --> D(CreateWorkspace);
        D --> E(CombineDatacards);
    ```

    === "PlotSignificanceScan"

        The `PlotSignificanceScan` task collects the fit results from the `MergeSignificanceScan` and visualizes them in a plot.
        It provides some handy cli parameters to manipulate the visualisation:

        --8<-- "content/snippets/parameters.md@-11"


    === "MergeSignificanceScan"

        The `MergeSignificanceScan` task collects the fit results from each of the `SignificanceScan` and merges them.


    === "SignificanceScan"

        The `SignificanceScan` runs the fits for each point in the defined range.
        It provides some handy cli parameters to manipulate POIs, ranges and other options:

        --8<-- "content/snippets/parameters.md@-1,12-17"



### Running on HTCondor

When working with large workspaces, each particular branch can take quite a while to process.
To run all tasks as jobs over HTCondor, just add `--workflow htcondor` to the command.
To control the *workflow type* when executing an upstream task, use full parameter location `--SignificanceScan-workflow htcondor` instead.

When configured to run on HTCondor, a few additional **parameters** are enabled.

- `--poll-interval INT/STRING`: The time between status polls in minutes. Allows verbose duration strings such as e.g. `45s`. Defaults to `1min`.
- `--retries INT`: The number of retries per job. Defaults to `3`.
- `--max-runtime INT/STRING`: The maximum job runtime in hours. Allows verbose duration strings such as e.g. `45mins`. Defaults to `2h`.
- `--parallel-jobs`: The maximum number of parallel jobs being processed. Defaults to `-1` which means that all jobs are run in parallel.
- `--tasks-per-job`: The number of tasks that each job is processing. Defaults to `1`, meaning that each job runs exactly one task.
