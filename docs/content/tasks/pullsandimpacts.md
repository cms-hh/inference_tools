This section documents the usage of the tasks `PullsAndImpacts`, `MergePullsAndImpacts`, and `PlotPullsAndImpacts`.


## Task structure

The default command to run the entire task chain is

```shell hl_lines="1"
law run PlotPullsAndImpacts --version dev
```

and, as usual, you can check the task structure and current output status beforehand by appending `--print-status TASK_DEPTH` to the command. Let's choose -1 to see the structure down to the first task (`CombineDatacards`):

```shell hl_lines="1"
law run PlotPullsAndImpacts --version dev --print-status -4

print task status with max_depth -1 and target_depth 0

> PlotPullsAndImpacts(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_value=1.0, mc_stats=False, file_type=pdf, parameters_per_page=-1, skip_parameters=, order_parameters=, order_by_impact=False)
|   LocalFileTarget(path=$DHI_STORE/PlotPullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/pulls_impacts__kl.pdf)
|     absent
|
|  > MergePullsAndImpacts(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_value=1.0, mc_stats=False)
|  |   LocalFileTarget(path=$DHI_STORE/MergePullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/pulls_impacts__kl.json)
|  |     absent
|  |
|  |  > PullsAndImpacts(branch=-1, start_branch=0, end_branch=14, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_value=1.0, mc_stats=False, workflow=local)
|  |  |   collection: SiblingFileCollection(len=14, threshold=14.0, dir=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev)
|  |  |     absent (0/14)
|  |  |
|  |  |  > CreateWorkspace(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |  |  |   LocalFileTarget(path=$DHI_STORE/CreateWorkspace/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/workspace.root)
|  |  |  |     existent
|  |  |  |
|  |  |  |  > CombineDatacards(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |  |  |  |   LocalFileTarget(path=$DHI_STORE/CombineDatacards/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/datacard.txt)
|  |  |  |  |     existent
```

The overall dependency structure is

```
CombineDatacards
→ CreateWorkspace
→ PullsAndImpacts
→ MergePullsAndImpacts
→ PlotPullsAndImpacts
```


## `PullsAndImpacts`

`PullsAndImpacts` is a so-called ==worfklow==, i.e., a task that splits its workfload into multiple ==branches== that can be run in parallel - either locally or by submitting them to a batch system.
Here, `PullsAndImpacts` defines 14 branches (one for the nominal fit and one for each of the 13 nuisance parameters, dynamically read from the input workspace).
By definition, the output of a workflow is the ==collection== of all outputs of its branches, which is reflected in the status output above.

If you were to run the `PullsAndImpacts` task directly, the actual branch to run can be defined by adding `--branch N` to the command.
Passing -1 would trigger the entire workflow to be run (which is done locally by default) whereas positive numbers select a particular branch.
To get more insight into this mechanism, we can check the detailed output by passing a target depth (here 1) as a second value to `--print-status`. Note that the first line of the output now says `... and target_depth 1`.

```shell hl_lines="1"
law run PullsAndImpacts --version dev --print-status 0,1

print task status with max_depth 0 and target_depth 1

> PullsAndImpacts(branch=-1, start_branch=0, end_branch=14, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_value=1.0, mc_stats=False, workflow=local)
|   collection: SiblingFileCollection(len=14, threshold=14.0, dir=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev)
|     absent (0/14)
|     0: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__nominal.root))
|     1: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__electron_id_loose_ptlt20.root))
|     2: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__electron_id_tight.root))
|     3: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__electron_iso_loose_01.root))
|     4: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__electron_iso_loose_02.root))
|     5: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__l1_ecal_prefiring.root))
|     6: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__lumi_13TeV.root))
|     7: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__muon_id_tight.root))
|     8: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__muon_idiso_loose.root))
|     9: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__r.root))
|     10: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__top_pT_reweighting.root))
|     11: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__trigger_ee_sf.root))
|     12: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__trigger_emu_sf.root))
|     13: absent (LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__trigger_mumu_sf.root))
```

To get the pull and impact only for a particular nuisance, say `lumi_13TeV` which corresponds to branch 6, we could just run `law run PullsAndImpacts --version dev --branch 6`.

```shell hl_lines="1"
law run PullsAndImpacts --version dev --branch 6 --print-status 0

print task status with max_depth 0 and target_depth 0

> PullsAndImpacts(branch=6, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_value=1.0, mc_stats=False)
|   LocalFileTarget(path=$DHI_STORE/PullsAndImpacts/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev/fit__kl__lumi_13TeV.root)
|     absent
```

**Parameters**

- `--mass FLOAT`: The hypothetical mass of the underlying resonance. Defaults to `125.0`.
- `--hh-model MODULE`: The name of the HH model relative to `dhi.models` in the format `module:model_name`. Defaults to `hh:HHdefault`.
- `--version STRING`: Task version.
- `--datacards STRINGS`: Comma-separated paths or patterns to datacards to use. Accepts bin statements such as `emu=datacard.txt,...`. Defaults to `$DHI_EXAMPLE_DATACARDS`.
- `--poi STRING`: The name of the POI.
- `--mc-stats BOOL`: Whether to include nuisance parameters related to `autoMCStats`. Defaults to `False`.
- `--custom-args STRING`: Custom arguments to be passed to executed combine commands. Defaults to `''`.
- `--workers INT`: The number of cores to use for local processing. Defaults to `1`.
- `--print-status INT[,INT]`: When set, the first first value defines the depth of tasks whose status is printed. The second value configures the depth of nested targets and defaults to `0`. No task is actually processed.
- `--print-command INT`: When set, tasks which just execute a command via the shell print their command in a structured fashion. No task is actually processed.
- `--remove-output INT[,{a,i,d}]`: When set, task outputs are removed with a task depth corresponding to the first value. As this can be a dangerous step, the user is asked how targets should be removed, which can be either interactively (`i`), as a dry-run (`d`), or all without asking further questions ('a'). To avoid seeing this prompt, the mode can be directly set through the second value.


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


## `MergePullsAndImpacts`

The `MergePullsAndImpacts` collects the results of all branches of `PullsAndImpacts`, i.e., it requires the full workflow as can be seen in the first status output above (`PullsAndImpacts(branch=-1, ...)`).
It produces a single json file containing all pre- and postfit data in a structure that is similar to the one created by CombineHarvester's [`combineTool.py -M Impacts`](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#nuisance-parameter-impacts) command.

**Parameters**

Same as [`PullsAndImpacts`](#pullsandimpacts).


## `PlotPullsAndImpacts`

The `PlotPullsAndImpacts` task reads the json output of the `MergePullsAndImpacts` task and creates a plot where pulls and impacts are drawn into the same pad with two x-axes.
The underlying plotting function is located at [`dhi.plots.pulls_impacts_root`](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/plots/pulls_impacts_root.py) and is currently only implemented as a ROOT plot.
In general, it is also compatible the the output structure of the [`combineTool.py -M Impacts`](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/part3/nonstandard/#nuisance-parameter-impacts) command.

![Pulls and impacts](../images/pulls_impacts__kl__root.png)

**Parameters**

Same as [`PullsAndImpacts`](#pullsandimpacts) plus:

- `--parameters-per-page INT`: The number of parameters per plot page. Defaults to `-1`, meaning that all parameters are shown in the same plot.
- `--skip-parameters STRINGS`: List of parameters or files containing parameters line-by-line that should be skipped. No default.
- `--order-parameters STRINGS`: List of parameters or files containing parameters line-by-line for ordering.
- `--order-by-impact BOOL`: When True, `--parameter-order` is neglected and parameters are ordered by absolute maximum impact. Defaults to `False`.
