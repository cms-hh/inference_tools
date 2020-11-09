This section will explain how you can run one dimensional likelihood scans.

You can check the status of this task with:

```shell hl_lines="1"
law run LikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 2
print task status with max_depth 2 and target_depth 0

> LikelihoodScan1D(branch=-1, start_branch=0, end_branch=51, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_range=-25.0,25.0, poi_points=51, workflow=local)
|   collection: SiblingFileCollection(len=51, threshold=51.0, dir=$DHI_STORE/LikelihoodScan1D/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl/dev)
|     absent (0/51)
|
|  > CreateWorkspace(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |   LocalFileTarget(path=$DHI_STORE/CreateWorkspace/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/workspace.root)
|  |     existent
|  |
|  |  > CombineDatacards(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |  |   LocalFileTarget(path=$DHI_STORE/CombineDatacards/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/datacard.txt)
|  |  |     existent
```

As you can see `LikelihoodScan1D` produces by default a kappa lambda scan with a granularity of 61 points from `-30..30`.
It requires the presence of a workspace (`CreateWorkspace`), which furthermore requires the presence of a datacard (`CombineDatacards`).
The `LikelihoodScan1D` task has several cli options similary to the `UpperLimits` task. You can scan for multiple parameter of interests:

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

**Parameters**

- `--poi`: the parameter of interest you want to scan
- `--poi-range`: the range of the defined parameter of interest
- `--poi-points`: the number/granularity of scan points

(see also: [Upper limits](limits.md))


### Multiprocessing

As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --workflow local --workers 4
```


### HTCondor submission

For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:

```shell hl_lines="1"
law run LikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```


### Plotting

The `PlotLikelihoodScan1D` task takes the outputs from `LikelihoodScan1D` and `MergeLikelihoodScan1D` and plots the doubled, negative log-likehood curve over the POI parameter values in question.
There is a ROOT and a matplotlib version of the plot, which can be controlled with the `--plot-flavor` parameter.

Use `root` for the ROOT version,

```shell hl_lines="1"
law run PlotLikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --plot-flavor root
```

![1D likelihood scan with ROOT](../images/nll1d__kl_n51_-25.0_25.0__root.png)

and `mpl`for the matplotlib version,

```shell hl_lines="1"
law run PlotLikelihoodScan1D --version dev --datacards $DHI_EXAMPLE_CARDS --plot-flavor mpl
```

![1D likelihood scan with matplotlib](../images/nll1d__kl_n51_-25.0_25.0__mpl.png)

**Parameters**:

- `--plot-flavor STRING`: Either `root` or `mpl`. Defaults to `mpl` as this is the only implementation.
- `--y-log BOOL`: Logarithmic y-axis. Defaults to `False`.
