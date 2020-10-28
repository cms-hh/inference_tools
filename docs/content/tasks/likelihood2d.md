This section will explain how you can run two dimensional likelihood scans.

You can check the status of this task with:

```shell hl_lines="1"
law run LikelihoodScan2D --version dev --print-status 2
print task status with max_depth 2 and target_depth 0

> LikelihoodScan2D(branch=-1, start_branch=0, end_branch=1071, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi1=kl, poi2=kt, poi1_range=-25.0,25.0, poi2_range=-10.0,10.0, poi1_points=51, poi2_points=21, workflow=local)
|   collection: SiblingFileCollection(len=1071, threshold=1071.0, dir=$DHI_STORE/LikelihoodScan2D/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/kl__kt/dev)
|     absent (0/1071)
|
|  > CreateWorkspace(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |   LocalFileTarget(path=$DHI_STORE/CreateWorkspace/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/workspace.root)
|  |     existent
|  |
|  |  > CombineDatacards(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |  |   LocalFileTarget(path=$DHI_STORE/CombineDatacards/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/datacard.txt)
|  |  |     existent
```

As you can see `LikelihoodScan2D` produces by default a kappa lambda vs kappa top scan with a granularity of 1281 points (kl: `-30..30`, kt: `-10..10`, stepsize: 1).
It requires the presence of a workspace (`CreateWorkspace`), which furthermore requires the presence of a datacard (`CombineDatacards`).
The `LikelihoodScan2D` task has several cli options similary to the `UpperLimits` task.
You can scan for multiple combinations of two parameter of interests (but not the same ones):

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

**Parameters**

- `--poi1`: the parameter of interest you want to scan
- `--poi1-range`: the range of the defined parameter of interest
- `--poi1-points`: the number/granularity of scan points
- `--poi2`: the parameter of interest you want to scan
- `--poi2-range`: the range of the defined parameter of interest
- `--poi2-points`: the number/granularity of scan points


### Multiprocessing

As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan2D --version dev --workflow local --workers 4
```


### HTCondor submission

For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example usage:

```shell hl_lines="1"
law run LikelihoodScan2D --version dev --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:

```shell hl_lines="1"
law run LikelihoodScan2D --version dev --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```


### Plotting

The `PlotLikelihoodScan2D` task takes the outputs from `LikelihoodScan2D` and `MergeLikelihoodScan2D` and plots the doubled, negative log-likehood curve over the two POI parameter values in question.
There is a ROOT and a matplotlib version of the plot, which can be controlled with the `--plot-flavor` parameter.

Use `root` for the ROOT version,

```shell hl_lines="1"
law run PlotLikelihoodScan2D --version dev --poi1-points 61 --poi2-points 41 --plot-flavor root
```

![2D likelihood scan with ROOT](../images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log__root.png)

and `mpl`for the matplotlib version,

```shell hl_lines="1"
law run PlotLikelihoodScan2D --version dev --poi1-points 61 --poi2-points 41 --plot-flavor mpl
```

![2D likelihood scan with matplotlib](../images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log__mpl.png)

**Parameters**:

- `--plot-flavor STRING`: Either `root` or `mpl`. Defaults to `mpl` as this is the only implementation.
- `--y-log BOOL`: Logarithmic y-axis. Defaults to `False`.
