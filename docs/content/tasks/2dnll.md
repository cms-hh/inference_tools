# Two Dimensional Likelihood Scans

This section will explain how you can run two dimensional likelihood scans.

You can check the status of this task with:

```shell
law run LikelihoodScan2D --version dev --print-status 2
```
Output:
```shell
print task status with max_depth 2 and target_depth 0

> check status of LikelihoodScan2D(branch=-1, start_branch=0, end_branch=1281, branches=, version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault, poi1=kl, poi2=kt, poi1_range=-30.0,30.0, poi2_range=-10.0,10.0, poi1_points=61, poi2_points=21, workflow=local)
|  collection: SiblingFileCollection(len=1281, threshold=1281.0, dir=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/LikelihoodScan2D/m125.0/model_hh_HHdefault/kl__kt/dev)
|    absent (0/1281)
|
|  > check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root)
|  |    absent
|  |
|  |  > check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  |  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt)
|  |  |    absent
```

As you can see `LikelihoodScan2D` produces by default a kappa lambda vs kappa top scan with a granularity of 1281 points (kl: `-30..30`, kt: `-10..10`, stepsize: 1). It requires the presence of a workspace (`CreateWorkspace`), which furthermore requires the presence of a datacard (`CombineDatacards`). The `LikelihoodScan2D` task has several cli options similary to the `UpperLimits` task. You can scan for multiple combinations of two parameter of interests (but not the same ones):

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

Cli parameters:

First poi:

- `--poi1`: the parameter of interest you want to scan
- `--poi1-range`: the range of the defined parameter of interest
- `--poi1-points`: the number/granularity of scan points

Second poi:

- `--poi2`: the parameter of interest you want to scan
- `--poi2-range`: the range of the defined parameter of interest
- `--poi2-points`: the number/granularity of scan points

(see also: [Upper limits](limits.md))


### Multiprocessing
As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:
```shell
law run LikelihoodScan2D --version dev --workflow local --workers 4
```


### HTCondor submission
For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example usage:
```shell
law run LikelihoodScan2D --version dev --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:
```shell
law run LikelihoodScan2D --version dev --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```

---
**_NOTES_**

Be cautious with the `--poi1-range` and `--poi2-range` options. Defining a range which is outside of the one defined by the PhysicsModel might lead to unreasonable results.
