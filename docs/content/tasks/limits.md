Run upper limits on the cross-section with the `UpperLimits` task.

```shell
law run UpperLimits --version dev --print-status 2
```
Output:
```shell
print task status with max_depth 2 and target_depth 0

> check status of UpperLimits(branch=-1, start_branch=0, end_branch=61, branches=, version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault, poi=kl, poi_range=-30.0,30.0, poi_points=61, workflow=local)
|  collection: SiblingFileCollection(len=61, threshold=61.0, dir=$DHI_STORE/UpperLimits/m125.0/model_hh_HHdefault/kl/dev)
|    absent (0/61)
|
|  > check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  - LocalFileTarget(path=$DHI_STORE/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root)
|  |    absent
|  |
|  |  > check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  |  - LocalFileTarget(path=$DHI_STORE/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt)
|  |  |    absent
```
By default it will calculate limits as a function of kappa lambda from -30 to 30 with a step size of 1.
The parameter of interest (poi) can be changed to any of those:

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

Cli parameters:

- `--poi`: the parameter of interest you want to scan
- `--poi-range`: the range of the defined parameter of interest
- `--poi-points`: the number/granularity of scan points

Example:
```shell
law run UpperLimits --version dev --poi "C2V" --poi-range=-5,5 --poi-points 20 --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of UpperLimits(branch=-1, start_branch=0, end_branch=20, branches=, version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault, poi=C2V, poi_range=-5.0,5.0, poi_points=20, workflow=local)
|  collection: SiblingFileCollection(len=20, threshold=20.0, dir=$DHI_STORE/UpperLimits/m125.0/model_hh_HHdefault/C2V/dev)
|    absent (0/20)
```

### Multiprocessing
As you can see there will be one output file in the `SiblingFileCollection` for each point. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:
```shell
law run UpperLimits --version dev --workflow local --workers 4
```


### HTCondor submission
For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example usage:
```shell
law run UpperLimits --version dev --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:
```shell
law run UpperLimits --version dev --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```

---
**_NOTES_**

Be cautious with the `--poi-range` option. Defining a range which is outside of the one defined by the PhysicsModel might lead to unreasonable results.
