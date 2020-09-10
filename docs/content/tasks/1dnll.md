This section will explain how you can run one dimensional likelihood scans.

Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.
Note: Omit the `--print-status` cli option in order to run the task!

You can check the status of this task with:

```shell
law run dhi.NLOScan1D --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --print-status 2
```
Output:
```shell
print task status with max_depth 2 and target_depth 0

> check status of dhi.NLOScan1D(branch=-1, start_branch=0, end_branch=200, branches=, version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30, points=200, workflow=htcondor)
|  submission: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200/htcondor_submission_0To200.json, optional)
|    absent
|  status: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200/htcondor_status_0To200.json, optional)
|    absent
|  collection: SiblingFileCollection(len=200, threshold=200.0, dir=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200)
|    absent (0/200)
|
|  > check status of dhi.NLOT2W(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOT2W/dev/125/HHdefault/workspace_HHdefault.root)
|  |    absent
|  |
|  |  > check status of dhi.CombDatacards(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  |  datacard: LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/CombDatacards/dev/125/HHdefault/datacard.txt)
|  |  |    absent
```

As you can see `dhi.NLOScan1D` produces by default a kappa lambda scan with a granularity of 200 points. It requires the presence of a workspace (`dhi.NLOT2W`), which furthermore requires the presence of a datacard (`dhi.CombDatacards`). The `dhi.NLOScan1D` task has several cli options similary to the `dhi.NLOLimit` task. You can scan for multiple parameter of interests:

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

and it's range can be adjusted within the possible ranges of the PhysicsModel (see: [Upper limits](limits.md) for further details).

The granularity of these scans can be adjusted by passing a different number of points to the `--points` cli option.


### Multiprocessing
As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:
```shell
law run dhi.NLOScan1D --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow local --workers 4
```


### HTCondor submission
For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the CERN HTCondor cluster.

Example usage:
```shell
law run dhi.NLOScan1D --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow htcondor --poll-intervall 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:
```shell
law run dhi.NLOScan1D --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow htcondor --poll-intervall 30sec --tasks-per-job 3
```
