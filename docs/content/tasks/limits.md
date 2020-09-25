Run upper limits on the cross-section with the `NLOLimit` task.

Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.
Note: Omit the `--print-status` cli option in order to run the task!

```shell
law run NLOLimit --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of NLOLimit(branch=-1, start_branch=0, end_branch=61, branches=, version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30, workflow=htcondor)
|  submission: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/kl_-30_30/htcondor_submission_0To61.json, optional)
|    absent
|  status: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/kl_-30_30/htcondor_status_0To61.json, optional)
|    absent
|  collection: SiblingFileCollection(len=61, threshold=61.0, dir=/eos/user/<u>/<username>/dhi/store/NLOLimit/dev/125/HHdefault/kl_-30_30)
|    absent (0/61)
```
By default it will calculate limits as a function of kappa lambda from -30 to 30. The parameter of interest (poi) can be changed to any of those:

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

and it's range can be adjusted within the possible ranges of the PhysicsModel.

Example:
```shell
law run NLOLimit --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --poi "C2V" --poi-range=0,5 --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of NLOLimit(branch=-1, start_branch=0, end_branch=6, branches=, version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=C2V, poi_range=0,5, workflow=htcondor)
|  submission: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/C2V_0_5/htcondor_submission_0To6.json, optional)
|    absent
|  status: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/C2V_0_5/htcondor_status_0To6.json, optional)
|    absent
|  collection: SiblingFileCollection(len=6, threshold=6.0, dir=/eos/user/<u>/<username>/dhi/store/NLOLimit/dev/125/HHdefault/C2V_0_5)
|    absent (0/6)
```

### Multiprocessing
As you can see there will be one output file in the `SiblingFileCollection` for each point within the `--poi-range`. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example usage:
```shell
law run NLOLimit --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow local --workers 4
```


### HTCondor submission
For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the CERN HTCondor cluster.

Example usage:
```shell
law run NLOLimit --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow htcondor --poll-intervall 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:
```shell
law run NLOLimit --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --workflow htcondor --poll-intervall 30sec --tasks-per-job 3
```
