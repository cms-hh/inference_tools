This section will explain how to produce plots from finalized fits and scans.

Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.
Note: Omit the `--print-status` cli option in order to run the task!

## Plot: Upper Limits on the Cross Section

Check the task status with:
```shell
law run dhi.PlotScan --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --print-status 1
```
Output:
```shell
print task status with max_depth 1 and target_depth 0

> check status of dhi.PlotScan(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30)
|  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/PlotScan/dev/125/HHdefault/kl_-30_30/scan.pdf)
|    absent
|
|  > check status of dhi.NLOLimit(branch=-1, start_branch=0, end_branch=61, branches=, version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30, workflow=htcondor)
|  |  submission: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/kl_-30_30/htcondor_submission_0To61.json, optional)
|  |    absent
|  |  status: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOLimit/dev/125/HHdefault/kl_-30_30/htcondor_status_0To61.json, optional)
|  |    absent
|  |  collection: SiblingFileCollection(len=61, threshold=61.0, dir=/eos/user/<u>/<username>/dhi/store/NLOLimit/dev/125/HHdefault/kl_-30_30)
|  |    absent (0/61)
```

As you can see the `dhi.PlotScan` task requires the presence of calculated limits by the `dhi.NLOLimit` task. Again all cli options are upstreamed to the required tasks.
Note: If you don't have a valid datacard yet, there is a default datacard, which is used if you don't modifiy/use the `--input-cards` cli option. You can produce a limit plot then with:
```shell
law run dhi.PlotScan --version dev --dhi.NLOLimit-workflow local --workers 8 --poi-range=-10,10
```
For simplicity the range of the parameter of interest is reduced to `-10..10`. Additionally these datacards are very simple, which means it is sufficent to use local multiprocessing with 8 workers (no HTCondor submission). We forward this option with to the `dhi.NLOLimit` task: `--dhi.NLOLimit-workflow local --workers 8`.

The resulting plot can be found in: `/eos/user/<u>/<username>/dhi/store/PlotScan/dev/125/HHdefault/kl_-10_10/scan.pdf` or more conveniently in the webbrowser: https://cernbox.cern.ch/ (path: `dhi/store/PlotScan/dev/125/HHdefault/kl_-10_10/scan.pdf`), which supports a direct PDF preview.


## Plot: One Dimensional Likelihood Scans

Check the task status with:
```shell
law run dhi.PlotNLL1D --version dev --print-status 2
```
Output:
```shell
print task status with max_depth 2 and target_depth 0

> check status of dhi.PlotNLL1D(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30)
|  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/PlotNLL1D/dev/125/HHdefault/kl_-30_30/nll.pdf)
|    absent
|
|  > check status of dhi.MergeScans1D(branch=-1, start_branch=-1, end_branch=-1, branches=, cancel_jobs=False, cleanup_jobs=False, version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30, points=200)
|  |  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/MergeScans1D/dev/125/HHdefault/kl_-30_30/200/scan1d_merged.npz)
|  |    absent
|  |
|  |  > check status of dhi.NLOScan1D(branch=-1, start_branch=0, end_branch=200, branches=, version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-30,30, points=200, workflow=htcondor)
|  |  |  submission: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200/htcondor_submission_0To200.json, optional)
|  |  |    absent
|  |  |  status: LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200/htcondor_status_0To200.json, optional)
|  |  |    absent
|  |  |  collection: SiblingFileCollection(len=200, threshold=200.0, dir=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOScan1D/dev/125/HHdefault/kl_-30_30/200)
|  |  |    absent (0/200)
```

As you can see the `dhi.PlotNLL1D` task requires the presence of calculated limits by the `dhi.NLOScan1D` task and their merged outputs by the `dhi.MergeScans1D` task. Again all cli options are upstreamed to the required tasks.
Note: If you don't have a valid datacard yet, there is a default datacard, which is used if you don't modifiy/use the `--input-cards` cli option. You can produce a 1D scan plot then with:
```shell
law run dhi.PlotNLL1D --version dev --dhi.MergeScans1D-workflow local --workers 8 --dhi.MergeScans1D-points 50
```
For simplicity the granularity of scan points of the parameter of interest is reduced to `50`. Additionally these datacards are very simple, which means it is sufficent to use local multiprocessing with 8 workers (no HTCondor submission). We forward this option with to the `dhi.MergeScans1D` task: `--dhi.MergeScans1D-workflow local --workers 8`.

The resulting plot can be found in: `/eos/user/<u>/<username>/dhi/store/PlotNLL1D/dev/125/HHdefault/kl_-30_30/nll.pdf` or more conveniently in the webbrowser: https://cernbox.cern.ch/ (path: `dhi/store/PlotNLL1D/dev/125/HHdefault/kl_-30_30/nll.pdf`), which supports a direct PDF preview.


## Plot: Two Dimensional Likelihood Scans

Completely equivalent usage to one dimensional likelihood scan plots. The corresponding task is called: `law run dhi.PlotNLL2D`.
