## Pre- and Postfit Shapes

This section will explain how you can produce prefit and postfit shapes.

Check task status:
```shell
law run dhi.PostFitShapes --version dev1 --print-status 1
```
Output:
```shell
print task status with max_depth 1 and target_depth 0

> check status of dhi.PostFitShapes(version=dnn_score_max, mass=125)
|  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/PostFitShapes/dnn_score_max/125/fitDiagnostics.root)
|    existent
|
|  > check status of dhi.NLOT2W(version=dnn_score_max, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOT2W/dnn_score_max/125/HHdefault/workspace_HHdefault.root)
|  |    existent
```


Run it in case the outputs are absent:
```shell
law run dhi.PostFitShapes --version dev1 [...]
```
Use `--help` to see all options. Be patient this step may take a while.


## Compare all Nuisances

If you want to further use the output of the `dhi.PostFitShapes` task to compare the nuisances for prefit and post fit for the background only or signal+background fit, you can use the `dhi.CompareNuisances` task:

```shell
law run dhi.CompareNuisances --version dev1
```

You can modify the output format to:

- HTML: `--format html`
- LaTeX: `--format latex`
- Text: `--format text`
