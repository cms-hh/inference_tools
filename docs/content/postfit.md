## Pre- and Postfit Shapes

This section will explain how you can produce prefit and postfit shapes.

Check task status:
```shell
law run PostFitShapes --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 2
```
Output:
```shell
print task status with max_depth 2 and target_depth 0

> check status of PostFitShapes(version=dev)
|  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/PostFitShapes/dev/fitDiagnostics.root)
|    absent
|
|  > check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root)
|  |    absent
|  |
|  |  > check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  |  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt)
|  |  |    absent
```


Run it in case the outputs are absent:
```shell
law run PostFitShapes --version dev1 --datacards $DHI_EXAMPLE_CARDS [...]
```
Use `--help` to see all options. Be patient this step may take a while.


## Compare all Nuisances

If you want to further use the output of the `PostFitShapes` task to compare the nuisances for prefit and post fit for the background only or signal+background fit, you can use the `CompareNuisances` task:

```shell
law run CompareNuisances --version dev --datacards $DHI_EXAMPLE_CARDS
```

You can modify the output format to:

- HTML: `--format html`
- LaTeX: `--format latex`
- Text: `--format text`
