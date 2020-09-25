This section will explain how you can produce pulls and impacts plots.

Check task status:
```shell
law run PlotImpacts --version dev --print-status 3
```
Output:
```shell
print task status with max_depth 3 and target_depth 0

> check status of PlotImpacts(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/PlotImpacts/dev/125/HHdefault/impacts.pdf)
|    absent
|
|  > check status of ImpactsPulls(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, r=1.0, r_range=0,30)
|  |  - LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/ImpactsPulls/dev/125/HHdefault/impacts.json)
|  |    absent
|  |
|  |  > check status of NLOT2W(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  |  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOT2W/dev/125/HHdefault/workspace_HHdefault.root)
|  |  |    absent
|  |  |
|  |  |  > check status of CombDatacards(version=dev, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt,/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  |  |  datacard: LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/CombDatacards/dev/125/HHdefault/datacard.txt)
|  |  |  |    absent
```

As you can see the `ImpactsPulls` calculates pulls and impacts from the workspace, which are afterwards then plotted by the `PlotImpacts` task.
Additionally you can modify the signal injection (signal strength) by passing `--ImpactsPulls-r 5` (default: `1`) and its range with `--ImpactsPulls-r-range` (default: `0..30`) to the `law run PlotImpacts --version dev ...` command.

Note: If you don't have a valid datacard yet, there is a default datacard, which is used if you don't modifiy/use the `--input-cards` cli option. You can produce a pulls and impacts plot then with:
```shell
law run PlotImpacts --version dev
```
Currently the calculation of pulls and impacts uses only CombineHarvester internal multiprocessing. The task automatically detects the number of cores on the current machine and uses this number for multiprocessing. A HTCondorWorkflow is currently work in progress for this task.

The resulting plot can be found in: `/eos/user/<u>/<username>/dhi/store/PlotImpacts/dev/125/HHdefault/impacts.pdf` or more conveniently in the webbrowser: https://cernbox.cern.ch/ (path: `dhi/store/PlotImpacts/dev/125/HHdefault/impacts.pdf`), which supports a direct PDF preview.
