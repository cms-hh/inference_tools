# General Options

## Print task status

Use `--print-status N` (N: task depth) to show the output status of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --print-status 1
```

Output:
```shell
print task status with max_depth 1 and target_depth 0

> check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root)
|    absent
|
|  > check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  - LocalFileTarget(path=/afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt)
|  |    absent
```

## Print commands

Use `--print-command N` (N: task depth) to show the underlying bash command of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --print-command 1
```

Output:
```shell
print task commands with max_depth 1

> CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  command: text2workspace.py /afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt -o /afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root -m 125.0 -P dhi.models.hh:HHdefault
|
|  > CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  command: combineCards.py /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt /afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt /afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt > /afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt
```

In case there is no underlying bash command, it will print `not a CommandTask`.


## Print task dependencies (no outputs)

Use `--print-deps N` (N: task depth) to show the dependency tree of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --print-deps 1
```

Output:
```shell
print task dependencies with max_depth 1

> CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|
|  > CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
```

## Print outputs only

Use `--print-output N` (N: task depth) to show the outputs of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --print-output 1
```

Output:
```shell
print task output with max_depth 1

file:///afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root
file:///afs/cern.ch/user/m/mfackeld/repos/inference/data/store/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt
```


## Fetch output to current directory

Use `--remove-output N` (N: task depth) to remove the outputs of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --fetch-output 1
```

Output (prompt):
```shell
fetch task output with max_depth 1
target directory is /afs/cern.ch/user/m/mfackeld/repos/inference
fetch mode? [i*(interactive), a(all), d(dry)]
```

Choose from three different modes: `interactive`, `dry`, `all`. Recommendation: `interactive`.


## Remove outputs safe and recursively

Use `--remove-output N` (N: task depth) to remove the outputs of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --version dev --remove-output 1
```

Output (prompt):
```shell
remove task output with max_depth 1
removal mode? [i*(interactive), d(dry), a(all)]
```

Choose from three different modes: `interactive`, `dry`, `all`. Recommendation: `interactive`.


## Version

Use `--version XY` to handle multiple inference calculations in parallel safely. Useful for testing or to safely switch between older and newer run versions.


## Mass

Use `--mass XY` (default: `125.0`) to pass a different mass to the combine calls, currently has no impact.


## HHModel

Use `--hh-model` to create a workspace with a different PhysicsModel. PhysicsModel needs exist relative to `dhi/models`. See more in [Text to workspace](tasks/t2w.md).


## Datacards

Use `--datacards` to define the input datacards, will be combined in the first step (supports globbing). See more in [Combine Datacards](tasks/combinedatacards.md).
