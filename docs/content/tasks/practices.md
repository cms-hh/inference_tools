# General Options

## Print task status

Use `--print-status N` (N: task depth) to show the output status of the current task up to a certain requirement depth:

```shell
law run CreateWorkspace --datacards $DHI_EXAMPLE_CARDS --version dev --print-status 1
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
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --print-command 1
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
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --print-deps 1
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
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --print-output 1
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
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --fetch-output 1
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
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --remove-output 1
```

Output (prompt):
```shell
remove task output with max_depth 1
removal mode? [i*(interactive), d(dry), a(all)]
```

Choose from three different modes: `interactive`, `dry`, `all`. Recommendation: `interactive`.


## Datacards

Use `--datacards` to define the input datacards, will be combined in the first step (supports globbing).
This parameter has **no default value**.
See more info in [Combine Datacards](tasks/combinedatacards.md).


## Version

Use `--version XY` to handle multiple inference calculations in parallel safely. Useful for testing or to safely switch between older and newer run versions.


## Mass

Use `--mass XY` (default: `125.0`) to pass a different mass to the combine calls, currently has no impact.


## HHModel

Use `--hh-model` to create a workspace with a different PhysicsModel.
PhysicsModel needs exist relative to `dhi/models`. See more in [Text to workspace](tasks/t2w.md).




If you are starting with multiple datacards you can use the `CombineDatacards` task to combine them.
You can run this task with:

Let combine automatically choose bin names:

```shell
law run CombineDatacards --version dev --datacards "/path/to/first/card.txt,/path/to/second/card.txt"
```
or use your own bin names:

```shell
law run CombineDatacards --version dev --datacards "first=/path/to/first/card.txt,second=/path/to/second/card.txt"
```

You can pass multiple comma-seperated datacard paths to the `--datacards` cli option.
It also supports globbing, such as:

```shell
law run CombineDatacards --version dev --datacards "/path/to/some/cards/but_only_these*.txt"
```

Relative datacards paths are resolved to your current directory, or, when no file was found, relative to the `datacards_run2` directory in the top level of the repository.
Also, when the path you passed matches a directory, it automatically tries to find a file "datacard.txt" in that directory.
This is meant to simplify the task configuration on the command line as it allows to pass `--datacards bbgg/v0`, provided that the file `repo_path/datacards_run2/bbgg/v0/datacard.txt` exists.

In case you want to give your combined datacard a certain prefix you can use the `--dc-prefix` cli option:

```shell
law run CombineDatacards --version dev --datacards "/path/to/some/cards/but_only_these*.txt" --dc-prefix "my_" --print-status 0
```

Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=my_, hh_model=hh:HHdefault)
|  - LocalFileTarget(path=$DHI_STORE/CombineDatacards/m125.0/model_hh_HHdefault/dev/my_datacard.txt)
|    absent
```

---

**_NOTES_**

As many datacards can make the Task representation unreadable, the input datacard names are hashed and used for the `__repr__`.
In case you pass only 1 datacard to the `--datacards` cli option, this datacard will just be forwarded and nothing happens.


After combining your datacards you need to create a workspace using a PhysicsModel. We will use different HH PhysicsModel to perform next-to-leading order fits.

This task can be run with:

```shell
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS
```

Now since we require a combined datacard, we will see the power of `law` workflows.
This task does indeed require the output of the `CombineDatacards` task.

Let's have a look at a higher task depth:
```shell
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 1
```
Output:
```shell
print task status with max_depth 1 and target_depth 0

> check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  - LocalFileTarget(path=$DHI_STORE/CreateWorkspace/m125.0/model_hh_HHdefault/dev/workspace.root)
|    absent
|
|  > check status of CombineDatacards(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=hh:HHdefault)
|  |  - LocalFileTarget(path=$DHI_STORE/CombineDatacards/m125.0/model_hh_HHdefault/dev/datacard.txt)
|  |    absent
```

As you can see there is a task hierarchy and in fact `CreateWorkspace` will not be exectued until the output of `CombineDatacards` exists.
Another thing to notice is that cli options (here: `--datacards`) are upstreamed to all required tasks.

Here we can use a new cli option: `--hh-model`, which default is the standard HH PhysicsModel.
In case you want to use a new PhysicsModel, you just have to add it to `dhi/models` and pass it to `--hh-model` (relative to `dhi/models`), such as:

(Assume we created a PhysicsModel called `MyCoolPhysicsModel` in `dhi/models/my_model.py`)
```shell
law run CreateWorkspace --version dev --datacards $DHI_EXAMPLE_CARDS --hh-model my_model:MyCoolPhysicsModel --print-status 0
```

Output:

```shell
print task status with max_depth 0 and target_depth 0

> check status of CreateWorkspace(version=dev, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=my_model:MyCoolPhysicsModel)
|  - LocalFileTarget(path=$DHI_STORE/CreateWorkspace/m125.0/model_my_model_MyCoolPhysicsModel/dev/workspace.root)
|    absent
```

It will automatically look for `MyCoolPhysicsModel` in `dhi/models/my_model.py` and will use this to create the workspace.

---
**_NOTES_**

Currently the default PhysicsModel is the one presented [here](https://indico.cern.ch/event/885273/contributions/3812533/attachments/2016615/3370728/HH_combine_model_7Apr2018.pdf).
