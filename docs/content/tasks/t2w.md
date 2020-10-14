After combining your datacards you need to create a workspace using a PhysicsModel. We will use different HH PhysicsModel to perform next-to-leading order fits.

This task can be run with:

```shell
law run CreateWorkspace --version dev
```

Now since we require a combined datacard, we will see the power of `law` workflows. This task does indeed require the output of the `CombineDatacards` task.

Let's have a look at a higher task depth:
```shell
law run CreateWorkspace --version dev --print-status 1
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
As you can see there is a task hierarchy and in fact `CreateWorkspace` will not be exectued until the output of `CombineDatacards` exists. Another thing to notice is that cli options (here: `--datacards`) are upstreamed to all required tasks.

Here we can use a new cli option: `--hh-model`, which default is the standard HH PhysicsModel. In case you want to use a new PhysicsModel, you just have to add it to `dhi/models` and pass it to `--hh-model` (relative to `dhi/models`), such as:

(Assume we created a PhysicsModel called `MyCoolPhysicsModel` in `dhi/models/my_model.py`)
```shell
law run CreateWorkspace --version dev --hh-model my_model:MyCoolPhysicsModel --print-status 0
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
