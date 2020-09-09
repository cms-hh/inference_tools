Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.

After combining your datacards you need to create a workspace using a PhysicsModel. We will use Luca Cadamuro's HH PhysicsModel to perform next-to-leading order fits: https://indico.cern.ch/event/885273/contributions/3812533/attachments/2016615/3370728/HH_combine_model_7Apr2018.pdf

This task can be run with:

```shell
law run dhi.NLOT2W --version dev
```

Now since we require a combined datacard, we will see the power of `law` workflows. This task does indeed require the output of the `dhi.CombDatacards` task.

Let's have a look at a higher task depth:
```shell
law run dhi.NLOT2W --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --print-status 1
```
Output:
```shell
print task status with max_depth 1 and target_depth 0

> check status of dhi.NLOT2W(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOT2W/dev/125/HHdefault/workspace_HHdefault.root)
|    absent
|
|  > check status of dhi.CombDatacards(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  datacard: LocalFileTarget(path=/eos/user/<u>/<username>/dhi/store/CombDatacards/dev/125/HHdefault/datacard.txt)
|  |    absent
```
As you can see there is a task hierarchy and in fact `dhi.NLOT2W` will not be exectued until the output of `dhi.CombDatacards` exists. Another thing to notice is that cli options (here: `--input-cards`) are upstreamed to all required tasks.

Here we can use a new cli option: `--hh-model`, which default is the aforementioned PhysicsModel by Luca Cadamuro. In case you want to use a new PhysicsModel, you just have to add it to `dhi.utils.models.py` and pass it to `--hh-model`, such as:

```shell
law run dhi.NLOT2W --version dev --input-cards "/path/to/first/card.txt,/path/to/second/card.txt" --hh-model MyCoolPhysicsModel --print-status 0
```
Output:
```shell
print task status with max_depth 0 and target_depth 0

> check status of dhi.NLOT2W(version=dev, mass=125, input_cards=, dc_prefix=, hh_model=MyCoolPhysicsModel, stack_cards=False)
|  - LocalFileTarget(path=/afs/cern.ch/work/<u>/<username>/dhi_store/NLOT2W/dev/125/MyCoolPhysicsModel/workspace_MyCoolPhysicsModel.root)
|    absent
```
It will automatically look for `MyCoolPhysicsModel` in `dhi.utils.models.py` and will use this to create the workspace.


### Todo

* Add PhysicsModel with single Higgs extension
