This guide highlights some of the task features and paramters used throughout the inference tools and explains some of the best practices for working with the [law](https://github.com/riga/law) package.


### Special task parameters of the inference tools

#### `--datacards` parameter

The `--datacards` parameter defines the datacards to be combined and converted into a workspace.
It accepts multiple, comma-separated paths as well as patterns and provides further conveniences.
Assuming the directory structure below,

```
$PWD/
├── ch1/
│   ├── datacard.txt
│   └── shapes.root
├── ch2/
│   ├── datacard.txt
│   └── shapes.root
└── cards3/
    ├── datacard.txt
    └── shapes.root

$DHI_DATACARDS_RUN2
└── analysis1/
    ├── datacard.txt
    └── shapes.root
```

the following examples demonstrate the datacard resolution.

1. `--datacards ch1/datacard.txt,ch2/datacard.txt,cards3/datacard.txt`: Finds all of the above datacards.
2. `--datacards ch1,ch2,cards3`: When existing directories are passed, tries to find a file called "datacard.txt". Therefore, it finds the same files as in example 1.
3. `--datacards ch?/datacard.txt`: Finds the cards located in `ch1` and `ch2`.
4. `--datacards ch?`: Same as example 3 (see 2).
5. `--datacards c*`: Matches all directies in `PWD`, so it again finds all datacards.
6. `--datacards analysis1`: When relative paths lead to matches in the current directory, the path resolution is checked again relative to the directory `$DHI_DATACARDS_RUN2` when set. Therefore in finds `$DHI_DATACARDS_RUN2/analysis1/datacard.txt`.


#### `--hh-model` parameter

The `combine` physics model, which is used during workspace creation, can be controlled with the `--hh-model` parameter.
It should have the format `module_name.model_name[@OPT][...]` where the module named `module_name` must be importable within the current Python environment and should contain a member called `model_name`, referring to the physics model instance to be used.
Certain features of the model can be disabled by passing options in the quoted format.
Available options:

- `"noNNLOscaling"`: Disables the NLO to NNLO scaling of the ggF signal.
- `"noBRscaling"`: Disables the scaling of Higgs branching ratios with model parameters.
- `"noHscaling"`: Disables the scaling of single Higgs production cross sections with model parameters.
- `"noklDependentUnc"`: Disables the kl-dependence of the ggF signal cross section.

The current default is `HHModelPinv:model_default`, referring to a model located in [dhi/models](https://gitlab.cern.ch/hh/tools/inference/-/tree/master/dhi/models).


#### `--version` parameter

Almost all inference tasks have a parameter `--version STR`.
This string is encoded deterministically in the paths of output targets which effectively introduces a simple versioning mechanism.
Changing the value will result in different targets being written and thus, allow for the creating of several sets of results in parallel (also for development and testing purposes).


#### `--mass` parameter

Most of the inference tasks have a parameter `--mass` whichd defaults to `125.0`.
This value is used in `text2workspace.py` and all `combine` commands to select the underlying mass hypothesis.
As we currently use only one hypothesis, changing the paramter has no effect but might become relevant once resonant searches are covered.


### Working with task parameters

#### Parameter definition

Task parameters are defined as *class members* on law tasks, such as:

```python
import law
import luigi

class MyTask(law.Task):

    s = luigi.Parameter()
    f = luigi.FloatParameter(default=1.)
    i = law.CSVParameter(cls=luigi.IntParameter, unique=True)
```

`luigi`, as the underlying core package behind law, already provides plenty of objects for defining task workflows.
In the above example, we use the standard string `Parameter` as well as the `FloatParameter` from luigi, whereas the `CSVParameter` is shipped with law.
The two former have a straight forward behavior for decoding from and encoding to values on the command line.
The `CSVParameter` interprets strings such as `1,2,3` as a python tuple `(1, 2, 3)` to be used in the task.
Here, we also configure the type of each particular item to be an integer with the `cls` argument, and we want each value to appear only once by using the `unique` argument.
See the [`CSVParameter` documentation](https://law.readthedocs.io/en/latest/api/parameter.html#law.parameter.CSVParameter) for more info.

These parameter definitions directly translate to the command line.
Calling

```shell
law run MyTask --s foo --f 2 --i 3,4
```

will create an instance of `MyTask`, set its parameters accordingly, and execute it.
When the task instance is created, the parameters defined on class-level are compared with the command line parameters and propagated to the instance.
Considering the initialization method

```python
def __init__(self, *args, **kwargs):
    super(MyTask, self).__init__(*args, **kwargs)

    print(self.s)
    print(self.f)
    print(self.i)
```

the *instance members* `:::python self.s`, `:::python self.f` and `:::python self.i` will, as expected, refer to actual parameter *values*, rather than `Parameter` instances defined in the block above.


#### Passing parameters upstream

Consider two tasks, called `TaskA` and `TaskB`, which are defined as

```python
import law
import luigi

class TaskA(law.Task):

    x = luigi.Parameter()
    y = luigi.Parameter(default="y_value")


class TaskB(law.Task):

    x = luigi.Parameter()
    z = luigi.Parameter()

    def requires(self):
        return TaskA.req(self)
```

`TaskB` requires `TaskA`, so that running

```shell
law run TaskB --x foo --z bar
```

will also invoke `TaskA`.
The tasks share the parameter `x` so it seems only natural to pass the *value* of `x` from `TaskB` on to `TaskA` when the requirement is defined.
==This is achieved by calling `:::python Task.req()` which is defined on all classes inheriting from `law.Task`.==
The method determines the intersection of parameters between a task class and a task instance, and creates an instance of `TaskA` with parameter values taken from the `TaskB` instance.
Thus, `:::python TaskA.req(self)` is identical to `:::python TaskA(x=self.x)` in the example above.
The intersection of common parameters is often larger, so the benefit of using `:::python Task.req()` becomes more obvious. Opposed to that, the parameteter `TaskB.z` is not *forwarded*.
Similarly, `TaskA.y` will use its default value `"y_value"`.
However, there are several ways to control this value externally, depending on the use case.

All parameters that are not explicitely set when task instances are created can be defined on the command line, so when running

```shell
law run TaskB --x foo --z bar --TaskA-y different_value
```

`TaskA.y` will be `"different_value"`.

Also, `:::python Task.req()` accepts generic keyword arguments that are forwarded to the task instantiation, so

```python
    def requires(self):
        return TaskA.req(self, y="other_value")
```

forces `TaskA.y` to be `"other_value"`.
The parameter value resolution order is "constructor" >> "command line" >> "default value" >> "config file".


### Workflows

#### Workflows in law

law provides a type of task, called *workflow*, that allows to parallelize a collection of tasks sharing the same implementation but that vary only in a single parameter.
An example is the extraction of limits, scanned over the range of another parameter as described [here](limits.md).
The way the limit is computed is always the same, independent on the (changing) value of the scan parameter.

The differences between each of these tasks is defined in the so-called *branch map* of a workflow, which is defined in `:::python create_branch_map()`.
This dictionary maps integer numbers (the *branch*), starting at zero, to an arbitrary payload (the *branch_data*) which contains the data to process, e.g. a particular value of a scan parameter.

Using the example of upper limit calculation,

```shell hl_lines="5"
law run UpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --scan-parameters kl,-10,10,21
```

extracts upper limits on POI `r` over a range of `kl` values from -10 to 10 in 21 points, i.e., one value per integer step including edges.
Executing the command as it is runs ==all== 21 *branch* tasks sequentially (or with `N` parallel processes when adding `--workers N`).
However, when you are only interested in, say, the first task at `kl=-10`, you can add the parameter `--branch 0` to the command, i.e.,

```shell hl_lines="6"
law run UpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --scan-parameters kl,-10,10,21 \
    --branch 0
```

Note that the `UpperLimits` task defines the branch map based on what is defined as `--scan-parameters`, and thus, the mapping `--branch 0` --> `kl=-10`.
An example implementation for this example could be

```python
class UpperLimits(..., law.LocalWorkflow, HTCondorWorkflow):

    # parameters defined here
    ...

    def create_branch_map(self):
        # 1D implementation
        start, stop, points = self.scan_parameters[0][1:]
        step_size = (stop - start) / (points - 1)
        return {
            b: start + b * step_size
            for b in range(points)
        }
```

Adding `--branch -1` to the command line (the default) will run all tasks.
The exact behavior can be further controlled with parameters:

- `--workflow STRING`: Defines the type of workflow, e.g., `local` or `htcondor` (see below).
- `--start-branch INT`: Defines the first branch to run.
- `--end-branch INT`: Defines the last branch to run.
- `--branches INT,...`: Defines a granular selection of branches to run. Supports range patterns in the format `START:END` (inclusive).

See the [workflow documentation](https://law.readthedocs.io/en/latest/workflows.html) for more details.


#### Remote workflows

There are several types of law workflows that wrap the task execution into batch jobs, e.g. via HTCondor.
These remote workflows first submit jobs and then enter a state of performing status queries until they succeeded.
It is safe to stop the status polling process and continue it later on by executing the exact same command as before.
Law stores job IDs in additional (optional) file targets and resumes without resubmitting jobs if not necessary, e.g. in case of failed jobs.

In the inference tools, tasks that support remote job submission inherit from `HTCondorWorkflow` declared in [dhi/tasks/base.py](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/tasks/base.py).
This feature is enabled on the wokflow task by adding `--workflow htcondor` to the command.
==However, note that the task passed to the `law run` command is not always the workflow itself==, so you have to pick the correct one.
For instance, when running the `PlotUpperLimits` task via

```shell hl_lines="6"
law run PlotUpperLimits \
    --version dev \
    --datacards $DHI_EXAMPLE_CARDS \
    --pois r \
    --scan-parameters kl,-10,10,21 \
    --UpperLimits-workflow htcondor
```

the `UpperLimits` is configured to run on HTCondor.

Remote workflows provide additional parameters to control job submission and status polling.

- `--tasks-per-job INT`: The number of tasks that each job should process. Values greater than one will reduce the number of jobs required. Defaults to `1`.
- `--retries INT`: The number of attempts per job. Defaults to `5`.
- `--poll-interval DURATION`: The interval duration between two consecutive status polls. Any valid time unit is accepted, defaulting to minutes. Defaults to `60sec`.
- `--parallel-jobs INT`: The maximum number of jobs to be run in parallel. Defaults to infinity.
- `--ignore-submission BOOL`: Ignores previously made submissions. Defaults to `False`.
- `--max-runtime DURATION`: The maximum runtime of the job. Any valid time unit is accepted, defaulting to hours. Defaults to `2h`.


### Efficient working from the command line

There are several parameters provided by law that help you work efficiently right from the command line.
A selection of them is presented in the following.
You can also check them out interactively in this [introduction notebook](https://mybinder.org/v2/gh/riga/law/master?filepath=examples%2Floremipsum%2Findex.ipynb).


#### Printing task dependencies

If you are unclear which dependencies a certain `law run` command might trigger, you can add `--print-deps N` to your command.
Instead of running any task, this will visualize the dependencies of your desired task down to a certain depth `N` (relative to the triggered task), with `0` being the triggered task itself.
Example:

```shell hl_lines="1"
> law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --print-deps 2

print task dependencies with max_depth 2

> UpperLimits(branch=-1, version=dev, datacards=hash:75bd8098ff, ...)
|
|  > CreateWorkspace(version=dev, datacards=hash:75bd8098ff, ...)
|  |
|  |  > CombineDatacards(version=dev, datacards=hash:75bd8098ff, ...)
```

A depth of `-1` will show all dependencies recursively.


#### Print the task status

Similar to the task dependencies, you can visualize the current status of tasks, i.e. the existence of their outputs by adding `--print-status N[,M]` where `N` is again the depth of dependencies (relative to the triggered task) to be shown. Example:

```shell hl_lines="1"
> law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 2

print task status with max_depth 2 and target_depth 0

> UpperLimits(branch=-1, version=dev, datacards=hash:75bd8098ff, ...)
|   collection: SiblingFileCollection(len=61, threshold=61.0, dir=...)
|     absent (0/61)
|
|  > CreateWorkspace(version=dev, datacards=hash:75bd8098ff, ...)
|  |   LocalFileTarget(path=.../workspace.root)
|  |     absent
|  |
|  |  > CombineDatacards(version=dev, datacards=hash:75bd8098ff, ...)
|  |  |   LocalFileTarget(path=.../datacard.txt)
|  |  |     absent
```

Some targets, such as file collections, support printing their status message with different verbosity levels.
This level can be configured with `M` and defaults to `0`.
Example:

```shell hl_lines="1"
> law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 0,1

print task status with max_depth 0 and target_depth 1

> UpperLimits(branch=-1, version=dev, datacards=hash:75bd8098ff, ...)
|   collection: SiblingFileCollection(len=61, threshold=61.0, dir=...)
|     absent (0/61)
|     0: absent (LocalFileTarget(path=.../*.root))
|     1: absent (LocalFileTarget(path=.../*.root))
|     2: absent (LocalFileTarget(path=.../*.root))
|     ...
|     60: absent (LocalFileTarget(path=.../*.root))
```


#### Removing outputs

As explained above, tasks whose outputs already exist are not executed again.
However, sometimes it can be useful rerun certain tasks (especially during development and testing) which requires that their outputs must be removed.
While this can be done with standard tools in your shell, a more convenient way is to add `--remove-output N[,MODE][,CONTINUE]` to the command you like to execute.
This will trigger the removal of outputs of tasks down to a certain depth `N` of dependencies.

As usual, the removal of a large number of files can be dangerous and is done interactively in law by default, i.e., you are prompted each time before a file is removed.
This behavior can be controlled with the removal `MODE` which defaults to `i` (interactive).
When you are absolutely sure about the files to be removed, use `a` (all) to remove all outputs, no questions asked.
To test the command first, use `d` (dry) to perform a dry run and see which outputs *will* be removed.

Normally, no tasks are run after one of the interactive parameters (`--print-deps`, `--print-status`, `--remove-output`, etc) was evaluated.
However, it can make sense to start the task processing after the removal of outputs.
For this purpose, you can add `y` as the third value (`CONTINUE`).
Example:

```shell hl_lines="1 10 11"
> law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --remove-output 0,a,y

remove task output with max_depth 0
selected all mode

> UpperLimits(branch=-1, version=dev, datacards=hash:75bd8098ff, ...)
|   collection: SiblingFileCollection(len=61, threshold=61.0, dir=...)
|     removed

# task processing continues here
...
```


#### Printing `combine` commands

Some of the inference tasks build shell commands involving `combine` based on the parameters they received in the command line and print them when their `run()` method is invoked.
If you are interested in the exact combine commands prior to running them, add `--print-command N` to your `law run` command.
As before, `N` is the depth of dependencies (relative to the triggered task) for which the command is printed.
For workflows, the command of the first branch (usually `--branch 0`) is printed.
Example:

```shell hl_lines="1"
> law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --print-command 0

print task commands with max_depth 0

> UpperLimits(branch=-1, version=dev, datacards=hash:75bd8098ff, ...)
|  command (from branch 0): combine -M AsymptoticLimits .../workspace.root -v 1 -m 125.0 -t -1 ...
```


#### Increasing the number of parallel (local) processes

The execution of a task is done in two steps.
First, a dependency tree is built to reflect which upstream tasks need to be run in order to provide the results that are requested when calling `law run SomeTask`.
Tasks that are already complete, i.e., tasks whose outputs already exist are identified.
Then, all incomplete tasks are run with a configurable number of worker processes.
This number can be controlled by adding `--workers N` to any of your commands.


#### Help

All law tasks allow adding a `--help` parameter which shows the full set of parameters, their descriptions and further information.
