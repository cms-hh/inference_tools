# Introduction

## First steps (setup)

This repository uses submodules (currently only one), so you should clone it recursively via

```shell
git clone --recursive https://gitlab.cern.ch/hh/tools/inference.git
cd inference
```

Now proceed with setting up the software and environment.

1. Source the setup:

The `setup.sh` script will install (and compile if necessary) all relevant software packages.

```shell
source setup.sh
```

When the software is set up for the first time, this might take a few moments to complete.

==By default, the setup script makes a few assumptions on where software is installed, outputs are stored, etc, but this is fully configurable.==
To do so, run

```shell
source setup.sh some_name
```

where the value of `some_name` is your choice, and the script interactively guides you through the quick setup process.
To use the same configuration the next time, ==make sure to use the same value you passed before.==

**Note**: In case you want to reinstall the software stack or combine from scratch, prepend `DHI_REINSTALL_SOFTWARE=1` or `DHI_REINSTALL_COMBINE=1` to the `source` commend, e.g.

```shell
DHI_REINSTALL_SOFTWARE=1 source setup.sh some_name
```


2. Let law index your tasks and their parameters:

```shell
law index --verbose
```

While indexing always sounds somewhat cumbersome, the law index file is just a human-readable file summarizing your tasks, the corresponding python modules, and their parameters.
Law uses that file only to accelerate the autocompletion of the command line interface.

You should see:

```shell
indexing tasks in 5 module(s)
loading module 'dhi.tasks.base', done
loading module 'dhi.tasks.nlo.base', done
loading module 'dhi.tasks.nlo.inference', done
loading module 'dhi.tasks.nlo.plotting', done
loading module 'dhi.tasks.misc', done

module 'law.contrib.git', 1 task(s):
    - law.git.BundleGitRepository

module 'dhi.tasks.base', 2 task(s):
    - BundleRepo
    - BundleSoftware

module 'dhi.tasks.nlo.inference', 10 task(s):
    - UpperLimits
    - LikelihoodScan1D
    - LikelihoodScan2D
    - PullsAndImpacts
    - CombineDatacards
    - CreateWorkspace
    - MergePullsAndImpacts
    - MergeLikelihoodScan1D
    - MergeLikelihoodScan2D
    - MergeUpperLimits

module 'dhi.tasks.nlo.plotting', 6 task(s):
    - PlotUpperLimits
    - PlotUpperLimitsAtPOI
    - PlotLikelihoodScan1D
    - PlotLikelihoodScan2D
    - PlotPullsAndImpacts
    - PlotMultipleUpperLimits

module 'dhi.tasks.misc', 2 task(s):
    - PostFitShapes
    - CompareNuisances

written 21 task(s) to index file '/your/path/inference/.law/index'
```

You can type

```shell
law run <tab><tab>
```

to see the list of available tasks in your shell, and

```shell
law run SomeTask <tab><tab>
```

to list all parameters of `SomeTask`.

Now you are done with the setup and can start running the statistical inference!
