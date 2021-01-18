# Introduction

## Cloning the repository

This repository uses submodules, so you should clone it recursively via

```shell
# ssh
git clone --recursive ssh://git@gitlab.cern.ch:7999/hh/tools/inference.git

# or

# https
git clone --recursive https://gitlab.cern.ch/hh/tools/inference.git
```


## Environment and software setup

### 1. Source the setup

The `setup.sh` script will install (and compile if necessary) all relevant software packages.

```shell
cd inference
source setup.sh
```

When the software is set up for the first time, this might take a few moments to complete.

By default, the setup script makes a few assumptions on where software is installed, outputs are stored, etc, **but this is fully configurable.**
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


### 2. Let law index your tasks and their parameters

```shell
law index --verbose
```

While indexing always sounds somewhat cumbersome, the law index file is just a human-readable file summarizing your tasks, the corresponding python modules, and their parameters.
Law uses that file only to accelerate the autocompletion of the command line interface.

You should see:

```shell
indexing tasks in 1 module(s)
loading module 'dhi.tasks', done

module 'law.contrib.git', 1 task(s):
    - law.git.BundleGitRepository

module 'dhi.tasks.base', 2 task(s):
    - BundleRepo
    - BundleSoftware

module 'dhi.tasks.limits', 6 task(s):
    - UpperLimits
    - PlotUpperLimits
    - PlotUpperLimitsAtPOI
    - PlotMultipleUpperLimitsByModel
    - MergeUpperLimits
    - PlotMultipleUpperLimits

module 'dhi.tasks.likelihoods', 3 task(s):
    - LikelihoodScan
    - PlotLikelihoodScan
    - MergeLikelihoodScan

module 'dhi.tasks.significances', 4 task(s):
    - SignificanceScan
    - PlotSignificanceScan
    - MergeSignificanceScan
    - PlotMultipleSignificanceScans

module 'dhi.tasks.pulls_impacts', 3 task(s):
    - PullsAndImpacts
    - PlotPullsAndImpacts
    - MergePullsAndImpacts

module 'dhi.tasks.test', 1 task(s):
    - test.TestPlots

module 'dhi.tasks.studies.model_selection', 3 task(s):
    - study.PlotMorphingScales
    - study.PlotMorphedDiscriminant
    - study.PlotStatErrorScan

module 'dhi.tasks.combine', 2 task(s):
    - CombineDatacards
    - CreateWorkspace

module 'dhi.tasks.postfit_shapes', 2 task(s):
    - PostFitShapes
    - PlotPostfitSOverB

module 'dhi.tasks.exclusion', 2 task(s):
    - PlotExclusionAndBestFit
    - PlotExclusionAndBestFit2D

written 29 task(s) to index file '/your/path/inference/.law/index'
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


## Documentation

The documentation is hosted at [cern.ch/cms-hh/tools/inference](https://cern.ch/cms-hh/tools/inference).

### For developers

It is built with [MkDocs](https://www.mkdocs.org) using the [material](https://squidfunk.github.io/mkdocs-material) theme and support for [PyMdown](https://facelessuser.github.io/pymdown-extensions) extensions.
Developing and building the documentation locally requires docker and a valid login at the CERN GitLab container registry.

To login, run

```shell
docker login gitlab-registry.cern.ch
```

and type your CERN username and password.
Then, to build the documentation, run

```shell
./docs/docker/run.sh build
```

which creates a directory `docs/site/` containing static HTML pages.
To start a server to browse the pages, run

```shell
./docs/docker/run.sh serve
```

and open your webbrowser at [http://localhost:8000](http://localhost:8000).
By default, all pages are *automatically rebuilt and reloaded* when a source file is updated.
