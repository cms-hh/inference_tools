# HH Inference Tools

[![Documentation badge](https://img.shields.io/badge/Documentation-passing-brightgreen)](http://cms-hh.web.cern.ch/cms-hh/tools/inference/index.html) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Introduction

This repository uses submodules (currently only one), so you should clone it recursively via

```shell
# ssh
git clone --recursive ssh://git@gitlab.cern.ch:7999/hh/tools/inference.git

# or

# https
git clone --recursive https://gitlab.cern.ch/hh/tools/inference.git
```

Now proceed with setting up the software and environment.

1. Source the setup:

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


## Documentation

The documentation is hosted at [cern.ch/cms-hh/tools/inference](https://cern.ch/cms-hh/tools/inference).

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


## For developers

A consistent code style is encouraged and can be checked (and even applied) with the [black](https://github.com/psf/black) formatter.

To run the linting (i.e. show locations in the code that require formatting), run

```shell
dhi_lint
```

and to automatically fix the formatting, add `fix` to the command.


## Contributors

- Peter Fackeldey: peter.fackeldey@cern.ch (email)
- Marcel Rieger: marcel.rieger@cern.ch (email), marcel_r88 (skype)
