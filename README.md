# HH Inference Tools

[![Documentation badge](https://img.shields.io/badge/Documentation-passing-brightgreen)](http://cms-hh.web.cern.ch/cms-hh/tools/inference/index.html) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


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
Internally, a file `.setups/some_name.sh` is created which contains export statements line by lines that you can be update anytime.


#### Integrating the `datacards_run2` repository

Datacards for the combination are stored in the (protected) [`datacards_run2` repository](https://gitlab.cern.ch/hh/results/datacards_run2).
Some of the inference tools (especially the automated workflows) can directly inspect and use these datacards without much configuration.
You only have to set the environment variable `DHI_DATACARDS_RUN2` to the location of your local checkout.
If you decided to use a configurable setup (see above), you will be asked for the value of this variable interactively.
For instance, you could clone the repository first to a location of your choice,

```shell
git clone --recursive ssh://git@gitlab.cern.ch:7999/hh/results/datacards_run2.git /your/path
```

(note that using `ssh://...` is recommended), and then use `/your/path` for the `DHI_DATACARDS_RUN2` variable later on.


#### Reinstalling software

In case you want to reinstall the software stack or combine from scratch, prepend `DHI_REINSTALL_SOFTWARE=1` or `DHI_REINSTALL_COMBINE=1` to the `source` commend, e.g.

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

module 'dhi.tasks.combine', 3 task(s):
    - InputDatacards
    - CombineDatacards
    - CreateWorkspace

module 'dhi.tasks.base', 2 task(s):
    - BundleRepo
    - BundleSoftware

module 'dhi.tasks.limits', 6 task(s):
    - UpperLimits
    - PlotUpperLimits
    - PlotUpperLimitsAtPoint
    - PlotMultipleUpperLimitsByModel
    - MergeUpperLimits
    - PlotMultipleUpperLimits

module 'dhi.tasks.likelihoods', 5 task(s):
    - LikelihoodScan
    - PlotLikelihoodScan
    - PlotMultipleLikelihoodScansByModel
    - MergeLikelihoodScan
    - PlotMultipleLikelihoodScans

module 'dhi.tasks.significances', 4 task(s):
    - SignificanceScan
    - PlotSignificanceScan
    - MergeSignificanceScan
    - PlotMultipleSignificanceScans

module 'dhi.tasks.pulls_impacts', 3 task(s):
    - PullsAndImpacts
    - PlotPullsAndImpacts
    - MergePullsAndImpacts

module 'dhi.tasks.postfit_shapes', 2 task(s):
    - PostfitShapes
    - PlotPostfitSOverB

module 'dhi.tasks.gof', 3 task(s):
    - GoodnessOfFit
    - PlotGoodnessOfFit
    - PlotMultipleGoodnessOfFits
    - MergeGoodnessOfFit

module 'dhi.tasks.test', 1 task(s):
    - test.TestPlots

module 'dhi.tasks.studies.model_selection', 3 task(s):
    - study.PlotMorphingScales
    - study.PlotMorphedDiscriminant
    - study.PlotStatErrorScan

module 'dhi.tasks.exclusion', 2 task(s):
    - PlotExclusionAndBestFit
    - PlotExclusionAndBestFit2D

written 36 task(s) to index file '/your/path/inference/.law/index'
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


## Developers

- Peter Fackeldey: peter.fackeldey@cern.ch (email)
- Marcel Rieger: marcel.rieger@cern.ch (email), marcel_r88 (skype)
