# Introduction

## First steps (setup)

Connect to a CentOS 7 node on lxplus:
```shell
ssh <cern_username>@lxplus.cern.ch
```

Clone this repository:
```shell
git clone https://gitlab.cern.ch/hh/tools/inference.git
cd inference
```

Now proceed with setting up the software and environment.
The `setup.sh` script will install (and compile if necessary) all relevant software packages:

- https://github.com/cms-sw/cmssw/tree/CMSSW_10_2_X (CMSSW_10_2_20_UL)
- https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git
- https://github.com/riga/CombineHarvester.git
- https://github.com/riga/law/tree/master/law
- https://github.com/psf/black (only for developers)

Run the following two steps everytime you newly login to a CentOS 7 node on lxplus in order to setup the correct software.

1. Be patient this may take some time if you set up the software for the first time:
```shell
source setup.sh
```

2. Let law index your tasks and their parameters (for autocompletion)
```shell
law index --verbose
```

You should see:
```bash
indexing tasks in 4 module(s)
loading module 'dhi.tasks.base', done
loading module 'dhi.tasks.nlo.inference', done
loading module 'dhi.tasks.nlo.plotting', done
loading module 'dhi.tasks.nlo.compare', done

module 'dhi.tasks.nlo.plotting', 5 task(s):
    - dhi.TestPlots
    - dhi.PlotImpacts
    - dhi.PlotScan
    - dhi.PlotNLL1D
    - dhi.PlotNLL2D

module 'dhi.tasks.nlo.inference', 11 task(s):
    - dhi.NLOLimit
    - dhi.NLOScan1D
    - dhi.NLOScan2D
    - dhi.DCBase
    - dhi.CombDatacards
    - dhi.NLOT2W
    - dhi.NLOBase1D
    - dhi.NLOBase2D
    - dhi.ImpactsPulls
    - dhi.MergeScans1D
    - dhi.MergeScans2D

module 'dhi.tasks.nlo.compare', 2 task(s):
    - dhi.CompareScan
    - dhi.CompareNLL1D

written 18 task(s) to index file '/your/path/inference/.law/index'
```

Now you are done with the setup and can start running the statistical inference!
