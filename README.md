# Statistical inference HH(bbWW)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## First steps (setup)

Connect to a CentOS 7 node on lxplus:
```bash
ssh <cern_username>@lxplus.cern.ch
```

Clone this repository:
```bash
git clone https://gitlab.cern.ch/cms-hh-bbww/statistical-inference.git
cd statistical-inference
```

Now you are good to go!


## law

[`law`](https://github.com/riga/law) can be used to build complex and large-scale task workflows.
It is build on top of [`luigi`](https://github.com/spotify/luigi) and adds abstractions for run locations, storage locations and software environments.
Law strictly disentangles these building blocks and ensures they remain interchangeable and resource-opportunistic.

The setup is defined in `setup.sh`.

1. Setup law together with CMSSW 10_2_X:
```bash
source setup.sh
```

2. Let law index your tasks and their parameters (for autocompletion)
```bash
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

module 'dhi.tasks.nlo.inference', 9 task(s):
    - dhi.NLOLimit
    - dhi.DCBase
    - dhi.CombDatacards
    - dhi.NLOT2W
    - dhi.NLOBase1D
    - dhi.NLOBase2D
    - dhi.ImpactsPulls
    - dhi.NLOScan1D
    - dhi.NLOScan2D

module 'dhi.tasks.nlo.compare', 2 task(s):
    - dhi.CompareScan
    - dhi.CompareNLL1D

written 16 task(s) to index file '/your/path/statistical-inference/.law/index'
```

## NLO scans

After successfully running the setup script, everything should be up and running to start inference and plotting tasks.

Now we can check the status of our tasks:
```bash
law run dhi.PlotScan --version dev1 --poi kl --print-status 4
```

**Note**: You can specify `--input-cards datacard1.txt,datacard2.txt`, where `datacard1.txt,datacard2.txt` are comma seperated paths (with globbing support) to your input datacards, in case you don't want to use the default example datacards.

You should see, where `your/path` is set in setup.sh with `DHI_STORE`:

```bash
print task status with max_depth 4 and target_depth 0

> check status of dhi.PlotScan(version=dev1, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/*/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-40,40)
|  - LocalFileTarget(path=/your/path/dhi/store/PlotScan/dev1/125/HHdefault/kl_-40_40/scan.pdf)
|    absent
|
|  > check status of dhi.NLOLimit(branch=-1, start_branch=0, end_branch=81, branches=, version=dev1, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/*/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False, poi=kl, poi_range=-40,40, workflow=local)
|  |  collection: SiblingFileCollection(len=81, threshold=81.0, dir=/your/path/dhi/store/NLOLimit/dev1/125/HHdefault/kl_-40_40)
|  |    absent (0/81)
|  |
|  |  > check status of dhi.NLOT2W(version=dev1, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/*/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  |  - LocalFileTarget(path=/your/path/dhi/store/NLOT2W/dev1/125/HHdefault/workspace_HHdefault.root)
|  |  |    absent
|  |  |
|  |  |  > check status of dhi.CombDatacards(version=dev1, mass=125, input_cards=/afs/cern.ch/user/m/mfackeld/public/datacards/*/datacard.txt, dc_prefix=, hh_model=HHdefault, stack_cards=False)
|  |  |  |  datacard: LocalFileTarget(path=/your/path/dhi/store/CombDatacards/dev1/125/HHdefault/datacard.txt)
|  |  |  |    absent
```

You see that the output of all tasks are "absent", which is obviously, because we have not run them yet.

Let's run the tasks! (Simply remove the `--print-status 4` cli argument. `4` denotes the recursion depth of tasks to consider.)

For doing the kappa lambda scan:

```bash
law run dhi.PlotScan --version dev1 --poi kl
```

For doing the C2V scan:

```bash
law run dhi.PlotScan --version dev1 --poi C2V
```

You can speed these scans up using multiprocessing.
Just add e.g. `--workers 4` to the `law run ...` command to scan 4 points in parallel using multiprocessing.

Also, you can set the range to scan by adding `--poi-range=start,stop` with both `start` and `stop` being integers.

Once everything is finished (takes some time to scan all those points ...) you can go to https://cernbox.cern.ch, login with your cern account and go to `dhi/store/PlotScan/dev1/scan.pdf` and view the plot.
You can also view the intermediate results for each task here: `dhi/store/<TASKNAME>/dev1/`, where `<TASKNAME>` corresponds to the python class names of the task.

If your have a visualization tool installed that supports showing images and pdfs right in your terminal (such as `imgcat`), add `--view-cmd imgcat` (example) to the `law run` command that produces a plot.

If you want to find out what the best fit value is for your parameter of interest, do:

```bash
law run dhi.PlotNLL1D --version dev1 --poi kl
```

Result plot can be inspected in https://cernbox.cern.ch/ under `dhi/store/PlotNLL1D/dev1/125/kl_-20_20/nll.pdf`.

### HTCondor submission
All fit/scan tasks are `law.Workflows` internally and can be submitted to the CERN HTCondor batch system:
* 1D scan:
```bash
law run dhi.NLOScan1D --version dev1 --workflow htcondor --tasks-per-job 4 --transfer-logs --poll-interval 30sec
```
* 2D scan:
```bash
law run dhi.NLOScan2D --version dev1 --workflow htcondor --tasks-per-job 5 --transfer-logs --poll-interval 30sec
```
* Limits:
```bash
law run dhi.NLOLimit --version dev1 --workflow htcondor --tasks-per-job 2 --transfer-logs --poll-interval 30sec
```

## LO scans

soon...


## Additional information

You can clear the target outputs interactively by adding the `--remove-output <depth>` cli argument.
It takes again a task depth as argument in the same way as `--print-status <depth>`.
E.g. remove the output of the current task by adding `--remove-output 0`.
For a quick introduction into some of the features of `law`, check out the [introduction notebook](https://mybinder.org/v2/gh/riga/law/master?filepath=examples%2Floremipsum%2Findex.ipynb).


## For developers

Code style is enforced with the formatter "black": https://github.com/psf/black.
The default line width is increased to 100 (see `pyproject.toml`).

## Contributors
* Peter Fackeldey: peter.fackeldey@cern.ch (email)
* Marcel Rieger: marcel.rieger@cern.ch (email)
