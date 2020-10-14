This section will explain how to produce plots from finalized fits and scans.

Default: Every law.Task comes with a `--version` parameter, in order to handle multiple inference analysis in parallel.
Note: Omit the `--print-status` cli option in order to run the task!


## Upper Limits

![Upper limits](images/limits__kl_n61_-30.0_30.0__xsec_brbbwwllvv_log.png)

Recreate this plot with:
```shell
law run PlotUpperLimits --version dev --workers 10 --xsec --br bbww --y-log
```

Cli parameters:

- `--workers 10`: local multiprocessing
- `--xsec`: calculate the limit on the cross-section and not the signal strength
- `--br`: when using `--xsec`, scale the cross section with the BR of the corresponding HH decay (see: [BRs](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/config.py#L14-49))
- `--y-log`: logarithmic y-axis


## 1D Likelihood Scans

![1D Likelihood Scan](images/nll1d__kl_n61_-30.0_30.0.png)

Recreate this plot with:
```shell
law run PlotLikelihoodScan1D --version dev --workers 10
```

Cli parameters:

- `--workers 10`: local multiprocessing


## 2D Likelihood Scans

![2D Likelihood Scan](images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log.png)

Recreate this plot with:
```shell
law run PlotLikelihoodScan2D --version dev --LikelihoodScan2D-workflow htcondor --LikelihoodScan2D-tasks-per-job 10 --z-log
```

Cli parameters:

- `--LikelihoodScan2D-workflow htcondor`: Run the actual scan on HTCondor
- `--LikelihoodScan2D-tasks-per-job 10`: Merge 10 scans into one job
- `--z-log`: logarithmic z-axis


---
**_NOTES_**

For more options use the autocompletion of law:

```shell
law run PlotLikelihoodScan2D <tab><tab>
```
