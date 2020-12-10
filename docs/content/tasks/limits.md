Run upper limits on the cross-section with the `UpperLimits` task.

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --print-status 2
print task status with max_depth 2 and target_depth 0

> UpperLimits(branch=-1, start_branch=0, end_branch=51, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=kl, poi_range=-25.0,25.0, poi_points=51, r_poi=r, workflow=local)
|   collection: SiblingFileCollection(len=51, threshold=51.0, dir=$DHI_STORE/UpperLimits/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/r__kl/dev)
|     existent (51/51)
|
|  > CreateWorkspace(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |   LocalFileTarget(path=$DHI_STORE/CreateWorkspace/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/workspace.root)
|  |     existent
|  |
|  |  > CombineDatacards(version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault)
|  |  |   LocalFileTarget(path=$DHI_STORE/CombineDatacards/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/dev/datacard.txt)
|  |  |     existent
```
By default it will calculate limits as a function of kappa lambda from -30 to 30 with a step size of 1.
The parameter of interest (poi) can be changed to any of those:

- kappa lambda: "kl"
- kappa top: "kt"
- CV: "CV"
- C2V: "C2V"

**Parameters**

- `--r-poi STRING`: The POI to obtain the upper limit for. Should be any of `r`, `r_gghh`, `r_qqhh`. Defaults to `r`.
- `--poi STRING`: The POI to scan. Defaults to `kl`.
- `--poi-range INT,INT`: The range of the POI to scan. Edges are included. Defaults to the minimum and maximum value of the parameter in the physics model.
- `--poi-points INT`: The number of points to scan. Defaults to a value such that the scan step size is one.

Example:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --r-poi r_qqhh --poi C2V --poi-range=-5,5 --poi-points 20 --print-status 0
print task status with max_depth 0 and target_depth 0

> UpperLimits(branch=-1, start_branch=0, end_branch=20, branches=, version=dev, custom_args=, datacards=hash:0101a84036, mass=125.0, dc_prefix=, hh_model=HHModelPinv:HHdefault, poi=C2V, poi_range=-5.0,5.0, poi_points=20, r_poi=r_qqhh, workflow=local)
|   collection: SiblingFileCollection(len=20, threshold=20.0, dir=$DHI_STORE/UpperLimits/model_HHModelPinv_HHdefault/datacards_d481e43b9e/m125.0/r_qqhh__C2V/dev)
|     absent (0/20)
```


### Multiprocessing

As you can see there will be one output file in the `SiblingFileCollection` for each point. In order to use local multiprocessing to speed up the runtime add `--workflow local` and `--workers 4` to calculate 4 limits in parallel.

Example:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow local --workers 4
```


### HTCondor submission

For heavy workloads, where you need to scan tens or hundreds of points and each fit takes several minutes it might be necessary to submit each fit to the (CERN) HTCondor cluster.

Example:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec
```

If you want to merge e.g. 3 fits in one job you can use the `--tasks-per-job` cli option:

```shell hl_lines="1"
law run UpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --workflow htcondor --poll-interval 30sec --tasks-per-job 3
```


### Plotting

#### Limits vs. POI

The `PlotUpperLimits` task takes the outputs from `UpperLimits` and `MergeUpperLimits` and plots them the usual way:

```shell hl_lines="1"
law run PlotUpperLimits --version dev --datacards $DHI_EXAMPLE_CARDS --r-poi r --xsec fb --br bbwwllvv --y-log
```

![Upper limits](../images/limits__r__kl_n51_-25.0_25.0__fb_bbwwllvv_log.png)


**Parameters**:

- `--xsec STRING`: Convert limits to cross sections in this unit rather than on the signal strength. An empty value (identical to `NO_STR`) will use the latter. Choices are `pb`, `fb` and `""` (`NO_STR`). Defaults to `NO_STR`.
- `--br STRING`: When using `--xsec`, scale the cross section with the BR of the corresponding HH decay. The value should be the name of a final state defined in the `br_hh` mapping in [`dhi.config`](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/config.py). No default.
- `--y-log BOOL`: Logarithmic y-axis. Defaults to `False`.


#### Multiple limits vs. POI

There are two plots that provide a visual comparison between multiple *configurations* - these can be different versions of datacards, or even channels or analyses.

The first one, `PlotMultipleUpperLimits`, draws multiple limits obtained on an `--r-poi` by scanning over a range of `--poi` values, just as done with `PlotUpperLimits`.
The only difference is that only the median limit is shown to provide a better visual aid.

Instead of a parameter `--datacards`, this task introduces a `--multi-datacards` parameter.
It takes several CSV sequences of datacard paths, separated with a colon, e.g. `--multi-datacards card_ee_1.txt,card_ee_2.txt:card_mumu_1.txt,card_mumu_2.txt`.
In this example, the two `card_ee_*.txt` and the two `card_mumu_*.txt` cards will result in two dedicated measurmments, following the same task requirements, i.e., `UpperLimits` and `MergeUpperLimits`, as described above.

```shell hl_lines="1-3"
law run PlotMultipleUpperLimits --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt \
    --datacard-names ee,emu,mumu --xsec fb --y-log
```

![Upper limit comparison](../images/multilimits__r__kl_n51_-25.0_25.0__fb_log.png)

**Parameters**:

Same as `UpperLimits` plus:

- `--multi-datacards STRINGS:STRINGS`: Multiple CSV sequences of datacard paths, separated by colons. Mandatory.
- `--datacard-names STRINGS`: Names of datacard sequences for plotting purposes. When set, the number of names must match the number of sequences in `--multi-datacards`. No default.
- `--datacard-order INTS`: Indices of datacard sequences for reordering during plotting. When set, the number of ids must match the number of sequences in `--multi-datacards`. No default.
- `--xsec STRING`: Convert limits to cross sections in this unit rather than on the signal strength. An empty value (identical to `NO_STR`) will use the latter. Choices are `pb`, `fb` and `""` (`NO_STR`). Defaults to `NO_STR`.
- `--br STRING`: When using `--xsec`, scale the cross section with the BR of the corresponding HH decay. The value should be the name of a final state defined in the `br_hh` mapping in [`dhi.config`](https://gitlab.cern.ch/hh/tools/inference/-/blob/master/dhi/config.py). No default.
- `--y-log BOOL`: Logarithmic y-axis. Defaults to `False`.


#### Multiple limits at a certain POI value

The second task, `PlotUpperLimitsAtPOI`, creates a plot comparing upper limits at a certain POI value.
Just as for the `PlotMultipleUpperLimits` task above, it includes a parameter `--multi-datacards` that accepts multiple CSV sequences of datacard paths, searated with a colon to denote multiple measurements.

Example:

```shell hl_lines="1-3"
law run PlotUpperLimitsAtPOI --version dev \
    --multi-datacards /afs/cern.ch/user/m/mfackeld/public/datacards/ee_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/emu_tight/datacard.txt:/afs/cern.ch/user/m/mfackeld/public/datacards/mumu_tight/datacard.txt \
    --datacard-names ee,emu,mumu
```

![Upper limits at POI](../images/limitatpoi__r__kl_1.0.png)

**Parameters**:

Same as `UpperLimits` plus:

- `--multi-datacards STRINGS:STRINGS`: Multiple CSV sequences of datacard paths, separated by colons. Mandatory.
- `--datacard-names STRINGS`: Names of datacard sequences for plotting purposes. When set, the number of names must match the number of sequences in `--multi-datacards`. No default.
- `--datacard-order INTS`: Indices of datacard sequences for reordering during plotting. When set, the number of ids must match the number of sequences in `--multi-datacards`. No default.
- `--xsec STRING`: Convert limits to cross sections in this unit rather than on the signal strength. An empty value (identical to `NO_STR`) will use the latter. Choices are `pb`, `fb` and `""` (`NO_STR`). Defaults to `NO_STR`.
- `--poi-value FLOAT`: The value of the POI at which limits on the `--r-poi` are obtained. Defaults to `1.`.
- `--x-log BOOL`: Logarithmic x-axis. Defaults to `False`.
