|        Parameter        |                                                                    Description                                                                    |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--campaign`            | Data-taking campaign to be shown in the plot. No default.                                                                                         |
| `--file-type`           | File type of the output plot, `"png"` (default) or `"pdf"`.                                                                                       |
| `--x-min`               | Minimum of x-axis. No default.                                                                                                                    |
| `--x-max`               | Maximum of x-axis. No default.                                                                                                                    |
| `--y-min`               | Minimum of y-axis. No default.                                                                                                                    |
| `--y-max`               | Maximum of y-axis. No default.                                                                                                                    |
| `--y-log`               | Logarithmic y-axis. Defaults to `False`.                                                                                                          |
| `--xsec`                | Convert limits to cross-sections. No default.                                                                                                     |
| `--br`                  | Scale to specific final state, can only be used with `--xsec`. No default.                                                                        |
| `--show-parameters`     | Show parameters in plot. No default.                                                                                                              |
| `--workflow`            | Workflow backend, `"local"` (default) or `"htcondor"`.                                                                                            |
| `--parameter-values`    | Set model parameters to certain values. No default.                                                                                               |
| `--scan-parameters`     | Colon-separated parameters to scan, each in the format `name[,start,stop][,points]`. Defaults to `kl,-30,30,61`.                                  |
| `--pois`                | Comma-separated parameters of interest to pass to combine. Defaults to `r`.                                                                       |
| `--frozen-parameters`   | Comma-separated names of parameters to be frozen in addition to non-POI model and scan parameters. No default.                                    |
| `--frozen-groups`       | Comma-separated names of groups of parameters to be frozen in addition to non-POI  modeland scan parameters. No default.                          |
| `--hh-model`            | Defines the HHModel to be used for the workspace creation in the format `module.model[@option][@...]`. Defaults to `"HHModelPinv.model_default"`. |
| `--datacards`           | Comma-separated paths to input datacards with globbing support. No default.                                                                       |
| `--multi-datacards`     | Colon-separated sequences of datacards, each separated by comma with globbing support. No default.                                                |
| `--datacard-names`      | Names of datacard sequences for plotting purposes. Must match the number of sequences in `--multi-datacards`. No default.                         |
| `--datacard-order`      | Indices of datacard sequences for reordering during plotting. Must match the number of sequences in `--multi-datacards`. No default.              |
| `--z-min`               | Minimum of z-axis. No default.                                                                                                                    |
| `--z-max`               | Maximum of z-axis. No default.                                                                                                                    |
| `--z-log`               | Logarithmic z-axis. No default.                                                                                                                   |
| `--parameters-per-page` | Parameters shown per page, creates a single page when < 1 (default).                                                                              |
| `--skip-parameters`     | List of parameters or files containing parameters line-by-line that should be skipped, with pattern support. No default.                          |
| `--order-parameters`    | list of parameters or files containing parameters line-by-line for ordering, with pattern support. No default.                                    |
| `--order-by-impact`     | When True, `--parameter-order` is neglected and parameters are ordered by absolute maximum impact. No default.                                    |
| `--pull-range`          | The maximum value of pulls on the lower x-axis. Defaults to `2.0`.                                                                                |
| `--impact-range`        | The maximum value of impacts on the upper x-axis. Defaults to `5.0`.                                                                              |
| `--mc-stats`            | Calculate pulls and impacts for MC stats nuisances as well. Defaults to `False`.                                                                  |
