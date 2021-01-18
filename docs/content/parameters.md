| Parameter | Description | Types | Choices | Default |
|---	|---	|---	|---	|---	|
| `--campaign` | Data-taking campaign | `str` | - | `''` |
| `--file-type` | File type of the output plot | `str` | `pdf, png` | `'png'` |
| `--x-min` | Minimum of x-axis | `float` | - | - |
| `--x-max` | Maximum of x-axis | `float` | - | - |
| `--y-min` | Minimum of y-axis | `float` | - | - |
| `--y-max` | Maximum of y-axis | `float` | - | - |
| `--y-log` | Logarithmic y-axis | `bool` | - | `False` |
| `--xsec` | Convert limits to cross-sections | `str` | `'', pb, fb` | `''` |
| `--br` | Scale to specific final state, can only be used with `--xsec` | `str` | `'', bbzzllll, bbww, bbwwlvlv, bbgg, bbzz, bbbb, bbzzqqll, wwww, ttzz, bbwwqqlv, tttt, bbtt, wwzz, zzzz, bbvv,ttww` | `''` |
| `--show-parameters` | Show parameters in plot | `list<str>` | `[]` | - |
| `--workflow` | Computational backend | `str` | `local, htcondor` | `local` |
| `--parameter-values` | Set parameters to certain values | `list<str>` | - | - |
| `--scan-parameters` | Colon-separated parameters to scan, each in the format `name[,start,stop][,points]` | `list<list<str>>>` | - | `[[kl,],]` | 
| `--pois` | Define the parameter(s) of interest | `list<str>` | `r,r_qqhh,r_gghh,kl,kt,CV,C2V` | `['r']` |
| `--frozen-parameters` | Comma-separated names of parameters to be frozen in addition to non-POI and scan parameter, e.g. freeze JES and JER with `--frozen-parameters 'rgx{^CMS_JE}'` | `list<str>` | - | - |
| `--frozen-groups` | Comma-separated names of groups of parameters to be frozen in addition to non-POI and scan parameter | `list<str>` | - | - |
| `--hh-model` | Defines the HHModel to be used for the workspace creation | `str` | - | `HHModelPinv:model_default` |
| `--datacards` | Defines the path to the input datacards, supports globbing | `str` | - | - |
| `--multi-datacards` | Multiple path sequences to input datacards separated by a colon, supports globbing | `str:str:...` | - | - |
| `--datacard-names` | Names of datacard sequences for plotting purposes. When set, the number of names must match the number of sequences in --multi-datacards | `str` | - | - |
| `--datacard-order` | Indices of datacard sequences for reordering during plotting. When set, the number of ids must match the number of sequences in --multi-datacards | `int` | - | - |
| `--z-min` | Minimum of z-axis | `float` | - | - |
| `--z-max` | Maximum of z-axis | `float` | - | - |
| `--z-log` | Logarithmic z-axis | `bool` | - | `False` |
| `--parameters-per-page` | Parameters shown per page, creates a single page when < 1 | `int` | - | `-1` | 
| `--skip-parameters` | List of parameters or files containing parameters line-by-line that should be skipped, supports patterns | `str` | - | `''` |
| `--order-parameters` | list of parameters or files containing parameters line-by-line for ordering, supports pattern | `str` | - | `''` |
| `--order-by-impact` | When True, `--parameter-order` is neglected and parameters are ordered by absolute maximum impact | `bool` | - | `False` | 
| `--pull-range` | The maximum value of pulls on the lower x-axis | `float` | - | `2.0` |
| `--impact-range` | The maximum value of impacts on the upper x-axis | `float` | - | `5.0` |
| `--mc-stats` | Calculate pulls and impacts for MC stats nuisances as well | `bool` | - | `False` |