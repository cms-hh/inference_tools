There are a couple of scripts located in [dhi/scripts](https://gitlab.cern.ch/hh/tools/inference/-/tree/master/dhi/scripts) that allow you to manipulate datacards from the command line.

All scripts have the ability to move, or *bundle* a datacard and all the shape files it refers to into a new directory in order to apply manipulations on copies of the original files.
When doing so, the locations of shape files are changed consistently within the datacard.
The argument to configure this directory is identical across all scripts.

```shell
script_name.py DATACARD [OTHER_ARGUMENTS] --directory/-d DIRECTORY
```

Please note that, when no output directory is given, ==the content of datacards and shape files is changed in-place==.


## Adjusting parameters

### Remove

```shell hl_lines="1"
> remove_parameters.py --help

usage: remove_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--log-level LOG_LEVEL]
                            DATACARD NAME [NAME ...]

Script to remove one or multiple (nuisance) parameters from a datacard.
Example usage:

# remove certain parameters
> remove_parameters.py datacard.txt CMS_btag_JES CMS_btag_JER -d output_directory

# remove parameters via fnmatch wildcards (note the quotes)
> remove_parameters.py datacard.txt "CMS_btag_JE?" -d output_directory

# remove parameters listed in a file
> remove_parameters.py datacard.txt parameters.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of parameters or files containing parameter
                        names to remove line by line; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


### Rename

```shell hl_lines="1"
> rename_parameters.py --help

usage: rename_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--mass MASS] [--log-level LOG_LEVEL]
                            DATACARD OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple (nuisance) parameters in a datacard.
Example usage:

# rename via simple rules
> rename_parameters.py datacard.txt btag_JES=CMS_btag_JES -d output_directory

# rename via rules in files
> rename_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple parameter names
                        in the format 'old_name=new_name', or files containing
                        these rules in the same format line by line

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not change parameter names in shape files
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


### Add

```shell hl_lines="1"
> add_parameter.py --help

usage: add_parameter.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                        [--log-level LOG_LEVEL]
                        DATACARD NAME TYPE [SPEC [SPEC ...]]

Script to add arbitrary parameters to the datacard.
Example usage:

# add auto MC stats
> add_parameter.py datacard.txt "*" autoMCStats 10 -d output_directory

# add a lnN nuisance for a specific process across all bins
> add_parameter.py datacard.txt new_nuisance lnN "*,ttZ,1.05" -d output_directory

# add a lnN nuisance for all processes in two specific bins
> add_parameter.py datacard.txt new_nuisance lnN "bin1,*,1.05" "bin2,*,1.07" -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  name of the parameter to add
  TYPE                  type of the parameter to add
  SPEC                  specification of parameter arguments; for columnar
                        parameter types (e.g. lnN or shape* nuisances), comma-
                        separated triplets in the format 'bin,process,value'
                        are expected; patterns are supported and evaluated in
                        the given order for all existing bin process pairs;
                        for all other types, the specification is used as is

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


### Merge

```shell hl_lines="1 24-27"
> merge_parameters.py --help

usage: merge_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--flip-parameters FLIP_PARAMETERS]
                           [--auto-rate-flip] [--auto-rate-max]
                           [--auto-rate-envelope] [--auto-shape-average]
                           [--auto-shape-envelope] [--digits DIGITS]
                           [--mass MASS] [--log-level LOG_LEVEL]
                           DATACARD MERGED_NAME names [names ...]

Script to merge multiple (nuisance) parameters of the same type into a new, single one.
Currently, only parameters with columnar type "lnN", "lnU" and "shape" are supported.
Example usage:

# merge two parameters
> merge_parameters.py datacard.txt CMS_eff_m_combined CMS_eff_m_iso CMS_eff_m_id -d output_directory

# merge parameters via fnmatch wildcards (note the quotes)
> merge_parameters.py datacard.txt CMS_eff_m_combined "CMS_eff_m_*" -d output_directory

Note 1: The use of an output directory is recommended to keep input files unchanged.

Note 2: This script is not intended to be used to merge incompatible systematic uncertainties. Its
        only purpose is to reduce the number of parameters by merging the effect of (probably small)
        uncertainties that are related at analysis level, e.g. multiple types of lepton
        efficiency corrections. Please refer the doc string of "merge_parameters()" for more info.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  MERGED_NAME           name of the newly merged parameter
  names                 names of parameters or files containing names of
                        parameters line by line to merge; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --flip-parameters FLIP_PARAMETERS
                        comma-separated list of parameters whose effect should
                        be flipped, i.e., flips effects of up and down
                        variations; supports patterns
  --auto-rate-flip      only for lnN and lnU; when set, up and down variations
                        of a parameter are swapped when they change the rate
                        in the relative opposite directions; otherwise, an
                        error is raised
  --auto-rate-max       only for lnN and lnU; when set, the maximum effect of
                        a parameter is used when both up and down variation
                        change the rate in the same direction; otherwise, an
                        error is raised
  --auto-rate-envelope  only for lnN and lnU; when set, the effect on the new
                        parameter is constructed as the envelope of effects of
                        parameters to merge
  --auto-shape-average  only for shape; when set and shapes to merge contain
                        both positive negative effects in the same bin,
                        propagate errors separately and then use their
                        average; otherwise, an error is raised
  --auto-shape-envelope
                        only for shape; when set, the merged shape variations
                        of the new parameter are constructed as the envelopes
                        of shapes of parameters to merge
  --digits DIGITS       the amount of digits for rounding merged parameters;
                        defaults to 3
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


## Adjusting processes

### Remove

```shell hl_lines="1"
> remove_processes.py --help

usage: remove_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--log-level LOG_LEVEL]
                           DATACARD NAME [NAME ...]

Script to remove one or multiple processes from a datacard.
Example usage:

# remove certain processes
> remove_processes.py datacard.txt qqHH_CV_1_C2V_2_kl_1 -d output_directory

# remove processes via fnmatch wildcards (note the quotes)
> remove_processes.py datacard.txt "qqHH_CV_1_C2V_*_kl_1" -d output_directory

# remove processes listed in a file
> remove_processes.py datacard.txt processes.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of processes or files containing process names
                        to remove line by line; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


### Rename

```shell hl_lines="1"
> rename_processes.py --help

usage: rename_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--mass MASS] [--log-level LOG_LEVEL]
                           DATACARD OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple processes in a datacard.
Example usage:

# rename via simple rules
> rename_processes.py datacard.txt ggH_process=ggHH_kl_1_kt_1 -d output_directory

# rename via rules in files
> rename_processes.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple process names in
                        the format 'old_name=new_name', or files containing
                        these rules in the same format line by line

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not change process names in shape files
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```

## Adjusting bin process pairs

### Remove

```shell hl_lines="1"
> remove_bin_process_pairs.py --help

usage: remove_bin_process_pairs.py [-h] [--directory [DIRECTORY]]
                                   [--no-shapes] [--log-level LOG_LEVEL]
                                   DATACARD BIN_NAME,PROCESS_NAME
                                   [BIN_NAME,PROCESS_NAME ...]

Script to remove one or multiple bin process pairs from a datacard.
Example usage:

# remove a certain bin process pair
> remove_bin_process_pairs.py datacard.txt ch1,ttZ -d output_directory

# remove all processes for a specific bin via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt "ch1,*" -d output_directory

# remove all bins for a specific process via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt "*,ttZ" -d output_directory

# remove bin process pairs listed in a file
> remove_bin_process_pairs.py datacard.txt pairs.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  BIN_NAME,PROCESS_NAME
                        names of bin process pairs to remove in the format
                        'bin_name,process_name' or files containing these
                        pairs line by line; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


## Adjusting bins

### Remove

```shell hl_lines="1"
> remove_bins.py --help

usage: remove_bins.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                      [--log-level LOG_LEVEL]
                      DATACARD NAME [NAME ...]

Script to remove one or multiple bins from a datacard.
Example usage:

# remove certain bins
> remove_bins.py datacard.txt ch1 -d output_directory

# remove bins via fnmatch wildcards (note the quotes)
> remove_bins.py datacard.txt "ch*" -d output_directory

# remove bins listed in a file
> remove_bins.py datacard.txt bins.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of bins or files containing bin names to remove
                        line by line; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```


## Miscellaneous

### Prettify datacard

```shell hl_lines="1"
> prettify_datacard.py --help

usage: prettify_datacard.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--no-preamble] [--log-level LOG_LEVEL]
                            DATACARD

Script to prettify a datacard.
Example usage:

> prettify_datacard.py datacard.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --no-preamble         remove any existing preamble
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
```
