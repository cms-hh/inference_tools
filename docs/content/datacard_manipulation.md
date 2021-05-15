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
                            [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                            DATACARD SPEC [SPEC ...]

Script to remove one or multiple (nuisance) parameters from a datacard.
Example usage:

# remove certain parameters
> remove_parameters.py datacard.txt CMS_btag_JES CMS_btag_JER -d output_directory

# remove parameters via fnmatch wildcards
# (note the quotes)
> remove_parameters.py datacard.txt 'CMS_btag_JE?' -d output_directory

# remove a parameter from all processes in a certain bin
# (note the quotes)
> remove_parameters.py datacard.txt 'OS_2018,*,CMS_btag_JES' -d output_directory

# remove a parameter from a certain processes in all bins
# (note the quotes)
> remove_parameters.py datacard.txt '*,tt,CMS_btag_JES' -d output_directory

# remove parameters listed in a file
> remove_parameters.py datacard.txt parameters.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  SPEC                  specifications of parameters to remove or a file
                        containing these specifications line by line; a
                        specification should have the format
                        '[BIN,PROCESS,]PARAMETER'; when a bin name and process
                        are defined, the parameter should be of a type that is
                        defined on a bin and process basis, and is removed
                        only in this bin process combination; all values
                        support patterns; prepending '!' to a bin or process
                        pattern negates its meaning

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_parameters
```


### Rename

```shell hl_lines="1"
> rename_parameters.py --help

usage: rename_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--mass MASS] [--log-level LOG_LEVEL]
                            [--log-name LOG_NAME]
                            DATACARD OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple (nuisance) parameters in a datacard.
Example usage:

# rename via simple rules
> rename_parameters.py datacard.txt btag_JES=CMS_btag_JES -d output_directory

# rename via rules in files
> rename_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple parameter names
                        in the format 'OLD_NAME=NEW_NAME', or files containing
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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        rename_parameters
```


### Add

```shell hl_lines="1"
> add_parameter.py --help

usage: add_parameter.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                        [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                        DATACARD NAME TYPE [SPEC [SPEC ...]]

Script to add arbitrary parameters to the datacard.
Example usage:

# add auto MC stats
> add_parameter.py datacard.txt '*' autoMCStats 10 -d output_directory

# add a lnN nuisance for a specific process across all bins (note the quotes)
> add_parameter.py datacard.txt new_nuisance lnN '*,ttZ,1.05' -d output_directory

# add a lnN nuisance for all processes in two specific bins (note the quotes)
> add_parameter.py datacard.txt new_nuisance lnN 'bin1,*,1.05' 'bin2,*,1.07' -d output_directory

# add a lnN nuisance for all but ttbar processes in all bins (note the quotes)
> add_parameter.py datacard.txt new_nuisance lnN '*,!tt*,1.05' -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  name of the parameter to add
  TYPE                  type of the parameter to add
  SPEC                  specification of parameter arguments; for columnar
                        parameter types (e.g. lnN or shape* nuisances), comma-
                        separated triplets in the format '[BIN,PROCESS,]VALUE'
                        are expected; when no bin and process names are given,
                        the parameter is added to all existing ones; patterns
                        are supported and evaluated in the given order for all
                        existing bin process pairs; prepending '!' to a
                        pattern negates its meaning; for all other types, the
                        specification is used as is

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        add_parameter
```


### Merge

```shell hl_lines="1 24-28"
> merge_parameters.py --help

usage: merge_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--unique] [--flip-parameters FLIP_PARAMETERS]
                           [--auto-rate-flip] [--auto-rate-max]
                           [--auto-rate-envelope] [--auto-shape-average]
                           [--auto-shape-envelope] [--digits DIGITS]
                           [--mass MASS] [--log-level LOG_LEVEL]
                           [--log-name LOG_NAME]
                           DATACARD MERGED_NAME names [names ...]

Script to merge multiple (nuisance) parameters of the same type into a new,
single one. Currently, only parameters with columnar type "lnN", "lnU" and
"shape" are supported. Example usage:

# merge two parameters
> merge_parameters.py datacard.txt CMS_eff_m CMS_eff_m_iso CMS_eff_m_id -d output_directory

# merge parameters via fnmatch wildcards (note the quotes)
> merge_parameters.py datacard.txt CMS_eff_m 'CMS_eff_m_*' -d output_directory

Note 1: The use of an output directory is recommended to keep input files
        unchanged.

Note 2: This script is not intended to be used to merge incompatible systematic
        uncertainties. Its only purpose is to reduce the number of parameters by
        merging the effect of (probably small) uncertainties that are related at
        analysis level, e.g. multiple types of lepton efficiency corrections.
        Please refer the doc string of "merge_parameters()" for more info.

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
  --unique, -u          only merge parameters when at most on of them as an
                        effect in a bin process pair
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
                        both positive abd negative effects in the same bin,
                        propagate errors separately and then use their
                        average; otherwise, an error is raised
  --auto-shape-envelope
                        only for shape; when set, the merged shape variations
                        of the new parameter are constructed as the envelopes
                        of shapes of parameters to merge
  --digits DIGITS       the amount of digits for rounding merged parameters;
                        defaults to 4
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        merge_parameters
```


### Split

```shell hl_lines="1"
> split_parameter.py --help

usage: split_parameter.py [-h] [--ensure-unique] [--ensure-all]
                          [--directory [DIRECTORY]] [--no-shapes]
                          [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                          DATACARD PARAM_NAME NEW_NAME,BIN,PROCESS
                          [NEW_NAME,BIN,PROCESS ...]

Script to split a "lnN" or "lnU" (nuisance) parameter into several ones,
depending on expressions matching relevant bin and process names. Example usage:

# split the "lumi" parameter depending on the year, encoded in the bin names
# (note the quotes)
> split_parameter.py datacard.txt lumi 'lumi_2017,*2017*,*' 'lumi_2018,*2018*,*'

# split the "pdf" parameter depending on the process name (note the quotes)
> split_parameter.py datacard.txt pdf 'pdf_ttbar,*,TT' 'pdf_st,*,ST'

# split the "pdf" parameter depending on the process name with pattern negation
# (note the quotes)
> split_parameter.py datacard.txt pdf 'pdf_ttbar,*,TT' 'pdf_rest,*,!TT'

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  PARAM_NAME            name of the parameter to split
  NEW_NAME,BIN,PROCESS  specification of new parameters, each in the format
                        'NEW_NAME,BIN,PROCESS'; supports patterns; prepending
                        '!' to a pattern negates its meaning

optional arguments:
  -h, --help            show this help message and exit
  --ensure-unique, -u   when set, a check is performed to ensure that each
                        value is assigned to not more than one new parameter
  --ensure-all, -a      when set, a check is performed to ensure that each
                        value is assigned to at least one new parameter
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        split_parameter
```


### Flip

```shell hl_lines="1"
> flip_parameters.py --help

usage: flip_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                          [--mass MASS] [--log-level LOG_LEVEL]
                          [--log-name LOG_NAME]
                          DATACARD NAME [NAME ...]

Script to flip the effect of one or multiple (nuisance) parameters in a
datacard. Currently, only parameters with columnar type "lnN", "lnU" and "shape"
are supported. Example usage:

# flip via simple names
> flip_parameters.py datacard.txt alpha_s_ttH alpha_s_tt -d output_directory

# flip via name patterns (note the quotes)
> flip_parameters.py datacard.txt 'alpha_s_*' -d output_directory

# flip via bin, process and name patterns (note the quotes)
> flip_parameters.py datacard.txt '*,ch1,alpha_s_*' -d output_directory

# flip via rules in files
> flip_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of parameters whose effect should be flipped in
                        the format '[BIN,PROCESS,]PARAMETER'; when a bin and
                        process names are given, the effect is only flipped in
                        those; patterns are supported; prepending '!' to a
                        pattern negates its meaning; a name can also refer to
                        a file with names in the above format line by line

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        flip_parameters
```


## Adjusting pairs of datacard bins and processes

### Remove

```shell hl_lines="1"
> remove_bin_process_pairs.py --help

usage: remove_bin_process_pairs.py [-h] [--directory [DIRECTORY]]
                                   [--no-shapes] [--log-level LOG_LEVEL]
                                   [--log-name LOG_NAME]
                                   DATACARD BIN_NAME,PROCESS_NAME
                                   [BIN_NAME,PROCESS_NAME ...]

Script to remove one or multiple bin process pairs from a datacard.
Example usage:

# remove a certain bin process pair
> remove_bin_process_pairs.py datacard.txt ch1,ttZ -d output_directory

# remove all processes for a specific bin via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt 'ch1,*' -d output_directory

# remove all bins for a specific process via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt '*,ttZ' -d output_directory

# remove bin process pairs listed in a file
> remove_bin_process_pairs.py datacard.txt pairs.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  BIN_NAME,PROCESS_NAME
                        names of bin process pairs to remove in the format
                        'bin_name,process_name' or files containing these
                        pairs line by line; supports patterns; prepending '!'
                        to a pattern negates its meaning

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_bin_process_pairs
```


## Adjusting processes

### Remove

```shell hl_lines="1"
> remove_processes.py --help

usage: remove_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                           DATACARD NAME [NAME ...]

Script to remove one or multiple processes from a datacard.
Example usage:

# remove certain processes
> remove_processes.py datacard.txt qqHH_CV_1_C2V_2_kl_1 -d output_directory

# remove processes via fnmatch wildcards (note the quotes)
> remove_processes.py datacard.txt 'qqHH_CV_1_C2V_*_kl_1' -d output_directory

# remove processes listed in a file
> remove_processes.py datacard.txt processes.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of processes or files containing process names
                        to remove line by line; supports patterns; prepending
                        '!' to a pattern negates its meaning

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_processes
```


### Remove depending on rate

```shell hl_lines="1"
> remove_empty_processes.py --help

usage: remove_empty_processes.py [-h] [--skip-signal]
                                 [--directory [DIRECTORY]] [--no-shapes]
                                 [--mass MASS] [--log-level LOG_LEVEL]
                                 [--log-name LOG_NAME]
                                 DATACARD BIN,PROCESS,THRESHOLD
                                 [BIN,PROCESS,THRESHOLD ...]

Script to remove processes from datacard bins whose rate is below a certain
threshold. Bins, processes and the threshold value can be fully configured.
Example usage:

# remove a certain process from all bins where its rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt '*,tt,0.1' -d output_directory

# remove all processes in a certain bin whose rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt 'OS_2018,*,0.1' -d output_directory

# remove all processes except signal in a certain bin whose rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt 'OS_2018,*,0.1' -s -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  BIN,PROCESS,THRESHOLD
                        names of bins, processes and a threshold value below
                        which processes are removed in the format
                        'bin_name,process_name,threshold_value'; both names
                        support patterns where a leading '!' negates their
                        meaning; each argument can also be a file containing
                        'BIN,PROCESS,THRESHOLD' values line by line

optional arguments:
  -h, --help            show this help message and exit
  --skip-signal, -s     skip signal processes, independent of whether they are
                        matched by a process pattern
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not change process names in shape files
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_empty_processes
```


### Rename

```shell hl_lines="1"
> rename_processes.py --help

usage: rename_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--mass MASS] [--log-level LOG_LEVEL]
                           [--log-name LOG_NAME]
                           DATACARD OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple processes in a datacard.
Example usage:

# rename via simple rules
> rename_processes.py datacard.txt ggH_process=ggHH_kl_1_kt_1 -d output_directory

# rename via rules in files
> rename_processes.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple process names in
                        the format 'OLD_NAME=NEW_NAME', or files containing
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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        rename_processes
```


## Adjusting datacard bins

### Remove

```shell hl_lines="1"
> remove_bins.py --help

usage: remove_bins.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                      [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                      DATACARD NAME [NAME ...]

Script to remove one or multiple bins from a datacard.
Example usage:

# remove certain bins
> remove_bins.py datacard.txt ch1 -d output_directory

# remove bins via fnmatch wildcards (note the quotes)
> remove_bins.py datacard.txt 'ch*' -d output_directory

# remove bins listed in a file
> remove_bins.py datacard.txt bins.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  NAME                  names of bins or files containing bin names to remove
                        line by line; supports patterns; prepending '!' to a
                        pattern negates its meaning

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_bins
```


## Adjusting shape bins

### Remove

```shell hl_lines="1"
> remove_shape_bins.py --help

usage: remove_shape_bins.py [-h] [--directory [DIRECTORY]] [--mass MASS]
                            [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                            DATACARD BIN,EXPRESSION[,EXPRESSION]
                            [BIN,EXPRESSION[,EXPRESSION] ...]

Script to remove histogram bins from datacard shapes using configurable rules.
Shapes stored in workspaces are not supported. The bins to remove can be hard
coded, depend on signal or background content, or be identified through a
custom function. Example usage:

# remove the first 5 shape bins in a specific datacard bin
> remove_shape_bins.py datacard.txt 'OS_2018,1-5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events AND
# a S/sqrt(B) ratio (signal-to-noise) above 0.5
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5,STN>0.5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events OR
# a S/sqrt(B) ratio (signal-to-noise) above 0.5
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5' '*,STN>0.5' -d output_directory

# remove shape bins in all datacard bins using an exteral function
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,my_module.func_name" -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  BIN,EXPRESSION[,EXPRESSION]
                        removal rules for shape bins in a datacard bin 'BIN',
                        which supports patterns; prepending '!' to a bin
                        pattern negates its meaning; an 'EXPRESSION' can
                        either be a list of colon-separated bin indices to
                        remove (starting at 1) with values 'A-B' being
                        interpreted as ranges from A to B (inclusive), a
                        simple expression 'PROCESS(<|>)THRESHOLD' (with
                        special processes 'S', 'B', 'SB', 'SOB' and 'STN'
                        being interpreted as combined signal, background,
                        signal+background, signal/background, and
                        signal/sqrt(background)), or the location of a
                        function in the format 'module.func_name' with
                        signature (datacard_content, datacard_bin, histograms)
                        that should return indices of bins to remove; mutliple
                        rules passed in the same expression are AND
                        concatenated; the rules of multiple arguments are OR
                        concatenated; each argument can also be a file
                        containing 'BIN,EXPRESSION[,EXPRESSION]' values line
                        by line

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        remove_shape_bins
```


### Update

```shell hl_lines="1"
> update_shape_bins.py --help

usage: update_shape_bins.py [-h] [--batch-processes] [--directory [DIRECTORY]]
                            [--mass MASS] [--log-level LOG_LEVEL]
                            [--log-name LOG_NAME]
                            DATACARD BIN,PROCESS,FUNCTION
                            [BIN,PROCESS,FUNCTION ...]

Script to update histogram bins in datacard shapes using configurable rules.
Shapes stored in workspaces are not supported. Histograms can be updated
in-place using a referenceable function that is called with the signature
(bin_name, process_name, nominal_shape, systematic_shapes), where the latter
is a dictionary mapping systematic names to down and up varied shapes. Example
usage:

# file my_code.py
# ---------------
def func(bin_name, process_name, nominal_shape, systematic_shapes):
    for b in range(1, nominal_shape.GetNbinsX() + 1):
        nominal_shape.SetBinContent(b, ...)
# ---------------

# apply a function in all datacard bins to a specific process
# (note the quotes)
> update_shape_bins.py datacard.txt '*,ttbar,my_code.func' -d output_directory

# apply a function to all process in in all but one specific datacard bins
# (note the quotes)
> update_shape_bins.py datacard.txt '!CR,*,my_code.func' -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

positional arguments:
  DATACARD              the datacard to read and possibly update (see
                        --directory)
  BIN,PROCESS,FUNCTION  rules for updating datacard shape bins; 'BIN' and
                        'PROCESS' support patterns where a prepended '!'
                        negates their meaning; 'FUNCTION' should have the
                        format <MODULE_NAME>.<FUNCTION_NAME> to import a
                        function 'FUNCTION_NAME' from the module
                        'MODULE_NAME'; the function should have the signature
                        (bin_name, process_name, nominal_hist, syst_hists);
                        this parameter also supports files that contain the
                        rules in the described format line by line

optional arguments:
  -h, --help            show this help message and exit
  --batch-processes, -b
                        handle all processes in a bin by a single call to the
                        passed function; 'process_name', 'nominal_hist' and
                        'syst_hists' will be lists of the same length
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        update_shape_bins
```


## Plotting

### Plot systematic shifts of datacard shapes

```shell hl_lines="1"
> plot_datacard_shapes.py --help

usage: plot_datacard_shapes.py [-h] [--stack] [--directory DIRECTORY]
                               [--nom-format NOM_FORMAT]
                               [--syst-format SYST_FORMAT] [--mass MASS]
                               [--binning {original,numbers,numbers_width}]
                               [--x-title X_TITLE] [--y-min Y_MIN]
                               [--y-max Y_MAX] [--y-min2 Y_MIN2]
                               [--y-max2 Y_MAX2] [--y-log]
                               [--campaign CAMPAIGN] [--log-level LOG_LEVEL]
                               [--log-name LOG_NAME]
                               DATACARD BIN,PROCESS[,SYSTEMATIC]
                               [BIN,PROCESS[,SYSTEMATIC] ...]

Script to plot histogram shapes of a datacard using configurable rules.
Shapes stored in workspaces are not supported. Example usage:

# plot all nominal shapes in a certain datacard bin
# (note the quotes)
> plot_datacard_shapes.py datacard.txt 'ee_2018,*'

# plot all nominal shapes of a certain process in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,ttbar'

# plot all systematic shapes of a certain process in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,ttbar,*'

# plot all systematic shapes of all signals in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,S,*'

# plot all systematic shapes of two stacked processes in all bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt --stack '*,ttbar+singlet,*'

positional arguments:
  DATACARD              the datacard to read
  BIN,PROCESS[,SYSTEMATIC]
                        rules defining which shapes to plot; 'BIN', 'PROCESS'
                        and 'SYSTEMATIC' support patterns where a prepended
                        '!' negates their meaning; special process names are
                        'S', 'B', and 'SB' which are interpreted as combined
                        signal, background, and signal+background; multiple
                        process patterns can be concatenated with '+'; when no
                        'SYSTEMATIC' is given, only nominal shapes are
                        plotted; special systematic names 'S' and 'R' are
                        interpreted as all shape and all rate systematics;
                        this parameter also supports files that contain the
                        rules in the described format line by line

optional arguments:
  -h, --help            show this help message and exit
  --stack, -s           instead of creating separate plots per process machted
                        by a rule, stack distributions and create a single
                        plot
  --directory DIRECTORY, -d DIRECTORY
                        directory in which produced plots are saved; defaults
                        to the current directory
  --nom-format NOM_FORMAT
                        format for created files when plotting only nominal
                        shapes; default: {bin}__{process}.pdf
  --syst-format SYST_FORMAT
                        format for created files when plotting only nominal
                        shapes; default: {bin}__{process}__{syst}.pdf
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --binning {original,numbers,numbers_width}, -b {original,numbers,numbers_width}
                        the binning strategy; 'original': use original bin
                        edges; 'numbers': equidistant edges using bin numbers;
                        'numbers_width': same as 'numbers' and divide by bin
                        widths
  --x-title X_TITLE     x-axis label; default: 'Datacard shape'
  --y-min Y_MIN         min y value of the top pad; no default
  --y-max Y_MAX         max y value of the top pad; no default
  --y-min2 Y_MIN2       min y value of the bottom pad; no default
  --y-max2 Y_MAX2       max y value of the bottom pad; no default
  --y-log               transform y-axis to log scale
  --campaign CAMPAIGN   label to be shown at the top right; no default
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        plot_datacard_shapes
```


## Miscellaneous

### Extract datacard content as json

```shell hl_lines="1"
> extract_datacard_content.py --help

usage: extract_datacard_content.py [-h] [--output OUTPUT] [--log-level LOG_LEVEL]
                                [--log-name LOG_NAME]
                                DATACARD

Script to extract datacard content into a json file in the structure:

{
    "bins": [{"name": bin_name}],
    "processes": [{"name": process_name, "id": process_id}],
    "rates": {bin_name: {process_name: float}},
    "observations": {bin_name: float},
    "parameters": [{"name": string, "type": string, "columnar": bool, "spec": object}],
}

Example usage:

# extract content and save to a certain file
> extract_datacard_content.py datacard.txt -o data.json

positional arguments:
  DATACARD              the datacard to read

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        location of the json output file; default:
                        DATACARD.json
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        extract_datacard_content
```


### Split a datacard by bins

```shell hl_lines="1"
> split_datacard_by_bins.py --help

usage: split_datacard_by_bins.py [-h] [--pattern PATTERN]
                                 [--save-bin-names SAVE_BIN_NAMES]
                                 [--directory [DIRECTORY]] [--no-shapes]
                                 [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                                 DATACARD

Script to split a datacard into all its bins with one output datacard per bin.
Example usage:

# split a datacard into bins, place new datacards in the same directory
> split_datacard_by_bins.py datacard.txt

# split a datacard into bins, placing new datacards in a new directory
> split_datacard_by_bins.py datacard.txt -d output_directory

positional arguments:
  DATACARD              the datacard to read and split

optional arguments:
  -h, --help            show this help message and exit
  --pattern PATTERN, -p PATTERN
                        pattern of names of created datacards where '{}' is
                        replaced with the bin name; default: DATACARD_{}.txt
  --save-bin-names SAVE_BIN_NAMES, -s SAVE_BIN_NAMES
                        location of a json file where the original bin names
                        are stored
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the datacard and its shape files
                        are first bundled into, and where split datacards are
                        saved
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        split_datacard_by_bins
```


### Bundle a datacard

```shell hl_lines="1"
> bundle_datacard.py --help

usage: bundle_datacard.py [-h] [--shapes-directory SHAPES_DIRECTORY]
                          [--log-level LOG_LEVEL] [--log-name LOG_NAME]
                          DATACARD DIRECTORY

Script to bundle a datacard, i.e., the card itself and the shape files it
contains are copied to a target directory. Example usage:

# bundle a single datacard
> bundle_datacard.py datacard.txt some_directory

# bundle multiple cards (note the quotes)
> bundle_datacard.py 'datacard_*.txt' some_directory

# bundle a single datacard and move shapes into a subdirectory
> bundle_datacard.py -s shapes 'datacard_*.txt' some_directory

positional arguments:
  DATACARD              the datacard to bundle into the target directory;
                        supports patterns to bundle multiple datacards
  DIRECTORY             target directory

optional arguments:
  -h, --help            show this help message and exit
  --shapes-directory SHAPES_DIRECTORY, -s SHAPES_DIRECTORY
                        an optional subdirectory when shape files are bundled,
                        relative to the target directory; default: .
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        bundle_datacard
```


### Prettify a datacard

```shell hl_lines="1"
> prettify_datacard.py --help

usage: prettify_datacard.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--no-preamble] [--log-level LOG_LEVEL]
                            [--log-name LOG_NAME]
                            DATACARD

Script to prettify a datacard.
Example usage:

> prettify_datacard.py datacard.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.

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
  --log-name LOG_NAME   name of the logger on the command line; default:
                        prettify_datacard
```
