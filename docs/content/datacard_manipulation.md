There are a couple of scripts located in [dhi/scripts](https://gitlab.cern.ch/hh/tools/inference/-/tree/master/dhi/scripts) that allow you to manipulate datacards from the command line.

All scripts have the ability to move, or *bundle* a datacard and all the shape files it refers to into a new directory.
When doing so, the locations of shape files are changed consistently within the datacard.
The argument to configure this directory is identical across all scripts.

```shell
script_name.py DATACARD [OTHER_ARGUMENTS] --directory/-d DIRECTORY
```

Please note that, when no output directory is given, ==the content of datacards and shape files is changed in-place==.


## Remove parameters

```shell hl_lines="1"
> remove_parameters.py --help

usage: remove_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--log-level LOG_LEVEL]
                            input names [names ...]

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
  input                 the datacard to read and possibly update (see
                        --directory)
  names                 names of parameters or files containing parameter
                        names to remove; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        pythonic log level
```


## Rename parameters

```shell hl_lines="1"
> rename_parameters.py --help

usage: rename_parameters.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                            [--mass MASS] [--log-level LOG_LEVEL]
                            input OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple (nuisance) parameters in a datacard.
Example usage:

# rename via simple rules
> rename_parameters.py datacard.txt btag_JES=CMS_btag_JES -d output_directory

# rename via rules in files
> rename_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  input                 the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple parameter names
                        in the format 'old_name=new_name', or files containing
                        these rules in the same format

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not change parameter names in shape files
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        pythonic log level
```


## Remove processes

```shell hl_lines="1"
> remove_processes.py --help

usage: remove_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--log-level LOG_LEVEL]
                           input names [names ...]

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
  input                 the datacard to read and possibly update (see
                        --directory)
  names                 names of processes or files containing process names
                        to remove; supports patterns

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not copy shape files to the output directory when
                        --directory is set
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        pythonic log level
```


## Rename processes

```shell hl_lines="1"
> rename_processes.py --help

usage: rename_processes.py [-h] [--directory [DIRECTORY]] [--no-shapes]
                           [--mass MASS] [--log-level LOG_LEVEL]
                           input OLD_NAME=NEW_NAME [OLD_NAME=NEW_NAME ...]

Script to rename one or multiple processes in a datacard.
Example usage:

# rename via simple rules
> rename_processes.py datacard.txt ggH_process=ggHH_kl_1_kt_1 -d output_directory

# rename via rules in files
> rename_processes.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.

positional arguments:
  input                 the datacard to read and possibly update (see
                        --directory)
  OLD_NAME=NEW_NAME     translation rules for one or multiple process names in
                        the format 'old_name=new_name', or files containing
                        these rules in the same format

optional arguments:
  -h, --help            show this help message and exit
  --directory [DIRECTORY], -d [DIRECTORY]
                        directory in which the updated datacard and shape
                        files are stored; when not set, the input files are
                        changed in-place
  --no-shapes, -n       do not change process names in shape files
  --mass MASS, -m MASS  mass hypothesis; default: 125
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        pythonic log level
```
