In addition to [scripts that manipulate datacards](datacard_manipulation.md), a collection of further useful scripts is located in [dhi/scripts](https://gitlab.cern.ch/hh/tools/inference/-/tree/master/dhi/scripts). They are documented below.


## Printing statistics tables

### Yields from datacards

Currently, quoted uncertainties only contain statistical errors due to the limited amount of MC statistics.

```shell hl_lines="1"
> yield_table.py --help

usage: yield_table.py [-h] [--bins BINS [BINS ...]] [--log-level LOG_LEVEL]
                      [--table-fmt TABLE_FMT] [--precision PRECISION]
                      [--data-name DATA_NAME]
                      DATACARD

Script to get yield tables from a datacard.
Example usage:

> yield_table.py datacard.txt --table-fmt latex

positional arguments:
  DATACARD              the datacard to read

optional arguments:
  -h, --help            show this help message and exit
  --bins BINS [BINS ...]
                        Regex to group bins, supports multiple expressions.
                        Default: '.+'.
  --log-level LOG_LEVEL, -l LOG_LEVEL
                        python log level; default: INFO
  --log-name LOG_NAME   name of the logger on the command line; default:
                        yield_table
  --table-fmt TABLE_FMT
                        Table format. Default: 'github'.
  --precision PRECISION
                        Decimal precision. Default: '2'.
  --data-name DATA_NAME
                        Name of observation. Default: 'data_obs'.
```


## Generate JSON for interactive covariance viewer

Generate a [interactive covariance viewer](view_cov_json.html)-compatiable JSON file.

```shell hl_lines="1"
> extract_fitresult_cov.json.py --help
usage: extract_fitresult_cov.json.py [-h] [-s SKIP] filename [filename ...]

Produces a .cov.json for every `RooFitResult` within the ROOT file given by
`filename`. This includes covariance and correlation Matrix, column/row
labels, and covariance quality. These files can be explored via the
`view_cov_json.html` viewer. The output filenames are automatically produced
by joining `filename` without extension (i.e. ".root"), the name of the
`RooFitResult`, and the "cov.json" extension using the ".". Example:
folder/input_file.root -> folder/input_file.result_name.cov.json If `skip`
(default "prefit") is given and not empty, all `RooFitResult`s with their name
containing `skip`are skipped. Multiple `filename`s can be given
simultaneously.

positional arguments:
  filename

optional arguments:
  -h, --help            show this help message and exit
  -s SKIP, --skip SKIP  skip results containing this non-empty string,
                        default: prefit
```