#!/usr/bin/env python
# coding: utf-8

"""
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
"""

import os

from dhi.scripts import remove_bin_process_pairs
from dhi.datacard_tools import expand_file_lines
from dhi.util import create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def remove_processes(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes processes given by a list of *patterns*. A pattern can be a
    parameter name, a pattern that is matched via fnmatch, or a file containing patterns.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # expand patterns from files
    patterns = expand_file_lines(patterns)

    # add a bin wildcard to all of them
    pairs = [("*", p) for p in patterns]

    # just call remove_bin_process_pairs with our own logger
    with patch_object(remove_bin_process_pairs, "logger", logger):
        remove_bin_process_pairs.remove_bin_process_pairs(
            datacard,
            pairs,
            directory=directory,
            skip_shapes=skip_shapes,
        )


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        metavar="DATACARD",
        help="the datacard to read and possibly update (see --directory)",
    )
    parser.add_argument(
        "names",
        nargs="+",
        metavar="NAME",
        help="names of processes or files containing process names to remove line by line; "
        "supports patterns; prepending '!' to a pattern negates its meaning",
    )
    parser.add_argument(
        "--directory",
        "-d",
        nargs="?",
        default=script_name,
        help="directory in which the updated datacard and shape files are stored; when empty or "
        "'none', the input files are changed in-place; default: '{}'".format(script_name),
    )
    parser.add_argument(
        "--no-shapes",
        "-n",
        action="store_true",
        help="do not copy shape files to the output directory when --directory is set",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        help="python log level; default: INFO",
    )
    parser.add_argument(
        "--log-name",
        default=logger.name,
        help="name of the logger on the command line; default: {}".format(logger.name),
    )
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the removing
    with patch_object(logger, "name", args.log_name):
        remove_processes(
            args.input,
            args.names,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
        )
