#!/usr/bin/env python
# coding: utf-8

"""
Script to remove one or multiple bins from a datacard.
Example usage:

# remove certain bins
> remove_bins.py datacard.txt ch1 -d output_directory

# remove bins via fnmatch wildcards (note the quotes)
> remove_bins.py datacard.txt "ch*" -d output_directory

# remove bins listed in a file
> remove_bins.py datacard.txt bins.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.
"""

import os

from dhi.scripts import remove_bin_process_pairs
from dhi.datacard_tools import expand_file_lines
from dhi.util import create_console_logger, patch_object


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_bins(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes bins given by a list of *patterns*. A pattern can be a bin name,
    a pattern that is matched via fnmatch, or a file containing patterns.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # expand patterns from files
    patterns = expand_file_lines(patterns)

    # add a process wildcard to all of them
    pairs = [(p, "*") for p in patterns]

    # just call remove_bin_process_pairs with our own logger
    with patch_object(remove_bin_process_pairs, "logger", logger):
        remove_bin_process_pairs.remove_bin_process_pairs(datacard, pairs, directory=directory,
            skip_shapes=skip_shapes)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="the datacard to read and possibly update (see --directory)")
    parser.add_argument("names", nargs="+", help="names of bins or files containing bin names to "
        "remove line by line; supports patterns")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level)

    # run the removing
    remove_bins(args.input, args.names, directory=args.directory, skip_shapes=args.no_shapes)