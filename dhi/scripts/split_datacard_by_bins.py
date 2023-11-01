#!/usr/bin/env python3
# coding: utf-8

"""
Script to split a datacard into all its bins with one output datacard per bin.
Example usage:

# split a datacard into bins, place new datacards in the same directory
> split_datacard_by_bins.py datacard.txt

# split a datacard into bins, placing new datacards in a new directory
> split_datacard_by_bins.py datacard.txt -d output_directory
"""

import os
import shutil
import json

from dhi.scripts import remove_bins
from dhi.datacard_tools import read_datacard_structured, bundle_datacard
from dhi.util import real_path, create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def split_datacard_by_bins(
    datacard,
    pattern=None,
    store_file=None,
    directory=None,
    skip_shapes=False,
):
    """
    Splits a *datacard* into all its bins and saves each bin in a new datacard using *pattern* to
    define their basenames. ``"{}"`` in the pattern is replaced with the corresponding bin name.

    By default, the new datacards are saved in the directory of *datacard*. When *directory* is set,
    the datacard, all the shape files it refers to, and newly created datacards are stored in the
    specified directory. For consistency, this will also update the location of shape files in the
    datacards themselves. When *skip_shapes* is *True*, all shape files remain unchanged (the shape
    lines in the datacard itself are still changed).

    When *store_file* is set, the original bin names are stored in this file (relative to the new
    datacards) in json format.
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # default pattern
    if pattern is None:
        parts = os.path.basename(datacard).split(".", 1) + ["txt"]
        pattern = "{}_{{}}.{}".format(*parts[:2])

    # get bin names
    struct = read_datacard_structured(datacard)
    bin_names = [b["name"] for b in struct["bins"]]

    # remove bins one by one
    for bin_name in bin_names:
        bin_datacard = os.path.join(os.path.dirname(datacard), pattern.format(bin_name))
        logger.info("removing all bins but {} in datacard {}".format(bin_name, bin_datacard))

        # copy the original card to run the removal in place
        if os.path.exists(bin_datacard):
            os.remove(bin_datacard)
        shutil.copy2(datacard, bin_datacard)

        # actual removal
        with patch_object(remove_bins, "logger", logger):
            with patch_object(logger, "name", "{} (bin {})".format(logger.name, bin_name)):
                remove_bins.remove_bins(bin_datacard, ["!" + bin_name])

    # save the bin names
    if store_file:
        store_file = os.path.join(os.path.dirname(datacard), store_file)
        with open(store_file, "w") as f:
            json.dump(bin_names, f, indent=4)


if __name__ == "__main__":
    import argparse

    default_directory = os.getenv("DHI_DATACARD_SCRIPT_DIRECTORY") or script_name

    # setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        metavar="DATACARD",
        help="the datacard to read and split",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default=None,
        help="pattern defining the names of created datacards where '{}' is replaced with the bin "
        "name; default: DATACARD_{}.txt",
    )
    parser.add_argument(
        "--save-bin-names",
        "-s",
        metavar="PATH",
        help="location of a json file in which original bin names are stored",
    )
    parser.add_argument(
        "--directory",
        "-d",
        nargs="?",
        default=default_directory,
        help="directory in which the updated datacard and shape files are stored; when empty or "
        "'none', the input files are changed in-place; default: '{}'".format(default_directory),
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
        split_datacard_by_bins(
            args.input,
            pattern=args.pattern,
            store_file=args.save_bin_names,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
        )
