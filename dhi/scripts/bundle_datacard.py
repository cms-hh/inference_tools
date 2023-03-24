#!/usr/bin/env python
# coding: utf-8

"""
Script to bundle a datacard, i.e., the card itself and the shape files it
contains are copied to a target directory. Example usage:

# bundle a single datacard
> bundle_datacard.py datacard.txt some_directory

# bundle multiple cards (note the quotes)
> bundle_datacard.py 'datacard_*.txt' some_directory

# bundle a single datacard and move shapes into a subdirectory
> bundle_datacard.py -s shapes 'datacard_*.txt' some_directory
"""

import os
import glob

from dhi.datacard_tools import bundle_datacard
from dhi.util import create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


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
        help="the datacard to bundle into the target directory; supports patterns to bundle "
        "multiple datacards",
    )
    parser.add_argument(
        "directory",
        metavar="DIRECTORY",
        help="target directory",
    )
    parser.add_argument(
        "--shapes-directory",
        "-s",
        metavar="SHAPES_DIRECTORY",
        default=".",
        help="an optional subdirectory when shape files are bundled, relative to the target "
        "directory; default: .",
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

    # run the bundling
    with patch_object(logger, "name", args.log_name):
        pattern = os.path.expandvars(os.path.expanduser(args.input))
        for datacard in glob.glob(pattern):
            dst = bundle_datacard(datacard, args.directory, shapes_directory=args.shapes_directory)
            logger.info("bundled datacard {} to {}".format(datacard, dst))
