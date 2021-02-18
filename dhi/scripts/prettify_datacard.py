#!/usr/bin/env python
# coding: utf-8

"""
Script to prettify a datacard.
Example usage:

> prettify_datacard.py datacard.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import bundle_datacard, read_datacard_blocks, write_datacard_pretty
from dhi.util import real_path, create_console_logger, patch_object


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def prettify_datacard(datacard, directory=None, skip_shapes=False, skip_preamble=False):
    """
    Adds a new parameter with *param_name* and *param_type* to a *datacard*. When *param_spec* is
    given, it should be a list configuring the arguments to be placed behind the parameter
    definition. For columnar parameters, each value should be a 3-tuple in the format (bin_name,
    process_name, effect). Patterns are supported and evaluated in the given order for all bin
    process pairs. For all other parameter types, the values are added unchanged to the new
    parameter line.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).

    When *skip_preamble* is *True*, any existing preamble before the count block is removed.
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # read the datacard content in blocks
    blocks = read_datacard_blocks(datacard)

    # open the datacard and write prettified content
    with open(datacard, "w") as f:
        logger.info("write prettified datacard {}".format(datacard))
        write_datacard_pretty(f, blocks, skip_fields=["preamble"] if skip_preamble else None)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--no-preamble", action="store_true", help="remove any existing preamble")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # add the parameter
    with patch_object(logger, "name", args.log_name):
        prettify_datacard(args.input, directory=args.directory, skip_shapes=args.no_shapes,
            skip_preamble=args.no_preamble)
