#!/usr/bin/env python
# coding: utf-8

"""
Script to remove one or multiple (nuisance) parameters from a datacard.
Example usage:

# remove certain parameters
> remove_parameters.py datacard.txt CMS_btag_JES CMS_btag_JER -d output_directory

# remove parameters via fnmatch wildcards (note the quotes)
> remove_parameters.py datacard.txt "CMS_btag_JE?" -d output_directory

# remove parameters listed in a file
> remove_parameters.py datacard.txt parameters.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.
"""

import os
import re

from dhi.datacard_tools import bundle_datacard, manipulate_datacard
from dhi.util import real_path, multi_match, create_console_logger


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_parameters(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes parameters given by a list of *patterns*. A pattern can be a
    parameter name, a pattern that is matched via fnmatch, or a file containing patterns.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # expand patterns from files
    _patterns = []
    for pattern_or_path in patterns:
        # first try to interpret it as a file
        path = real_path(pattern_or_path)
        if not os.path.isfile(path):
            # not a file, use as is
            _patterns.append(pattern_or_path)
        else:
            # read the file line by line, accounting for empty lines and comments
            with open(path, "r") as f:
                for line in f.readlines():
                    pattern = line.split("#", 1)[0].strip()
                    if pattern:
                        _patterns.append(pattern)
    patterns = _patterns

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start removing
    with manipulate_datacard(datacard) as content:
        # remove from parameters
        if content.get("parameters"):
            to_remove = []
            for i, param_line in enumerate(content["parameters"]):
                param_name = param_line.split()[0]
                if multi_match(param_name, patterns):
                    logger.info("remove parameter {}".format(param_name))
                    to_remove.append(i)

            # change lines in-place
            lines = [line for i, line in enumerate(content["parameters"]) if i not in to_remove]
            del content["parameters"][:]
            content["parameters"].extend(lines)

        # remove from group listings
        if content.get("groups"):
            for i, group_line in enumerate(list(content["groups"])):
                m = re.match(r"^([^\s]+)\s+group\s+\=\s+(.+)$", group_line.split("#")[0].strip())
                if not m:
                    logger.error("invalid group line format: {}".format(group_line))
                    continue
                group_name = m.group(1)
                param_names = m.group(2).split()
                for param_name in list(param_names):
                    if param_name in param_names and multi_match(param_name, patterns):
                        logger.info("remove parameter {} in group {}".format(
                            param_name, group_name))
                        param_names.remove(param_name)
                group_line = "{} group = {}".format(group_name, " ".join(param_names))
                content["groups"][i] = group_line

        # remove group themselves
        if content.get("groups"):
            to_remove = []
            for i, group_line in enumerate(content["groups"]):
                group_name = group_line.split()[0]
                if multi_match(group_name, patterns):
                    logger.info("remove group {}".format(group_name))
                    to_remove.append(i)

            # change lines in-place
            lines = [line for j, line in enumerate(content["groups"]) if j not in to_remove]
            del content["groups"][:]
            content["groups"].extend(lines)

        # remove auto mc stats
        if content.get("auto_mc_stats"):
            to_remove = []
            for i, stats_line in enumerate(content["auto_mc_stats"]):
                bin_name = stats_line.split()[0]
                if bin_name != "*" and multi_match(bin_name, patterns):
                    logger.info("remove autoMCStats in bin {}".format(bin_name))
                    to_remove.append(i)

            # change lines in-place
            lines = [line for j, line in enumerate(content["auto_mc_stats"]) if j not in to_remove]
            del content["auto_mc_stats"][:]
            content["auto_mc_stats"].extend(lines)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="the datacard to read and possibly update (see --directory)")
    parser.add_argument("names", nargs="+", help="names of parameters or files containing "
        "parameter names to remove line by line; supports patterns")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level)

    # run the removing
    remove_parameters(args.input, args.names, directory=args.directory, skip_shapes=args.no_shapes)
