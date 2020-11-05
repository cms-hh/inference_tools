#!/usr/bin/env python
# coding: utf-8

"""
Script to remove one or multiple processes from a datacard.
Example usage:

# remove certain processes
> remove_processes.py datacard.txt qqHH_CV_1_C2V_2_kl_1 -d output_directory

# remove processes via fnmatch wildcards (note the quotes)
> remove_processes.py datacard.txt "qqHH_CV_1_C2V_*_kl_1" -d output_directory

# remove processes listed in a file
> remove_processes.py datacard.txt processes.txt -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.
"""

import os
import re

from dhi.datacard_tools import (
    columnar_parameter_directives, ShapeLine, bundle_datacard, manipulate_datacard,
)
from dhi.util import real_path, multi_match, create_console_logger


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_processes(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes processes given by a list of *patterns*. A pattern can be a
    parameter name, a pattern that is matched via fnmatch, or a file containing patterns. When
    *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the changed
    datacard and all the shape files it refers to are stored in the specified directory. For
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
        if not os.path.exists(path):
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
        # remove from process rates and remember column indices for removal in parameters
        removed_columns = []
        if content.get("rates"):
            bin_names = content["rates"][0].split()[1:]
            process_names = content["rates"][1].split()[1:]
            process_ids = content["rates"][2].split()[1:]
            rates = content["rates"][3].split()[1:]

            for i, process_name in enumerate(process_names):
                if multi_match(process_name, patterns):
                    logger.info("remove process {} from rates in bin {}".format(
                        process_name, bin_names[i]))
                    removed_columns.append(i)

            mask = lambda l: [elem for j, elem in enumerate(l) if j not in removed_columns]
            content["rates"][0] = "bin " + " ".join(mask(bin_names))
            content["rates"][1] = "process " + " ".join(mask(process_names))
            content["rates"][2] = "process " + " ".join(mask(process_ids))
            content["rates"][3] = "rate " + " ".join(mask(rates))

        # remove from certain parameters
        if content.get("parameters") and removed_columns:
            expr = r"^([^\s]+)\s+({})\s+(.+)$".format("|".join(columnar_parameter_directives))
            for i, param_line in enumerate(list(content["parameters"])):
                m = re.match(expr, param_line.split("#")[0].strip())
                if not m:
                    continue

                # split the line
                param_name = m.group(1)
                param_type = m.group(2)
                columns = m.group(3).split()
                if max(removed_columns) >= len(columns):
                    raise Exception("parameter line {} '{} {} ...' has less columns than defined "
                        "in rates".format(i, param_name, param_name))

                # remove columns and update the line
                logger.info("remove {} column(s) from parameter {}".format(
                    len(removed_columns), param_name))
                columns = [c for j, c in enumerate(columns) if j not in removed_columns]
                content["parameters"][i] = " ".join([param_name, param_type] + columns)

        # remove from shape lines
        if content.get("shapes"):
            shape_lines = [ShapeLine(line, j) for j, line in enumerate(content["shapes"])]
            to_remove = []
            for shape_line in shape_lines:
                if shape_line.process != "*" and multi_match(shape_line.process, patterns):
                    logger.info("remove shape line for process {} and bin {}".format(
                        shape_line.process, shape_line.bin))
                    to_remove.append((shape_line.i))

            # change lines in-place
            lines = [line for j, line in enumerate(content["shapes"]) if j not in to_remove]
            del content["shapes"][:]
            content["shapes"].extend(lines)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="the datacard to read and possibly update (see --directory)")
    parser.add_argument("names", nargs="+", help="names of processes or files containing "
        "process names to remove line by line; supports patterns")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level)

    # run the removing
    remove_processes(args.input, args.names, directory=args.directory, skip_shapes=args.no_shapes)
