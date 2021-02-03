#!/usr/bin/env python
# coding: utf-8

"""
Script to correct yield decreasing effects of lnN nuisances from
"1-u" to "1/(1+u)" syntax. Example usage:

# fix a certain parameter
> fix_lnN_parameter.py datacard.txt lumi_13TeV -d output_directory

# fix multiple parameters (note the quotes)
> fix_lnN_parameter.py datacard.txt "lumi_*" -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import bundle_datacard, manipulate_datacard, expand_file_lines
from dhi.util import real_path, multi_match, create_console_logger


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def fix_lnN_parameter(datacard, patterns, directory=None, skip_shapes=False, digits=4):
    """
    Reads a *datacard* and fixes the syntax of "lnN" parameters given by a list of *patterns*, such
    that uncertainties ``u`` decreasing the yield are no longer encoded as ``1-u``, but rather as
    ``1/(1+u)`` with a certain amount of significant *digits*. A pattern can be a parameter name, a
    pattern that is matched via fnmatch, or a file containing patterns.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # expand patterns from files
    patterns = expand_file_lines(patterns)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # helper to fix decreasing values from "1-u" to "1/(1+u)" syntax
    fix_lnN_effect = lambda v: "{{:.{}f}}".format(digits).format(v if v >= 1 else (1. / (2 - v)))

    # start fixing
    with manipulate_datacard(datacard) as content:
        if content.get("parameters"):
            for i, param_line in enumerate(list(content["parameters"])):
                param_line = param_line.split()

                # cannot process with less than two line elements
                if len(param_line) < 2:
                    continue

                # only select lnN nuisances whose name matches any pattern
                param_name = param_line[0]
                if not multi_match(param_name, patterns) or param_line[1] != "lnN":
                    continue

                # build a new line
                new_param_line = list(param_line[:2])

                # correct values one by one
                for v in param_line[2:]:
                    v_new = v
                    if "/" in v:
                        d, u = v.split("/", 1)
                        d = fix_lnN_effect(float(d))
                        u = fix_lnN_effect(float(u))
                        v_new = "{}/{}".format(d, u)
                    elif v != "-":
                        v_new = str(fix_lnN_effect(float(v)))
                    if v_new != v:
                        logger.debug("changed effect of lnN parameter {} from {} to {}".format(
                            param_name, v, v_new))
                    new_param_line.append(v_new)

                # save the new line
                content["parameters"][i] = " ".join(new_param_line)
                logger.info("fixed lnN effect of parameter {}".format(param_name))


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("names", nargs="+", metavar="NAME", help="names of lnN parameters or files "
        "containing lnN parameter names to fix line by line; supports patterns")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--digits", type=int, default=4, help="the amount of digits for rounding "
        "converted effects; defaults to 4")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # fix the parameters
    fix_lnN_parameter(args.input, args.names, directory=args.directory, skip_shapes=args.no_shapes,
        digits=args.digits)
