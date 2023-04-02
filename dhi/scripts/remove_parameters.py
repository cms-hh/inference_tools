#!/usr/bin/env python3
# coding: utf-8

"""
Script to remove one or multiple (nuisance) parameters from a datacard.
Example usage:

# remove certain parameters
> remove_parameters.py datacard.txt CMS_btag_JES CMS_btag_JER -d output_directory

# remove parameters via fnmatch wildcards
# (note the quotes)
> remove_parameters.py datacard.txt 'CMS_btag_JE?' -d output_directory

# remove a parameter from all processes in a certain bin
# (note the quotes)
> remove_parameters.py datacard.txt 'OS_2018,*,CMS_btag_JES' -d output_directory

# remove a parameter from a certain processes in all bins
# (note the quotes)
> remove_parameters.py datacard.txt '*,tt,CMS_btag_JES' -d output_directory

# remove parameters listed in a file
> remove_parameters.py datacard.txt parameters.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
import re

from dhi.datacard_tools import (
    columnar_parameter_directives, bundle_datacard, manipulate_datacard, update_datacard_count,
    expand_file_lines, drop_datacard_lines,
)
from dhi.util import real_path, multi_match, create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def remove_parameters(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes parameters given by a list of *patterns*. A pattern can be a
    parameter pattern that is matched via fnmatch, a 3-tuple defining a certain bin, process and
    parameter combination, or a file containing these values line by line. When a bin and process
    pattern are present, they are negated when they start with a '!'.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # expand patterns from files and parse
    _patterns = expand_file_lines(patterns)
    patterns = []
    for pattern in _patterns:
        _pattern = tuple(pattern if isinstance(pattern, (list, tuple)) else pattern.split(","))
        if len(_pattern) == 1:
            _pattern = ("*", "*", _pattern[0])
        elif len(_pattern) != 3:
            raise ValueError("invalid parameter removal pattern '{}'".format(pattern))
        patterns.append(_pattern)

    # get parameter patterns which full bin and process wildcards as they are the only ones
    # considered for removing parameters other than columnar ones and rateParam's
    # only consider the parameter patterns with bin and process wildcards
    single_patterns = [
        param_pattern for bin_pattern, proc_pattern, param_pattern in patterns
        if (bin_pattern, proc_pattern) == ("*", "*")
    ]

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start removing
    with manipulate_datacard(datacard) as blocks:
        # remove from parameters, keep track of parameters that were fully removed
        removed_nuisances = []
        if blocks.get("parameters"):
            # get bin and process name pairs for removing the proper columns of columnar parameters
            bin_names = blocks["rates"][0].split()[1:]
            process_names = blocks["rates"][1].split()[1:]
            process_ids = blocks["rates"][2].split()[1:]
            rates = blocks["rates"][3].split()[1:]

            # check if all lists have the same lengths
            if not (len(bin_names) == len(process_names) == len(process_ids) == len(rates)):
                raise Exception(
                    "the number of bin names ({}), process names ({}), process ids ({}) "
                    "and rates ({}) does not match".format(
                        len(bin_names), len(process_names), len(process_ids), len(rates),
                    ),
                )

            # go through parameter lines
            # remember those to remove and updated columnar parameter lines
            to_remove = []
            for i, param_line in enumerate(list(blocks["parameters"])):
                param_line = param_line.split()
                if len(param_line) < 2:
                    continue

                param_name, param_type = param_line[:2]
                if multi_match(param_type, columnar_parameter_directives):
                    # store updated effects
                    effects = []
                    for bin_name, proc_name, f in zip(bin_names, process_names, param_line[2:]):
                        # when the effect is missing, do nothing
                        if f in ["-", "0", "0.0"]:
                            effects.append("-")
                            continue

                        # compare with all patterns
                        for bin_pattern, proc_pattern, param_pattern in patterns:
                            neg = bin_pattern.startswith("!")
                            if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                                continue

                            neg = proc_pattern.startswith("!")
                            if multi_match(proc_name, proc_pattern[int(neg):]) == neg:
                                continue

                            neg = param_pattern.startswith("!")
                            if multi_match(param_name, param_pattern[int(neg):]) == neg:
                                continue

                            effects.append("-")
                            logger.debug(
                                "remove effect {} from {} parameter {} in bin {} and "
                                "process {}".format(f, param_type, param_name, bin_name, proc_name),
                            )
                            break
                        else:
                            effects.append(f)

                    # when there is no effect left, remove it completely, otherwise update
                    if effects.count("-") == len(effects):
                        to_remove.append(i)
                        removed_nuisances.append(param_name)
                        logger.info("no effect left, remove {} parameter {}".format(
                            param_type, param_name,
                        ))
                    else:
                        blocks["parameters"][i] = " ".join([param_name, param_type] + effects)

                elif param_type == "rateParam":
                    # special treatment of rate parameters
                    if len(param_line) < 4:
                        continue
                    bin_name, proc_name = param_line[2:4]

                    # compare with all patterns
                    for bin_pattern, proc_pattern, param_pattern in patterns:
                        # special cases with patterns in the datacards
                        if bin_name == "*" and bin_pattern != "*":
                            continue
                        if proc_name == "*" and proc_pattern != "*":
                            continue

                        neg = bin_pattern.startswith("!")
                        if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                            continue

                        neg = proc_pattern.startswith("!")
                        if multi_match(proc_name, proc_pattern[int(neg):]) == neg:
                            continue

                        neg = param_pattern.startswith("!")
                        if multi_match(param_name, param_pattern[int(neg):]) == neg:
                            continue

                        to_remove.append(i)
                        logger.info("remove {} parameter {} in bin {} and process {}".format(
                            param_type, param_name, bin_name, proc_name,
                        ))
                        break

                else:
                    # general parameter without binding to a bin or process
                    # remove when a single pattern matches
                    if multi_match(param_name, single_patterns):
                        logger.info("remove {} parameter {}".format(param_type, param_name))
                        to_remove.append(i)

            # change lines in-place
            drop_datacard_lines(blocks, "parameters", to_remove)

        # remove from group listings
        if blocks.get("groups"):
            for i, group_line in enumerate(list(blocks["groups"])):
                m = re.match(r"^([^\s]+)\s+group\s+\=\s+(.+)$", group_line.strip())
                if not m:
                    logger.error("invalid group line format: {}".format(group_line))
                    continue
                group_name = m.group(1)
                param_names = m.group(2).split()
                for param_name in list(param_names):
                    if param_name not in param_name:
                        continue
                    if multi_match(param_name, removed_nuisances + single_patterns):
                        logger.info("remove parameter {} from group {}".format(
                            param_name, group_name,
                        ))
                        param_names.remove(param_name)
                group_line = "{} group = {}".format(group_name, " ".join(param_names))
                blocks["groups"][i] = group_line

        # remove groups themselves
        if blocks.get("groups"):
            to_remove = []
            for i, group_line in enumerate(blocks["groups"]):
                group_name = group_line.split()[0]
                if multi_match(group_name, single_patterns):
                    logger.info("remove group {}".format(group_name))
                    to_remove.append(i)

            # change lines in-place
            drop_datacard_lines(blocks, "groups", to_remove)

        # remove auto mc stats
        if blocks.get("auto_mc_stats"):
            to_remove = []
            for i, line in enumerate(blocks["auto_mc_stats"]):
                # the bin name is the actual parameter name, so compare it with the single patterns
                bin_name = line.strip().split()[0]
                if bin_name != "*" and multi_match(bin_name, single_patterns):
                    to_remove.append(i)
                    logger.info("remove autoMCStats for bin {}".format(bin_name))

            # change lines in-place
            drop_datacard_lines(blocks, "auto_mc_stats", to_remove)

        # remove certain nuisance edit lines
        if blocks.get("nuisance_edits"):
            to_remove = []
            for i, edit_line in enumerate(blocks["nuisance_edits"]):
                edit_line = edit_line.split()
                if len(edit_line) < 4 or tuple(edit_line[:2]) != ("nuisance", "edit"):
                    continue
                action = edit_line[2]

                if action == "rename":
                    if len(edit_line) == 5 and multi_match(edit_line[4], single_patterns):
                        to_remove.append(i)
                    elif len(edit_line) >= 7 and multi_match(edit_line[6], single_patterns):
                        logger.warning(
                            "removing 'nuisance edit add' lines with process and bin "
                            "options is not yet supported",
                        )
                elif action in ["freeze", "changepdf"]:
                    if multi_match(edit_line[3], single_patterns):
                        to_remove.append(i)
                elif action in ["add", "drop"]:
                    if len(edit_line) >= 6 and multi_match(edit_line[5], single_patterns):
                        to_remove.append(i)
                if to_remove and to_remove[-1] == i:
                    logger.debug("remove nuisance edit line '{}'".format(" ".join(edit_line)))

            # change lines in-place
            drop_datacard_lines(blocks, "nuisance_edits", to_remove)

        # decrease kmax in counts
        if removed_nuisances:
            update_datacard_count(
                blocks,
                "kmax",
                -len(removed_nuisances),
                diff=True,
                logger=logger,
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
        metavar="SPEC",
        help="specifications of parameters to remove or a file containing these specifications "
        "line by line; a specification should have the format '[BIN,PROCESS,]PARAMETER'; when a "
        "bin name and process are defined, the parameter should be of a type that is defined on a "
        "bin and process basis, and is removed only in this bin process combination; all values "
        "support patterns; prepending '!' to a bin or process pattern negates its meaning",
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
        remove_parameters(
            args.input,
            args.names,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
        )
