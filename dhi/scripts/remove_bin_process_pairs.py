#!/usr/bin/env python
# coding: utf-8

"""
Script to remove one or multiple bin process pairs from a datacard.
Example usage:

# remove a certain bin process pair
> remove_bin_process_pairs.py datacard.txt ch1,ttZ -d output_directory

# remove all processes for a specific bin via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt 'ch1,*' -d output_directory

# remove all bins for a specific process via wildcards (note the quotes)
> remove_bin_process_pairs.py datacard.txt '*,ttZ' -d output_directory

# remove bin process pairs listed in a file
> remove_bin_process_pairs.py datacard.txt pairs.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import (
    columnar_parameter_directives, ShapeLine, bundle_datacard, manipulate_datacard,
    expand_file_lines, update_datacard_count, drop_datacard_lines,
)
from dhi.util import real_path, multi_match, create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def remove_bin_process_pairs(datacard, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and removes bin process pairs given by a list of *patterns*. A pattern can be
    2-tuple or comma-separated string describing the bin and process patterns, or a file containing
    these patterns.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # expand patterns from files and convert to tuples
    _patterns = []
    for pattern in expand_file_lines(patterns):
        if not isinstance(pattern, (tuple, list)):
            pattern = [p.strip() for p in pattern.split(",")]
        if len(pattern) != 2:
            raise Exception("pattern {} must have length 2".format(pattern))
        _patterns.append(pattern)
    patterns = _patterns

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start removing
    with manipulate_datacard(datacard) as blocks:
        # remove from process rates and remember column indices for removal in parameters
        removed_columns = []
        bin_names = blocks["rates"][0].split()[1:]
        process_names = blocks["rates"][1].split()[1:]
        process_ids = blocks["rates"][2].split()[1:]
        rates = blocks["rates"][3].split()[1:]

        # quick check if all lists have the same lengths
        if not (len(bin_names) == len(process_names) == len(process_ids) == len(rates)):
            raise Exception(
                "the number of bin names ({}), process names ({}), process ids ({}) and rates ({}) "
                "does not match".format(
                    len(bin_names), len(process_names), len(process_ids), len(rates),
                ),
            )

        # go through bin and process names and compare with patterns
        for i, (bin_name, process_name) in enumerate(zip(bin_names, process_names)):
            for bin_pattern, process_pattern in patterns:
                neg = bin_pattern.startswith("!")
                if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                    continue

                neg = process_pattern.startswith("!")
                if multi_match(process_name, process_pattern[int(neg):]) == neg:
                    continue

                logger.debug("remove process {} from rates in bin {}".format(
                    process_name, bin_name,
                ))
                removed_columns.append(i)
                break

        # remove hits
        mask = lambda l: [elem for j, elem in enumerate(l) if j not in removed_columns]
        new_bin_names = mask(bin_names)
        new_process_names = mask(process_names)
        new_process_ids = mask(process_ids)
        new_rates = mask(rates)

        # check if certain bins or processes were removed entirely
        fully_removed_bin_names = set(bin_names) - set(new_bin_names)
        fully_removed_process_names = set(process_names) - set(new_process_names)

        # add back reduced lines
        blocks["rates"][0] = "bin " + " ".join(new_bin_names)
        blocks["rates"][1] = "process " + " ".join(new_process_names)
        blocks["rates"][2] = "process " + " ".join(new_process_ids)
        blocks["rates"][3] = "rate " + " ".join(new_rates)
        logger.info("removed {} entries from process rates".format(len(removed_columns)))

        # decrease imax in counts
        if fully_removed_bin_names:
            logger.info("removed all occurrences of bin(s) {}".format(
                ", ".join(fully_removed_bin_names),
            ))
            update_datacard_count(
                blocks,
                "imax",
                -len(fully_removed_bin_names),
                diff=True,
                logger=logger,
            )

        # decrease jmax in counts
        if fully_removed_process_names:
            logger.info("removed all occurrences of processes(s) {}".format(
                ", ".join(fully_removed_process_names),
            ))
            update_datacard_count(
                blocks,
                "jmax",
                -len(fully_removed_process_names),
                diff=True,
                logger=logger,
            )

        # remove fully removed bins from observations
        if blocks.get("observations") and fully_removed_bin_names:
            bin_names = blocks["observations"][0].split()[1:]
            observations = blocks["observations"][1].split()[1:]

            removed_obs_columns = []
            for i, bin_name in enumerate(bin_names):
                if bin_name in fully_removed_bin_names:
                    logger.info("remove bin {} from observations".format(bin_name))
                    removed_obs_columns.append(i)

            mask = lambda l: [elem for j, elem in enumerate(l) if j not in removed_obs_columns]
            blocks["observations"][0] = "bin " + " ".join(mask(bin_names))
            blocks["observations"][1] = "observation " + " ".join(mask(observations))

        # remove from shape lines
        if blocks.get("shapes"):
            shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
            to_remove = []
            for shape_line in shape_lines:
                # when both bin and process are wildcards, the shape line is not removed
                if shape_line.bin == "*" and shape_line.process == "*":
                    continue

                # when only the bin is a wildcard, the shape line is not removed when the process is
                # not fully removed
                if shape_line.bin == "*" and shape_line.process not in fully_removed_process_names:
                    continue

                # when only the process is a wildcard, the shape line is not removed when the bin is
                # not fully removed
                if shape_line.process == "*" and shape_line.bin not in fully_removed_bin_names:
                    continue

                # in any other case, compare with patterns
                for bin_pattern, process_pattern in patterns:
                    neg = bin_pattern.startswith("!")
                    if multi_match(shape_line.bin, bin_pattern[int(neg):]) == neg:
                        continue

                    neg = process_pattern.startswith("!")
                    if multi_match(shape_line.process, process_pattern[int(neg):]) == neg:
                        continue

                    logger.debug("remove shape line for process {} and bin {}".format(
                        shape_line.process, shape_line.bin,
                    ))
                    to_remove.append((shape_line.i))
                    break

            # change lines in-place
            drop_datacard_lines(blocks, "shapes", to_remove)

        # remove certain parameters
        if blocks.get("parameters"):
            # columnar parameters
            if removed_columns:
                for i, param_line in enumerate(list(blocks["parameters"])):
                    param_line = param_line.split()
                    if len(param_line) < 3:
                        continue

                    # split the line
                    param_name, param_type = param_line[:2]
                    if not multi_match(param_type, columnar_parameter_directives):
                        continue

                    columns = param_line[2:]
                    if max(removed_columns) >= len(columns):
                        raise Exception(
                            "parameter line {} '{} {} ...' has less columns than "
                            "defined in rates".format(i, param_name, param_type),
                        )

                    # remove columns and update the line
                    logger.debug("remove {} column(s) from {} parameter {} with {} columns".format(
                        len(removed_columns), param_type, param_name, len(columns),
                    ))
                    columns = [c for j, c in enumerate(columns) if j not in removed_columns]
                    blocks["parameters"][i] = " ".join([param_name, param_type] + columns)

            # other parameters
            to_remove = []
            for i, param_line in enumerate(blocks["parameters"]):
                param_line = param_line.split()
                if len(param_line) < 2:
                    continue
                param_name, param_type = param_line[:2]

                # check rateParam's
                if param_type == "rateParam" and len(param_line) >= 4:
                    bin_pattern, proc_pattern = param_line[2:4]
                    for bin_name, proc_name in zip(new_bin_names, new_process_names):
                        if (
                            multi_match(bin_name, bin_pattern) and
                            multi_match(proc_name, proc_pattern)
                        ):
                            break
                    else:
                        to_remove.append(i)
                        logger.debug("remove '{}' with no matching bin or process left".format(
                            " ".join(param_line),
                        ))

            # change lines in-place
            drop_datacard_lines(blocks, "parameters", to_remove)

        # remove fully removed bins from auto mc stats
        if blocks.get("auto_mc_stats") and fully_removed_bin_names:
            to_remove = []
            for i, line in enumerate(blocks["auto_mc_stats"]):
                bin_name = line.strip().split()[0]
                if bin_name in fully_removed_bin_names:
                    to_remove.append(i)
                    logger.info("remove autoMCStats for bin {}".format(bin_name))

            # change lines in place
            drop_datacard_lines(blocks, "auto_mc_stats", to_remove)

        # remove certain nuisance edit lines
        if blocks.get("nuisance_edits"):
            to_remove = []
            for i, edit_line in enumerate(blocks["nuisance_edits"]):
                edit_line = edit_line.split()
                if len(edit_line) < 5 or tuple(edit_line[:2]) != ("nuisance", "edit"):
                    continue
                action, proc_pattern, bin_pattern = edit_line[2:5]

                if action in ["add", "drop", "split", "merge"]:
                    for bin_name, proc_name in zip(new_bin_names, new_process_names):
                        if (
                            multi_match(bin_name, bin_pattern) and
                            multi_match(proc_name, proc_pattern)
                        ):
                            break
                    else:
                        to_remove.append(i)
                        logger.debug(
                            "remove nuisance edit action {} in bin {} and process {} with "
                            "no matching bin or process left".format(action, bin_name, proc_name),
                        )

            # change lines in-place
            drop_datacard_lines(blocks, "nuisance_edits", to_remove)


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
        metavar="BIN_NAME,PROCESS_NAME",
        help="names of bin process pairs to remove in the format 'bin_name,process_name' or files "
        "containing these pairs line by line; supports patterns; prepending '!' to a pattern "
        "negates its meaning",
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
        remove_bin_process_pairs(
            args.input,
            args.names,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
        )
