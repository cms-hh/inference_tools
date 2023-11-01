#!/usr/bin/env python3
# coding: utf-8

"""
Script to split a "lnN" or "lnU" (nuisance) parameter into several ones,
depending on expressions matching relevant bin and process names. Example usage:

# split the "lumi" parameter depending on the year, encoded in the bin names
# (note the quotes)
> split_parameter.py datacard.txt lumi 'lumi_2017,*2017*,*' 'lumi_2018,*2018*,*'

# split the "pdf" parameter depending on the process name (note the quotes)
> split_parameter.py datacard.txt pdf 'pdf_ttbar,*,TT' 'pdf_st,*,ST'

# split the "pdf" parameter depending on the process name with pattern negation
# (note the quotes)
> split_parameter.py datacard.txt pdf 'pdf_ttbar,*,TT' 'pdf_rest,*,!TT'

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
from collections import OrderedDict

from dhi.datacard_tools import (
    columnar_parameter_directives, bundle_datacard, manipulate_datacard, update_datacard_count,
)
from dhi.util import real_path, multi_match, create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def split_parameter(
    datacard,
    param_name,
    specs,
    ensure_unique=False,
    ensure_all=False,
    directory=None,
    skip_shapes=False,
):
    """
    Reads a *datacard* and splits a "lnN" or "lnU" parameter *param_name* into multiple new
    parameters of the same type configured by *specs*. A spec is a 3-tuple or a string in the format
    "NEW_NAME,BIN,PROCESS". Bin and process names can be patterns which, when matching, control if
    a parameter value should be assigned to the corresponding new parameter for that bin and
    process. Patterns starting with "!" negative their meaning. When *ensure_unique* is *True*, a
    check is performed ensuring that each parameter value is assigned to not more than one new
    parameter. Similarly, when *ensure_all* is *True*, a check is performed ensuring that each
    parameter value is assigned to at least one new parameter, in order to avoid loosing values.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # store supported types, which must always be a subset of all columnar types
    supported_types = ["lnN", "lnU"]
    assert all(multi_match(t, columnar_parameter_directives) for t in supported_types)

    # prepare the datacard path
    datacard = real_path(datacard)

    # validate specs
    _specs = OrderedDict()
    for s in specs:
        s = tuple(s) if isinstance(s, (tuple, list)) else tuple(s.split(","))
        if len(s) != 3:
            raise Exception("specification {} must have length 3".format(s))
        new_name, bin_pattern, process_pattern = s
        _specs.setdefault(new_name, []).append((bin_pattern, process_pattern))
    specs = _specs

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start splitting
    with manipulate_datacard(datacard) as blocks:
        if not blocks.get("parameters"):
            return

        # lookup the parameter line to split
        for line_idx, param_line in enumerate(blocks["parameters"]):
            param_line = param_line.split()

            # none of the new names should exist already
            if param_line and param_line[0] in specs:
                raise Exception("a parameter named {} already exists in the datacard".format(
                    param_line[0],
                ))

            # our parameter?
            if param_line[0] != param_name:
                continue

            # cannot process with less than two line elements
            if len(param_line) < 2:
                raise Exception("invalid parameter line: " + blocks["parameters"][line_idx])

            # check the type
            param_type = param_line[1]
            if not multi_match(param_type, supported_types):
                raise Exception("parameter type '{}' is not supported".format(param_type))

            param_values = param_line[2:]
            logger.info("found parameter {} to split with type {}".format(param_name, param_type))
            break
        else:
            raise Exception("parameter {} does not exist in datacard".format(param_name))

        # get bins and processes
        bin_names = blocks["rates"][0].split()[1:]
        process_names = blocks["rates"][1].split()[1:]
        if len(bin_names) != len(process_names):
            raise Exception(
                "number of bins ({}) and processes ({}) not matching in datacard rates".format(
                    len(bin_names), len(process_names),
                ))
        if len(bin_names) != len(param_values):
            raise Exception(
                "number of bins and processes ({}) not matching values of parameter {} ({})".format(
                    len(bin_names), param_name, len(param_values),
                ))

        # create new lines per spec
        spec_lines = []
        for new_name, patterns in specs.items():
            spec_values = []
            for bin_name, process_name, value in zip(bin_names, process_names, param_values):
                spec_value = "-"
                for bin_pattern, process_pattern in patterns:
                    # check the bin name pattern which may start with a negating "!"
                    neg = bin_pattern.startswith("!")
                    if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                        continue

                    # check the process name pattern which may start with a negating "!"
                    neg = process_pattern.startswith("!")
                    if multi_match(process_name, process_pattern[int(neg):]) == neg:
                        continue

                    # store the value
                    spec_value = value
                    logger.debug(
                        "spec of new parameter {} matched bin {} and process {}, assigning "
                        "value {}".format(new_name, bin_name, process_name, value),
                    )
                    break

                spec_values.append(spec_value)
            spec_lines.append([new_name, param_type] + spec_values)

        # uniqueness check
        groups = list(zip(*(spec_line[2:] for spec_line in spec_lines)))
        if ensure_unique:
            failed_idxs = [
                i for i, group in enumerate(groups)
                if group.count("-") < len(group) - 1
            ]
            if failed_idxs:
                f = failed_idxs[0]
                g = groups[f]
                msg = (
                    "uniqueness check failed in {} bin-process pairs, first error in\n"
                    "bin {} and process {} with {} matched parameters:\n"
                ).format(len(failed_idxs), bin_names[f], process_names[f], len(g) - g.count("-"))
                for i, v in enumerate(groups[f]):
                    msg += "  {}: {}\n".format(spec_lines[i][0], v)
                raise Exception(msg)

        # all check
        if ensure_all:
            failed_idxs = [
                i for i, group in enumerate(groups)
                if group.count("-") == len(group) and param_values[i] != "-"
            ]
            if failed_idxs:
                msg = (
                    "check ensuring that all values were assigned failed in {} bin-process pairs:\n"
                ).format(len(failed_idxs))
                for f in failed_idxs:
                    msg += "  bin {}, process {}\n".format(bin_names[f], process_names[f])
                raise Exception(msg)

        # remove the old line
        lines = [line for i, line in enumerate(blocks["parameters"]) if i != line_idx]
        del blocks["parameters"][:]
        blocks["parameters"].extend(lines)
        logger.info("removed parameter {} with type {} and {} values".format(
            param_name, param_type, len(param_values) - param_values.count("-")))

        # add the new lines
        for new_name, spec_line in zip(specs, spec_lines):
            blocks["parameters"].append(" ".join(spec_line))
            logger.info("added new parameter {} with type {} and {} values".format(
                new_name, param_type, len(spec_line) - spec_line.count("-") - 2,
            ))

        # update kmax in counts
        update_datacard_count(blocks, "kmax", len(spec_lines) - 1, diff=True, logger=logger)


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
        help="the datacard to read and possibly update (see --directory)",
    )
    parser.add_argument(
        "param",
        metavar="PARAM_NAME",
        help="name of the parameter to split",
    )
    parser.add_argument(
        "specs",
        nargs="+",
        metavar="NEW_NAME,BIN,PROCESS",
        help="specification of new parameters, each in the format 'NEW_NAME,BIN,PROCESS'; "
        "supports patterns; prepending '!' to a pattern negates its meaning",
    )
    parser.add_argument(
        "--ensure-unique",
        "-u",
        action="store_true",
        help="when set, a check is performed to ensure that each value is assigned to not more "
        "than one new parameter",
    )
    parser.add_argument(
        "--ensure-all",
        "-a",
        action="store_true",
        help="when set, a check is performed to ensure that each value is assigned to at least one "
        "new parameter",
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

    # run the splitting
    with patch_object(logger, "name", args.log_name):
        split_parameter(
            args.input,
            args.param,
            args.specs,
            ensure_unique=args.ensure_unique,
            ensure_all=args.ensure_all,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
        )
