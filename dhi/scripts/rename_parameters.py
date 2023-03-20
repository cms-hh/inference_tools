#!/usr/bin/env python
# coding: utf-8

r"""
Script to rename one or multiple (nuisance) parameters in a datacard.
Example usage:

# rename via simple rules
> rename_parameters.py datacard.txt btag_JES=CMS_btag_JES -d output_directory

# rename multiple parameters using a replacement rule
# (note the quotes)
> rename_parameters.py datacard.txt '^parameter_(.+)$=param_\1' -d output_directory

# rename via rules in files
> rename_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
import re

from dhi.datacard_tools import DatacardRenamer, ShapeLine, update_shape_name, expand_variables
from dhi.util import create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def rename_parameters(datacard, rules, directory=None, skip_shapes=False, mass="125"):
    """
    Reads a *datacard* and renames parameters according to translation *rules*. A rule should be a
    sequence of length 2 containing the old and the new parameter name. When the old name starts
    with "^" and ends with "$", it is interpreted as a regular expression and the new name can
    contain group placeholders as understood by *re.sub()*.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    When *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # create a DatacardRenamer to work with
    renamer = DatacardRenamer(
        datacard,
        rules,
        directory=directory,
        skip_shapes=skip_shapes,
        logger=logger,
    )

    # helper that determines the old and new name of a shape object given a combine pattern
    # as well as the containing object in case the pattern has the format "obj_name:shape_pattern"
    def parse_shape_pattern(shape_line, tfile, process_name, bin_name, syst_name, syst_dir):
        # get the new systematic name
        assert not skip_shapes
        assert renamer.has_rule(syst_name)
        new_syst_name = renamer.translate(syst_name)

        # get the pattern
        pattern = shape_line.syst_pattern

        # when it contains a ":", the preceeding string denotes the name of an owner object
        # (e.g. a workspace or a tree) which should not be renamed
        towner = tfile
        towner_name = None
        if ":" in pattern:
            towner_name, _pattern = pattern.split(":", 1)
            towner = renamer.get_tobj(tfile, towner_name, "UPDATE")
            if not towner:
                raise Exception("could not find object {} in {} with pattern {}".format(
                    towner_name, tfile, pattern),
                )
            pattern = _pattern

        # expand variables to get the old and new shape names
        variables = {"channel": bin_name, "process": process_name, "mass": mass}
        old_name = expand_variables(pattern, systematic=syst_name + syst_dir, **variables)
        new_name = expand_variables(pattern, systematic=new_syst_name + syst_dir, **variables)

        return old_name, new_name, towner

    # start renaming
    with renamer.start(expand="parameters") as blocks:
        # rename parameter names in the "parameters" block itself
        if blocks.get("parameters"):
            def sub_fn(match):
                old_name, rest = match.groups()
                new_name = renamer.translate(old_name)
                logger.info("rename parameter {} to {}".format(old_name, new_name))
                return " ".join([new_name, rest])

            for i, param_line in enumerate(list(blocks["parameters"])):
                parts = param_line.split()
                old_name = parts[0]
                if renamer.has_rule(old_name):
                    expr = r"^({})\s(.*)$".format(old_name)
                    param_line = re.sub(expr, sub_fn, param_line)
                    blocks["parameters"][i] = param_line

                    # for extArg and rateParam, change the name of the var in the linked workspace
                    expr1 = r".*\s+extArg\s+([^\s]+\.root)\:([^\s]+).*$"
                    expr2 = r".*\s+rateParam\s+[^\s]+\s+[^\s]+\s+([^\s]+\.root)\:([^\s]+).*$"
                    m = re.match(expr1, param_line) or re.match(expr2, param_line)
                    if m:
                        ws_file, ws_name = m.groups()
                        ws_file = os.path.join(os.path.dirname(renamer.datacard), ws_file)
                        ws = renamer.get_tobj(ws_file, ws_name, "UPDATE")
                        if not ws:
                            raise Exception(
                                "workspace {} not found in file {} referenced by extArg {}".format(
                                    ws_name, ws_file, old_name,
                                ),
                            )
                        new_name = renamer.translate(old_name)
                        targ = ws.arg(old_name)
                        if targ:
                            targ.SetName(new_name)
                            targ.SetTitle(targ.GetTitle().replace(old_name, new_name))
                        else:
                            logger.warning("extArg {} not found in workspace {} in tfile {}".format(
                                old_name, ws_name, ws_file,
                            ))

        # update them in group listings
        if blocks.get("groups"):
            def sub_fn(match):
                start, old_name, end = match.groups()
                new_name = renamer.translate(old_name)
                logger.info("rename parameter {} in group {} to {}".format(old_name,
                    start.split()[0], new_name))
                return " ".join([start, new_name, end]).strip()

            for i, group_line in enumerate(list(blocks["groups"])):
                for old_name in renamer.rules:
                    expr = r"^(.+\s+group\s+=.*)\s({})\s(.*)$".format(old_name)
                    group_line = re.sub(expr, sub_fn, group_line + " ")
                blocks["groups"][i] = group_line

        # update group names themselves
        if blocks.get("groups"):
            def sub_fn(match):
                old_name, rest = match.groups()
                new_name = renamer.translate(old_name)
                logger.info("rename group {} to {}".format(old_name, new_name))
                return " ".join([new_name, rest])

            for i, group_line in enumerate(list(blocks["groups"])):
                group_name = group_line.split()[0]
                if renamer.has_rule(group_name):
                    expr = r"^({})\s(.*)$".format(group_name)
                    group_line = re.sub(expr, sub_fn, group_line)
                    blocks["groups"][i] = group_line

        # update them in nuisace edit lines
        if blocks.get("nuisance_edits"):
            def sub_fn(match):
                start, old_name, end = match.groups()
                new_name = renamer.translate(old_name)
                logger.info("rename nuisance {} in nuisance edit line '{}' to {}".format(old_name,
                    start.split()[2], new_name,
                ))
                return " ".join([start, new_name, end]).strip()

            for i, edit_line in enumerate(list(blocks["nuisance_edits"])):
                # check if the action is supported and build a regexp for sub
                parts = edit_line.split()
                if len(parts) >= 6 and parts[2] in ["add", "drop"]:
                    expr = r"^(nuisance\s+edit\s+{}\s+[^\s]+\s+[^\s]+)\s+({})\s+(.*)$"
                elif len(parts) >= 4 and parts[2] in ["changepdf", "freeze"]:
                    expr = r"^(nuisance\s+edit\s+{})\s+({})\s+(.*)$"
                else:
                    continue

                for old_name in renamer.rules:
                    _expr = expr.format(parts[2], old_name)
                    edit_line = re.sub(_expr, sub_fn, edit_line + " ")
                blocks["nuisance_edits"][i] = edit_line

        # rename shapes
        if not skip_shapes and blocks.get("shapes"):
            # determine shape systematic names per (bin, process) pair
            shape_syst_names = renamer.get_bin_process_to_systs_mapping()

            # keep track of shapes yet to be updated
            unhandled_shapes = renamer.get_bin_process_pairs()

            # extract shape lines, sort them so that most specific ones (no wildcards) come first
            shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
            shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

            # go through shape lines and do the renaming
            for shape_line in shape_lines:
                # skip fake shape lines and those without a systematic pattern
                if shape_line.is_fake or not shape_line.syst_pattern:
                    continue

                # work on a temporary copy of the shape file
                src_path = os.path.join(os.path.dirname(renamer.datacard), shape_line.file)
                tfile = renamer.open_tfile(src_path, "UPDATE")

                # loop through processes and bins to be handled and see if the current line applies
                for bin_name, process_name in list(unhandled_shapes):
                    if shape_line.bin not in (bin_name, "*"):
                        continue
                    if shape_line.process not in (process_name, "*"):
                        continue
                    unhandled_shapes.remove((bin_name, process_name))

                    # the bin process pair should have shape systematics to be changed
                    syst_names = shape_syst_names.get((bin_name, process_name), [])
                    syst_names = filter(renamer.has_rule, syst_names)
                    if not syst_names:
                        continue

                    # loop through all systematic shapes
                    for syst_name in syst_names:
                        for syst_dir in ["Up", "Down"]:
                            # get the expanded old and new shape names and the owning object
                            old_name, new_name, towner = parse_shape_pattern(shape_line, tfile,
                                process_name, bin_name, syst_name, syst_dir)

                            # update the shape name
                            logger.debug(
                                "renaming syst shape {} to {} for process {} in bin {}".format(
                                    old_name, new_name, process_name, bin_name,
                                ),
                            )
                            update_shape_name(towner, old_name, new_name)


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
        "rules",
        nargs="+",
        metavar="OLD_NAME=NEW_NAME",
        help="translation rules for one or multiple parameter names in the format "
        "'OLD_NAME=NEW_NAME', or files containing these rules in the same format line by line; "
        "OLD_NAME can be a regular expression starting with '^' and ending with '$'; in this case, "
        "group placeholders in NEW_NAME are replaced with the proper matches as described in "
        "re.sub()",
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
        help="do not change parameter names in shape files",
    )
    parser.add_argument(
        "--mass",
        "-m",
        default="125",
        help="mass hypothesis; default: 125",
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

    # run the renaming
    with patch_object(logger, "name", args.log_name):
        rename_parameters(
            args.input,
            args.rules,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
            mass=args.mass,
        )
