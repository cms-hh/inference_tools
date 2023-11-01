#!/usr/bin/env python3
# coding: utf-8

r"""
Script to rename one or multiple processes in a datacard.
Example usage:

# rename via simple rules
> rename_processes.py datacard.txt ggH_process=ggHH_kl_1_kt_1 -d output_directory

# rename multiple processes using a replacement rule
# (note the quotes)
> rename_processes.py datacard.txt '^ggH_process_(.+)$=ggHH_kl_1_kt_1_\1' -d output_directory

# rename via rules in files
> rename_processes.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import DatacardRenamer, ShapeLine, update_shape_name, expand_variables
from dhi.util import create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def rename_processes(datacard, rules, directory=None, skip_shapes=False, mass="125"):
    """
    Reads a *datacard* and renames processes according to translation *rules*. A rule should be a
    sequence of length 2 containing the old and the new process name. When the old name starts with
    "^" and ends with "$", it is interpreted as a regular expression and the new name can contain
    group placeholders as understood by *re.sub()*.

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
    # as well as the new pattern with translation rules applied when possiblethe and the containing
    # object in case the pattern has the format "obj_name:shape_pattern"
    def parse_shape_pattern(shape_line, tfile, process_name, bin_name, syst_name=None):
        # get the new process name
        assert renamer.has_rule(process_name)
        new_process_name = renamer.translate(process_name)

        # get the pattern
        old_pattern = shape_line.syst_pattern if syst_name else shape_line.nom_pattern

        # when it contains a ":", the preceeding string denotes the name of a containing object
        # (e.g. a workspace or a tree) which should not be renamed
        towner = tfile
        towner_name = None
        if ":" in old_pattern:
            towner_name, _old_pattern = old_pattern.split(":", 1)
            if not skip_shapes:
                towner = renamer.get_tobj(tfile, towner_name, "UPDATE")
                if not towner:
                    raise Exception(
                        "could not find object {} in {} with pattern {}".format(
                            towner_name, tfile, old_pattern,
                        ),
                    )
            old_pattern = _old_pattern

        # try to update the pattern when the shape line's process is specific
        new_pattern = old_pattern
        if shape_line.process != "*" and process_name in old_pattern:
            new_pattern = old_pattern.replace(process_name, new_process_name)

        # expand variables to get the old and new shape names
        variables = {"channel": bin_name, "mass": mass}
        if syst_name:
            variables["systematic"] = syst_name
        old_name = expand_variables(old_pattern, process=process_name, **variables)
        new_name = expand_variables(new_pattern, process=new_process_name, **variables)

        # add the owner name back to the new pattern when given
        if towner_name:
            new_pattern = "{}:{}".format(towner_name, new_pattern)

        return old_name, new_name, new_pattern, towner

    # start renaming
    with renamer.start(expand="processes") as blocks:
        # rename names in process rates
        if blocks.get("rates"):
            line = blocks["rates"][1] + " "
            for old_name, new_name in renamer.rules.items():
                if (" " + old_name + " ") in line[len("process"):]:
                    logger.info("rename process {} to {}".format(old_name, new_name))
                    line = line.replace(" " + old_name + " ", " " + new_name + " ")
            blocks["rates"][1] = line.strip()

        # rename shapes
        if blocks.get("shapes"):
            # determine shape systematic names per (bin, process) pair
            shape_syst_names = renamer.get_bin_process_to_systs_mapping()

            # keep track of shapes yet to be updated and sort them by length of the process name in
            # decreasing order to rename longer names first
            unhandled_shapes = renamer.get_bin_process_pairs()
            unhandled_shapes = [(b, p) for b, p in unhandled_shapes if renamer.has_rule(p)]
            unhandled_shapes.sort(key=lambda tpl: -len(tpl[1]))

            # extract shape lines and sort them so that most specific ones
            # (i.e. without wildcards) come first
            shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
            shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

            # go through shape lines and do the renaming
            for shape_line in shape_lines:
                # copy the shape line to do track updates
                new_shape_line = shape_line.copy()

                # work on a temporary copy of the shape file
                tfile = None
                if not skip_shapes and not shape_line.is_fake:
                    src_path = os.path.join(os.path.dirname(renamer.datacard), shape_line.file)
                    tfile = renamer.open_tfile(src_path, "UPDATE")

                # loop through processes and bins to be handled and see if the current line applies
                for bin_name, process_name in list(unhandled_shapes):
                    if shape_line.bin not in (bin_name, "*"):
                        continue
                    if shape_line.process not in (process_name, "*"):
                        continue
                    unhandled_shapes.remove((bin_name, process_name))
                    process_is_wildcard = shape_line.process != process_name

                    # rename the process when not a wildcard
                    if not process_is_wildcard:
                        new_shape_line.process = renamer.translate(process_name)

                    # handle shape and pattern renaming below only when this is not a fake shape
                    if shape_line.is_fake:
                        continue

                    # get the expanded old and new shape names, the updated shape pattern
                    # and the owning object
                    old_name, new_name, new_pattern, towner = parse_shape_pattern(shape_line, tfile,
                        process_name, bin_name)

                    # update the shape name
                    if not skip_shapes:
                        logger.info(
                            "renaming shape {} to {} for process {} in bin {}".format(
                                old_name, new_name, process_name, bin_name,
                            ),
                        )
                        clone = update_shape_name(towner, old_name, new_name)
                        renamer._tfile_cache.write_tobj(tfile, clone, towner)

                    # update the pattern in the shape line
                    if not process_is_wildcard:
                        new_shape_line.nom_pattern = new_pattern

                    # same for all systematic shapes when a syst_pattern is given
                    syst_names = shape_syst_names.get((bin_name, process_name))
                    if shape_line.syst_pattern and syst_names:
                        for syst_name in syst_names:
                            for syst_dir in ["Up", "Down"]:
                                old_name, new_name, new_pattern, towner = parse_shape_pattern(
                                    shape_line, tfile, process_name, bin_name, syst_name + syst_dir)
                                if not skip_shapes:
                                    logger.debug(
                                        "renaming syst shape {} to {} for process {} in "
                                        "bin {}".format(old_name, new_name, process_name, bin_name),
                                    )
                                    clone = update_shape_name(towner, old_name, new_name)
                                    renamer._tfile_cache.write_tobj(tfile, clone, towner)
                                if not process_is_wildcard:
                                    new_shape_line.syst_pattern = new_pattern

                # add the new line back to blocks
                blocks["shapes"][new_shape_line.i] = str(new_shape_line)


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
        "rules",
        nargs="+",
        metavar="OLD_NAME=NEW_NAME",
        help="translation rules for one or multiple process names in the format "
        "'OLD_NAME=NEW_NAME', or files containing these rules in the same format line by line; "
        "OLD_NAME can be a regular expression starting with '^' and ending with '$'; in this case, "
        "group placeholders in NEW_NAME are replaced with the proper matches as described in "
        "re.sub()",
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
        help="do not change process names in shape files",
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
        rename_processes(
            args.input,
            args.rules,
            directory=None if args.directory.lower() in ["", "none"] else args.directory,
            skip_shapes=args.no_shapes,
            mass=args.mass,
        )
