#!/usr/bin/env python
# coding: utf-8

"""
Script to flip the effect of one or multiple (nuisance) parameters in a
datacard. Currently, only parameters with columnar type "lnN", "lnU" and "shape"
are supported. Example usage:

# flip via simple names
> flip_parameters.py datacard.txt alpha_s_ttH alpha_s_tt -d output_directory

# flip via name patterns (note the quotes)
> flip_parameters.py datacard.txt 'alpha_s_*' -d output_directory

# flip via bin, process and name patterns (note the quotes)
> flip_parameters.py datacard.txt '*,ch1,alpha_s_*' -d output_directory

# flip via rules in files
> flip_parameters.py datacard.txt my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import (
    DatacardRenamer, ShapeLine, update_shape_name, expand_variables, expand_file_lines,
)
from dhi.util import create_console_logger, patch_object, multi_match


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def flip_parameters(datacard, patterns, directory=None, skip_shapes=False, mass="125"):
    """
    Reads a *datacard* and flips the effect of parameters given by *patterns*. A pattern should
    match the name(s) of parameters to flip, or, when in the format ``"BIN,PROCESS,NAME"``, also
    select the bin and process in which to perform the flipping. Here, wildcard patterns are
    supported as well, with a leading ``"!"`` negating its meaning.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    When *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # expand patterns from files
    patterns = expand_file_lines(patterns)

    # check pattern formats
    _patterns = []
    for pattern in patterns:
        p = tuple(pattern) if isinstance(pattern, (tuple, list)) else tuple(pattern.split(","))
        if len(p) == 1:
            p = ("*", "*") + p
        elif len(p) != 3:
            raise Exception("patterns must have the format '[BIN,PROCESS,]NAME', got '{}'".format(
                pattern))
        _patterns.append(p)
    patterns = _patterns

    # create a DatacardRenamer to work with
    renamer = DatacardRenamer(datacard, directory=directory, skip_shapes=skip_shapes, logger=logger)

    # start renaming
    with renamer.start() as blocks:
        # get the lists of bin and process names from the "rates" block
        bin_names = blocks["rates"][0].split()[1:]
        process_names = blocks["rates"][1].split()[1:]
        if len(bin_names) != len(process_names):
            raise Exception("number of bins ({}) and processes ({}) not matching in datacard "
                "rates".format(len(bin_names), len(process_names)))

        # iterate through lines in the "parameters" block
        for i, param_line in enumerate(blocks.get("parameters", [])):
            param_line = param_line.split()
            if len(param_line) < 2:
                continue
            param_name, param_type = param_line[:2]

            # stop when the type cannot be flipped
            if not multi_match(param_type, ["lnN", "lnU", "shape*"]):
                continue

            # stop when no name pattern will match
            if not multi_match(param_name, [p for _, _, p in patterns]):
                continue

            # get the effects
            effects = param_line[2:]
            if len(effects) != len(bin_names):
                raise Exception("number of effects of parameter {} ({}) does not match number of "
                    "bins and processes ({})".format(param_name, len(effects), len(bin_names)))

            # check patterns for each bin-process combination
            new_effects = list(effects)
            for j, (bin_name, process_name, f) in enumerate(zip(bin_names, process_names, effects)):
                for bin_pattern, process_pattern, name_pattern in patterns:
                    neg = bin_pattern.startswith("!")
                    if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                        continue
                    neg = process_pattern.startswith("!")
                    if multi_match(process_name, process_pattern[int(neg):]) == neg:
                        continue
                    if not multi_match(param_name, name_pattern):
                        continue

                    # do the flipping
                    if param_type == "lnN" or param_type == "lnU":
                        if "/" not in f:
                            continue

                        f = "{1}/{0}".format(*f.split("/", 1))
                        logger.debug("flip effect of {} parameter {} in bin {} and process {} "
                            "to {}".format(param_type, param_name, bin_name, process_name, f))

                    elif multi_match(param_type, "shape*"):
                        if f == "-" or skip_shapes or not blocks.get("shapes"):
                            continue

                        # extract shape lines, sort them so that most specific ones (no wildcards)
                        # come first
                        shape_lines = [ShapeLine(l, k) for k, l in enumerate(blocks["shapes"])]
                        shape_lines.sort(key=lambda sl: sl.sorting_weight)

                        # find the first shape line that matches bin and process
                        for sl in shape_lines:
                            if sl.is_fake or not sl.syst_pattern:
                                continue
                            if not multi_match(bin_name, sl.bin):
                                continue
                            if not multi_match(process_name, sl.process):
                                continue

                            # get the syst pattern and find the owning object
                            # when it contains a ":", the preceeding string denotes the name of an
                            # owner object (e.g. a workspace or a tree) which should not be renamed
                            syst_pattern = sl.syst_pattern
                            src_path = os.path.join(os.path.dirname(renamer.datacard), sl.file)
                            tfile = renamer.open_tfile(src_path, "UPDATE")
                            towner = tfile
                            if ":" in syst_pattern:
                                towner_name, _syst_pattern = syst_pattern.split(":", 1)
                                towner = renamer.get_tobj(tfile, towner_name, "UPDATE")
                                if not towner:
                                    raise Exception("could not find object {} in {} with pattern "
                                        "{}".format(towner_name, tfile, syst_pattern))
                                syst_pattern = _syst_pattern

                            # expand variables to get shape name
                            tobj_name = expand_variables(syst_pattern, systematic=param_name,
                                channel=bin_name, process=process_name, mass=mass)

                            # roll names
                            update_shape_name(towner, tobj_name + "Up", tobj_name + "UpTMP")
                            update_shape_name(towner, tobj_name + "Down", tobj_name + "Up")
                            update_shape_name(towner, tobj_name + "UpTMP", tobj_name + "Down")
                            logger.debug("flip effect of {} parameter {} in bin {} and process {} "
                                "in shape file {} (entry '{}{{Up,Down}}')".format(param_type,
                                param_name, bin_name, process_name, sl.file, tobj_name))
                            break
                        else:
                            logger.warning("cannot find shape line for bin {} and process {} "
                                "to flip {} parameter {}".format(bin_name, process_name,
                                param_type, param_name))
                            continue

                    # store it and stop processing patterns
                    new_effects[j] = str(f)
                    break

            # replace the line
            param_line = " ".join([param_name, param_type] + new_effects)
            logger.info("adding new parameter line '{}'".format(param_line))
            blocks["parameters"][i] = param_line


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("names", nargs="+", metavar="NAME", help="names of parameters whose effect "
        "should be flipped in the format '[BIN,PROCESS,]PARAMETER'; when a bin and process names "
        "are given, the effect is only flipped in those; patterns are supported; prepending '!' to "
        "a pattern negates its meaning; a name can also refer to a file with names in the above "
        "format line by line")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not change parameter "
        "names in shape files")
    parser.add_argument("--mass", "-m", default="125", help="mass hypothesis; default: 125")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the renaming
    with patch_object(logger, "name", args.log_name):
        flip_parameters(args.input, args.names, directory=args.directory,
            skip_shapes=args.no_shapes, mass=args.mass)
