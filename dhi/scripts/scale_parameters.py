#!/usr/bin/env python
# coding: utf-8

"""
Script to scale the effect of one or multiple (nuisance) parameters in a
datacard. Currently, only parameters with columnar type "lnN" and "lnU" are
supported. Example usage:

# scale by 0.5 via simple names
> scale_parameters.py datacard.txt 0.5 alpha_s_ttH alpha_s_tt -d output_directory

# flip by 0.5 via name patterns (note the quotes)
> scale_parameters.py datacard.txt 0.5 'alpha_s_*' -d output_directory

# flip by 0.5 via bin, process and name patterns (note the quotes)
> scale_parameters.py datacard.txt 0.5 '*,ch1,alpha_s_*' -d output_directory

# scale by 0.5 via rules in files
> scale_parameters.py datacard.txt 0.5 my_rules.txt -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os

from dhi.datacard_tools import bundle_datacard, manipulate_datacard, expand_file_lines
from dhi.util import create_console_logger, patch_object, multi_match


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def scale_parameters(datacard, factor, patterns, directory=None, skip_shapes=False):
    """
    Reads a *datacard* and scales the effect of parameters given by *patterns* with *factor*. A
    pattern should match the name(s) of parameters to scale, or, when in the format
    ``"BIN,PROCESS,NAME"``, also select the bin and process in which to perform the flipping. Here,
    wildcard patterns are supported as well, with a leading ``"!"`` negating its meaning.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
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

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # helper to scale "e" in a value encoded as "1 +- e" by factor
    def scale_effect(f):
        f = float(f)
        return 1.0 + factor * (f - 1.0)

    # start removing
    with manipulate_datacard(datacard) as blocks:
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

            # stop if the type cannot be simply scaled
            if not multi_match(param_type, ["lnN", "lnU"]):
                continue

            # stop if no name pattern will match
            if not multi_match(param_name, [p for _, _, p in patterns]):
                continue

            # get the effects
            effects = param_line[2:]
            if len(effects) != len(bin_names):
                raise Exception("number of effects of parameter {} ({}) does not match number of "
                    "bins and processes ({})".format(param_name, len(effects), len(bin_names)))

            # check patterns for each bin-process combination
            new_effects = list(effects)
            n_changes = 0
            for j, (bin_name, process_name, f) in enumerate(zip(bin_names, process_names, effects)):
                f_orig = f

                # skip empty effects
                if f == "-":
                    continue

                for bin_pattern, process_pattern, name_pattern in patterns:
                    neg = bin_pattern.startswith("!")
                    if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                        continue
                    neg = process_pattern.startswith("!")
                    if multi_match(process_name, process_pattern[int(neg):]) == neg:
                        continue
                    if not multi_match(param_name, name_pattern):
                        continue

                    # apply the scaling
                    if "/" in f:
                        f = "{}/{}".format(*map(scale_effect, f.split("/", 1)))
                    else:
                        f = scale_effect(f)

                    # store it and stop processing patterns
                    new_effects[j] = str(f)
                    n_changes += 1
                    logger.debug("scaled effect of {} parameter {} in bin {} and process {} "
                        "from {} to {}".format(param_type, param_name, bin_name, process_name,
                        f_orig, f))
                    break

            # replace the line
            if n_changes > 0:
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
    parser.add_argument("factor", type=float, help="factor by which parameters are scaled")
    parser.add_argument("names", nargs="+", metavar="NAME", help="names of parameters whose effect "
        "should be scaled in the format '[BIN,PROCESS,]PARAMETER'; when a bin and process names "
        "are given, the effect is only scaled in those; patterns are supported; prepending '!' to "
        "a pattern negates its meaning; a name can also refer to a file with names in the above "
        "format line by line")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the scaling
    with patch_object(logger, "name", args.log_name):
        scale_parameters(args.input, args.factor, args.names, directory=args.directory)
