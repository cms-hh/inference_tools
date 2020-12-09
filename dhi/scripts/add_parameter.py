#!/usr/bin/env python
# coding: utf-8

"""
Script to add arbitrary parameters to the datacard.
Example usage:

# add auto MC stats
> add_parameter.py datacard.txt "*" autoMCStats 10 -d output_directory

# add a lnN nuisance for a specific process across all bins
> add_parameter.py datacard.txt new_nuisance lnN "*,ttZ,1.05" -d output_directory

# add a lnN nuisance for all processes in two specific bins
> add_parameter.py datacard.txt new_nuisance lnN "bin1,*,1.05" "bin2,*,1.07" -d output_directory

Note: The use of an output directory is recommended to keep input files unchanged.
"""

import os

from dhi.datacard_tools import (
    parameter_directives, columnar_parameter_directives, bundle_datacard, manipulate_datacard,
)
from dhi.util import real_path, multi_match, create_console_logger


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def add_parameter(datacard, param_name, param_type, param_spec=None, directory=None,
        skip_shapes=False):
    """
    Adds a new parameter with *param_name* and *param_type* to a *datacard*. When *param_spec* is
    given, it should be a list configuring the arguments to be placed behind the parameter
    definition. For columnar parameters, each value should be a 3-tuple in the format (bin_name,
    process_name, effect). Patterns are supported and evaluated in the given order for all bin
    process pairs. For all other parameter types, the values are added unchanged to the new
    parameter line.

    When *directory* is *None*, the input *datacard* is updated in-place. Otherwise, both the
    changed datacard and all the shape files it refers to are stored in the specified directory. For
    consistency, this will also update the location of shape files in the datacard. When
    *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # check if the param type is known
    if not multi_match(param_type, parameter_directives + ["group", "autoMCStats"]):
        raise Exception("unknown parameter type {}".format(param_type))

    # check and parse param spec
    is_columnar = multi_match(param_type, columnar_parameter_directives)
    if is_columnar:
        if not param_spec:
            raise Exception("a specification is required when adding columnar parameters")
        _param_spec = []
        for s in param_spec:
            if not isinstance(s, (tuple, list)):
                s = s.split(",")
            if len(s) != 3:
                raise Exception("specification {} must have length 3".format(s))
            _param_spec.append(s)
        param_spec = _param_spec

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start adding
    with manipulate_datacard(datacard) as content:
        # check if no parameter with that name already exists
        existing_names = []
        for key in ["parameters", "groups", "auto_mc_stats"]:
            for line in content.get(key):
                parts = line.strip().split()
                if len(parts) >= 2:
                    existing_names.append(parts[1])
        if param_name in existing_names:
            raise Exception("datacard already contains a parameter named {}".format(param_name))

        if not is_columnar:
            # when the parameter is not columnar, just add it
            param_line = " ".join(map(str, [param_name, param_type] + (param_spec or [])))
            if param_type == "group":
                key = "groups"
            elif param_type == "autoMCStats":
                key = "auto_mc_stats"
            else:
                key = "parameters"
            logger.info("adding new {} parameter line '{}'".format(key, param_line))
            content[key].append(param_line)

        else:
            # the parameter is columnar, so get a list of bins and processs in order of appearance
            if not content.get("rates"):
                raise Exception("adding a columnar parameter requires the datacard to have "
                    "process rates")

            bin_names = content["rates"][0].split()[1:]
            process_names = content["rates"][1].split()[1:]
            if len(bin_names) != len(process_names):
                raise Exception("number of bins ({}) and processes ({}) not matching in datacard "
                    "rates".format(len(bin_names), len(process_names)))

            # build the new parameter line by looping through bin process pairs
            parts = []
            for bin_name, process_name in zip(bin_names, process_names):
                # go through param spec and stop when the first match is found
                for spec_bin_name, spec_process_name, spec_effect in param_spec:
                    if not multi_match(bin_name, spec_bin_name):
                        continue
                    if not multi_match(process_name, spec_process_name):
                        continue
                    parts.append(str(spec_effect))
                    break
                else:
                    # no match found, insert a "-"
                    parts.append("-")

            # add the new line
            param_line = " ".join([param_name, param_type] + parts)
            logger.info("adding new parameter line '{}'".format(param_line))
            content["parameters"].append(param_line)

        # increase kmax in counts
        if content.get("counts"):
            for i, count_line in enumerate(list(content["counts"])):
                if count_line.startswith("kmax"):
                    parts = count_line.split()
                    if len(parts) >= 2 and parts[1] != "*":
                        n_old = int(parts[1])
                        n_new = n_old + 1
                        logger.info("increase kmax from {} to {}".format(n_old, n_new))
                        parts[1] = str(n_new)
                        content["counts"][i] = " ".join(parts)
                    break


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", help="the datacard to read and possibly update (see --directory)")
    parser.add_argument("name", help="name of the parameter to add")
    parser.add_argument("type", help="type of the parameter to add")
    parser.add_argument("spec", nargs="*", help="specification of parameter arguments; for "
        "columnar parameters types (e.g. lnN or shape* nuisances), comma-separated triplets in the "
        "format 'bin,process,value' are expected; patterns are supported and evaluated in the "
        "given order for all existing bin process pairs; for all other types, the specification is "
        "used as is")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level)

    # add the parameter
    add_parameter(args.input, args.name, args.type, param_spec=args.spec,
        directory=args.directory, skip_shapes=args.no_shapes)
