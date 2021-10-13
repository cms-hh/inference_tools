#!/usr/bin/env python
# coding: utf-8

"""
Script to remove all shapes from files referenced in a datacard that are not
used. This is necessary to combine cards with parameters of different types
(e.g. lnN and shape) yet same names which will end up in the merged "shape?"
type. During text2Workspace.py, the original type is recovered via
*ModelBuilder.isShapeSystematic* by exploiting either the presence or absence
of a shape with a particular name which could otherwise lead to ambiguous
outcomes. Example usage:

# remove unused shapes for a certain process in a certain bin
> remove_unused_shapes.py datacard.txt 'OS_2018,ttbar' -d output_directory

# remove all unused shapes in the datacard
# (note the quotes)
> remove_unused_shapes.py datacard.txt '*,*' -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
from collections import defaultdict

import six

from dhi.datacard_tools import (
    read_datacard_structured, bundle_datacard, expand_variables, expand_file_lines,
)
from dhi.util import TFileCache, import_ROOT, create_console_logger, patch_object, multi_match


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_unused_shapes(datacard, rules, directory=None, mass="125", inplace_shapes=False):
    """
    Reads a *datacard* and removes shapes from referenced files that are not used by any shape
    parameter and that any of the passed *rules*. A rule can be a 2-tuple containing a bin pattern
    and a process pattern. When a pattern starts with '!', its meaning is negated.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    """
    ROOT = import_ROOT()

    # expand rules from files and parse them
    _rules = expand_file_lines(rules)
    rules = []
    for _rule in _rules:
        if isinstance(_rule, six.string_types):
            _rule = _rule.split(",")
        if not isinstance(_rule, (tuple, list)):
            raise TypeError("invalid rule '{}'".format(_rule))
        if len(_rule) != 2:
            raise ValueError("rule '{}' must have at exactly two values".format(_rule))
        rules.append(list(_rule))

    # nothing to do when no rules exist
    if not rules:
        logger.debug("no rules found")
        return

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=False)

    # read the datacard and remember which shapes to keep
    keep_shapes = defaultdict(set)
    content = read_datacard_structured(datacard)

    # per shape file, build a list of all shape objects to keep
    for shape_data in content["shapes"]:
        # check if any rules matches
        for bin_pattern, proc_pattern in rules:
            neg = bin_pattern.startswith("!")
            if multi_match(shape_data["bin"], bin_pattern[int(neg):]) == neg:
                continue

            neg = proc_pattern.startswith("!")
            if multi_match(shape_data["process"], proc_pattern[int(neg):]) == neg:
                continue

            # there was a match
            break
        else:
            continue

        # removal from workspaces is currently not supported
        if ":" in shape_data["nom_pattern"]:
            raise Exception("shape removal from workspaces is currently not supported")

        # get the full shape file and use it as a key later on
        shape_file = os.path.join(os.path.dirname(datacard), shape_data["path"])
        shape_file = os.path.realpath(shape_file)

        # keep the nominal shape
        nom_shape_name = expand_variables(shape_data["nom_pattern"], channel=shape_data["bin"],
            process=shape_data["process"], mass=mass)
        keep_shapes[shape_file].add(nom_shape_name)

        # do the same for data_obs (this might be called multiple times)
        data_shape_name = expand_variables(shape_data["nom_pattern"], channel=shape_data["bin"],
            process="data_obs", mass=mass)
        keep_shapes[shape_file].add(data_shape_name)

        # check if the systematic shapes should be kept (if any)
        for syst_data in content["parameters"]:
            if not multi_match(syst_data["type"], "shape*"):
                continue
            if not shape_data["syst_pattern"]:
                continue
            if shape_data["bin"] not in syst_data["spec"]:
                continue
            if syst_data["spec"][shape_data["bin"]].get(shape_data["process"], "-") in ["0", "-"]:
                continue

            # keep them
            syst_shape_name_up = expand_variables(shape_data["syst_pattern"],
                channel=shape_data["bin"], process=shape_data["process"],
                systematic=syst_data["name"] + "Up", mass=mass)
            syst_shape_name_down = expand_variables(shape_data["syst_pattern"],
                channel=shape_data["bin"], process=shape_data["process"],
                systematic=syst_data["name"] + "Down", mass=mass)
            keep_shapes[shape_file].add(syst_shape_name_up)
            keep_shapes[shape_file].add(syst_shape_name_down)

    # use a TFileCache for removing files
    for shape_file, keep in keep_shapes.items():
        with TFileCache(logger=logger) as cache:
            tfile = cache.open_tfile(shape_file, "UPDATE", tmp=not inplace_shapes)

            # recursively get keys of all non-directory objects
            keys = set()
            lookup = [(tfile, None)]
            while lookup:
                tdir, owner_key = lookup.pop(0)
                for tkey in tdir.GetListOfKeys():
                    key = tkey.GetName()
                    tobj = tdir.Get(key)
                    abs_key = (owner_key + "/" + key) if owner_key else key
                    if isinstance(tobj, ROOT.TDirectory):
                        lookup.append((tobj, abs_key))
                    elif isinstance(tobj, ROOT.TH1):
                        keys.add(abs_key)

            keep &= keys
            logger.info("keeping {} of {} shapes in file {}".format(
                len(keep), len(keys), shape_file))

            # remove all keys that are not kept
            for key in keys:
                if key not in keep:
                    logger.debug("dropping {} from {}".format(key, shape_file))
                    cache.delete_tobj(tfile, key + ";*")


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("rules", nargs="*", metavar="BIN,PROCESS", default=["*,*"], help="names of "
        "bins and processes for which unused shapes are removed; both names support patterns where "
        "a leading '!' negates their meaning; each argument can also be a file containing "
        "'BIN,PROCESS' values line by line; defaults to '*,*', removing unused shapes in all bins "
        "and processes")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--mass", "-m", default="125", help="mass hypothesis; default: 125")
    parser.add_argument("--inplace-shapes", "-i", action="store_true", help="change shape files "
        "in-place rather than in a temporary file first")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the renaming
    with patch_object(logger, "name", args.log_name):
        remove_unused_shapes(args.input, args.rules, directory=args.directory, mass=args.mass,
            inplace_shapes=args.inplace_shapes)
