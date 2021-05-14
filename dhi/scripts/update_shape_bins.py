#!/usr/bin/env python
# coding: utf-8

"""
Script to update histogram bins in datacard shapes using configurable rules.
Shapes stored in workspaces are not supported. Histograms can be updated
in-place using a referenceable function that is called with the signature
(bin_name, process_name, nominal_shape, systematic_shapes), where the latter
is a dictionary mapping systematic names to down and up varied shapes. Example
usage:

# file my_code.py
# ---------------
def func(bin_name, process_name, nominal_shape, systematic_shapes):
    for b in range(1, nominal_shape.GetNbinsX() + 1):
        nominal_shape.SetBinContent(b, ...)
# ---------------

# apply a function in all datacard bins to a specific process
# (note the quotes)
> update_shape_bins.py datacard.txt '*,ttbar,my_code.func' -d output_directory

# apply a function to all process in in all but one specific datacard bins
# (note the quotes)
> update_shape_bins.py datacard.txt '!CR,*,my_code.func' -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
import importlib

import six

from dhi.datacard_tools import (
    ShapeLine, manipulate_datacard, expand_variables, expand_file_lines, bundle_datacard,
)
from dhi.util import TFileCache, create_console_logger, patch_object, multi_match


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def update_shape_bins(datacard, rules, batch_processes=False, directory=None, mass="125"):
    """
    Reads a *datacard* and updates shape bins in nominal and / or systematically shifted histgrams
    according to certain *rules*. A rule should be a tuple consisting of a datacard bin, a process
    name, and either a function and a string in the format "module.function" to import a function.
    Bin and process names support patterns where a prepended '!' negates its meaning.

    The function is called with the signature (bin_name, process_name, nominal_hist, syst_hists),
    where *syst_hists* is a dictionary mapping systematic names to a tuple of down and up varied
    histograms. When *batch_processes* is *True*, the function is called only once per bin with
    *process_name*, *nominal_hist* and *syst_hists* being lists of same length, effectively handling
    all matching processes at once.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    The *mass* hypothesis is used to expand the '$MASS' field in shape line patterns.

    .. note::

        This function currently only supports shapes passed as histograms in standard ROOT files and
        does not work for RooFit workspaces.
    """
    # expand rules from files and parse them
    _rules = expand_file_lines(rules)
    rules = []
    for rule in _rules:
        if isinstance(rule, six.string_types):
            rule = rule.split(",")
        if not isinstance(rule, (tuple, list)):
            raise TypeError("invalid update rule '{}'".format(rule))
        rule = list(rule)
        if len(rule) != 3:
            raise ValueError("update rule '{}' must have three values".format(rule))

        # import functions
        if isinstance(rule[2], six.string_types) and "." in rule[2]:
            mod_id, func_name = rule[2].rsplit(".", 1)
            mod = importlib.import_module(mod_id)
            rule[2] = getattr(mod, func_name)

        if not callable(rule[2]):
            raise ValueError("function '{}' is not callable".format(rule[2]))
        rules.append(rule)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=False)

    # nothing to do when no rules exist
    if not rules:
        logger.debug("no rules found")
        return

    # read the datacard content
    with manipulate_datacard(datacard, read_structured=True) as (blocks, content):
        # nothing to do when there are no shape lines
        if not content["shapes"]:
            logger.debug("datacard does not contain shape lines")
            return

        # extract shape lines, sort them so that most specific ones (no wildcards) come first
        shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
        shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

        # get a mapping of bin to contained processes
        process_names = {
            bin_name: list(rates)
            for bin_name, rates in content["rates"].items()
        }

        # go through all rules
        matches = []
        for bin_pattern, proc_pattern, func in rules:
            for bin_name, proc_names in process_names.items():
                # match the bin
                neg = bin_pattern.startswith("!")
                if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                    continue

                # collect matching processes
                proc_data = []
                for proc_name in proc_names:
                    neg = proc_pattern.startswith("!")
                    if multi_match(proc_name, proc_pattern[int(neg):]) == neg:
                        continue

                    # find the first matching shape line
                    for sl in shape_lines:
                        if sl.is_fake or not sl.nom_pattern:
                            continue
                        if not multi_match(bin_name, sl.bin):
                            continue
                        if not multi_match(proc_name, sl.process):
                            continue

                        # reject shapes in workspaces
                        if ":" in sl.nom_pattern:
                            raise Exception("shape line for bin {} and process {} refers to "
                                "workspace in nominal pattern {} which is not supported".format(
                                    bin_name, proc_name, sl.nom_pattern))

                        # also find all matching shape uncertainties
                        syst_names = []
                        for param in content["parameters"]:
                            if not multi_match(param["type"], "shape*"):
                                continue
                            if param["spec"].get(bin_name, {}).get(proc_name, "-") != "-":
                                syst_names.append(param["name"])

                        # store data
                        proc_data.append((proc_name, sl, syst_names))
                        break
                    else:
                        logger.warning("no shape line found for bin {} and process {}".format(
                            bin_name, proc_name))

                # store data
                matches.append((bin_name, proc_data, func))
                logger.debug("found {} matching process(es) in datacard bin {}".format(
                    len(proc_data), bin_name))

        # get names of values of rates for manipulation downstream
        rate_bin_names = blocks["rates"][0].split()[1:]
        rate_proc_names = blocks["rates"][1].split()[1:]
        rate_values = blocks["rates"][3].split()[1:]
        if not (len(rate_bin_names) == len(rate_proc_names) == len(rate_values)):
            raise Exception("the number of bin names ({}), process names ({}) and rates ({}) "
                "does not match".format(len(rate_bin_names), len(rate_proc_names),
                len(rate_values)))

        # start a tfile cache for opening and updating shape files
        with TFileCache(logger=logger) as cache:
            # collect all process shapes per bin
            for bin_name, proc_data, func in matches:
                proc_shapes = []
                for proc_name, sl, syst_names in proc_data:
                    # open the file for writing
                    file_path = os.path.join(os.path.dirname(datacard), sl.file)
                    tfile = cache.open_tfile(file_path, "UPDATE")

                    # get the nominal shape
                    shape_name = expand_variables(sl.nom_pattern, process=proc_name,
                        channel=bin_name, mass=mass)
                    nom_shape = tfile.Get(shape_name)
                    if not nom_shape:
                        logger.warning("nominal shape named {} not found in file {} for bin {} and "
                            "process {}".format(shape_name, file_path, bin_name, proc_name))
                        continue

                    # get systematic shapes
                    syst_shapes = {}
                    if sl.syst_pattern:
                        for syst_name in syst_names:
                            shape_name = expand_variables(sl.syst_pattern, process=proc_name,
                                channel=bin_name, systematic=syst_name, mass=mass)
                            d_shape = tfile.Get(shape_name + "Down")
                            u_shape = tfile.Get(shape_name + "Up")
                            if not d_shape or not u_shape:
                                logger.warning("incomplete systematic shape named {}(Up|Down) "
                                    "in file {} for bin {} and process {}".format(
                                        shape_name, file_path, bin_name, proc_name))
                                continue
                            syst_shapes[syst_name] = (d_shape, u_shape)

                    # register all shapes for being rewritten upon cache context termination
                    cache.write_tobj(tfile, nom_shape)
                    for d_shape, u_shape in syst_shapes.values():
                        cache.write_tobj(tfile, d_shape)
                        cache.write_tobj(tfile, u_shape)

                    # store shapes
                    proc_shapes.append((proc_name, nom_shape, syst_shapes))

                # call the updating function
                if batch_processes:
                    func(
                        bin_name,
                        [proc_name for proc_name, _, _ in proc_shapes],
                        [nom_shape for _, nom_shape, _ in proc_shapes],
                        [syst_shapes for _, _, syst_shapes in proc_shapes],
                    )
                else:
                    for proc_name, nom_shape, syst_shapes in proc_shapes:
                        func(bin_name, proc_name, nom_shape, syst_shapes)

                for i, (rate_bin, rate_proc) in enumerate(zip(rate_bin_names, rate_proc_names)):
                    if rate_bin != bin_name:
                        continue
                    for proc_name, nom_shape, _ in proc_shapes:
                        if proc_name != rate_proc:
                            continue
                        rate_values[i] = "{:.4f}".format(nom_shape.Integral())
                logger.debug("updated {} rate(s) in datacard bin {}".format(
                    len(proc_shapes), bin_name))

            # write updated rate values back to the content
            blocks["rates"][3] = "rate " + " ".join(rate_values)
            logger.info("updated rates line with new values")


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("rules", nargs="+", metavar="BIN,PROCESS,FUNCTION", help="rules for "
        "updating datacard shape bins; 'BIN' and 'PROCESS' support patterns where a prepended '!' "
        "negates their meaning; 'FUNCTION' should have the format <MODULE_NAME>.<FUNCTION_NAME> to "
        "import a function 'FUNCTION_NAME' from the module 'MODULE_NAME'; the function should have "
        "the signature (bin_name, process_name, nominal_hist, syst_hists); this parameter also "
        "supports files that contain the rules in the described format line by line")
    parser.add_argument("--batch-processes", "-b", action="store_true", help="handle all processes "
        "in a bin by a single call to the passed function; 'process_name', 'nominal_hist' and "
        "'syst_hists' will be lists of the same length")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--mass", "-m", default="125", help="mass hypothesis; default: 125")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the renaming
    with patch_object(logger, "name", args.log_name):
        update_shape_bins(args.input, args.rules, batch_processes=args.batch_processes,
            directory=args.directory, mass=args.mass)
