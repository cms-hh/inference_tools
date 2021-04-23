#!/usr/bin/env python
# coding: utf-8

"""
Script to remove histogram bins from datacard shapes using configurable rules.
Shapes stored in workspaces are not supported. The bins to remove can be hard
coded, depend on signal or background content, or be identified through a
custom function. Example usage:

# remove the first 5 shape bins in a specific datacard bin
> remove_shape_bins.py datacard.txt 'OS_2018,1-5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events AND
# a S/sqrt(B) ratio (signal-to-noise) above 0.5
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5,STN>0.5' -d output_directory

# remove shape bins in all datacard bins with more than 5 signal events OR
# a S/sqrt(B) ratio (signal-to-noise) above 0.5
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,S>5' '*,STN>0.5' -d output_directory

# remove shape bins in all datacard bins using an exteral function
# (note the quotes)
> remove_shape_bins.py datacard.txt '*,my_module.func_name" -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
import re
import importlib
from collections import OrderedDict

import six

from dhi.datacard_tools import (
    ShapeLine, manipulate_datacard, expand_variables, expand_file_lines, bundle_datacard,
)
from dhi.util import TFileCache, create_console_logger, patch_object, multi_match, make_unique


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_shape_bins(datacard, rules, directory=None, skip_shapes=False, mass="125"):
    """
    Reads a *datacard* and remove shape bins from histgrams according to certain *rules*. A rule
    should be a tuple consisting of a datacard bin and at least one removal expression. Bin names
    support patterns where a prepended '!' negates its meaning. Three tpyes of expressions are
    interpreted:

    1. Colon-separated bin indices to remove, starting at 1. Values in the format 'A-B' refer to a
       range from A to B (inclusive). Omitting B will select all bins equal to and above A.
    2. An expression 'PROCESS(<|>)THRESHOLD', with special processes 'S', 'B', 'SB', 'SOB', and
       'STN' being interpreted as combined signal, background, signal+background, signal/background
       and signal/sqrt(background). Process names support patterns where a leading '!' negates their
       meaning. Process names can be joined via '+' to create sums.
    3. The location of a function in the format 'module.func_name' with signature
       (datacard_content, datacard_bin, histograms) that should return indices of bins to remove.

    Multiple removal expressions of the same rule are AND concatenated. *rules* themselves are OR
    concatenated (non-exclusive). Example:

    .. code-block:: python

        # rules for removing bins with more than 5 signal events AND a S/sqrt(B) value above 0.5
        rules = |("*", "S>5", "STN>0.5")]

        # rules for removing bins with more than 5 signal events OR a S/sqrt(B) value above 0.5
        rules = |("*", "S>5"), ("*", "STN>0.5")]

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    When *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).

    .. note::

        This function currently only supports shapes passed as histograms in standard ROOT files and
        does not work for RooFit workspaces.
    """
    # some constants
    INDICES, COMP, FUNC = range(3)

    # expand rules from files and parse them
    _rules = expand_file_lines(rules)
    rules = []
    for _rule in _rules:
        if isinstance(_rule, six.string_types):
            _rule = _rule.split(",")
        if not isinstance(_rule, (tuple, list)):
            raise TypeError("invalid removal rule '{}'".format(_rule))
        _rule = list(_rule)
        if len(_rule) < 2:
            raise ValueError("removal rule '{}' must have at least two values".format(_rule))
        rule = [str(_rule[0])]
        for expr in _rule[1:]:
            # bin indices?
            m = re.match(r"^[0-9\:\-]+$", expr)
            if m:
                indices = set()
                for part in expr.split(":"):
                    if "-" in part:
                        start, stop = part.split("-", 1)
                        # the stop value is optional
                        if not stop:
                            stop = 1000
                        indices |= set(range(int(start), int(stop) + 1))
                    else:
                        indices.add(int(part))
                rule.append((INDICES, list(indices)))
                continue

            # comparison expression
            m = re.match(r"^([^\<\>]+)(\<|\>)([\d\.]+)$", expr)
            if m:
                procs, comp_op, threshold = m.groups()
                comp_fn = (lambda v: v < threshold) if comp_op == "<" else (lambda v: v > threshold)
                rule.append((COMP, procs.split("+"), comp_op, float(threshold), comp_fn))
                continue

            # custom function?
            if "." in expr:
                mod_id, func_name = expr.rsplit(".", 1)
                try:
                    mod = importlib.import_module(mod_id)
                except ImportError:
                    mod = None
                if mod:
                    rule.append((FUNC, getattr(mod, func_name)))
                    continue

            raise Exception("cannot interpret removal expression '{}'".format(expr))

        rules.append(rule)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

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

        # get list of signal and background processes per bin
        all_signal_names = [p["name"] for p in content["processes"] if p["id"] <= 0]
        all_background_names = [p["name"] for p in content["processes"] if p["id"] > 0]
        signal_names = {
            bin_name: make_unique([name for name in rates if name in all_signal_names])
            for bin_name, rates in content["rates"].items()
        }
        background_names = {
            bin_name: make_unique([name for name in rates if name in all_background_names])
            for bin_name, rates in content["rates"].items()
        }
        logger.debug("found a total of {} signal and {} background processes".format(
            len(all_signal_names), len(all_background_names)))

        # check if all processes passed in comparison rules exist
        for rule in rules:
            for expr in rule[1:]:
                if expr[0] != COMP:
                    continue
                for proc in expr[1]:
                    if proc in ["S", "B", "SB", "SOB", "STN", "D"]:
                        continue
                    elif proc not in all_signal_names + all_background_names:
                        raise Exception("process '{}' in rule {} does not exist in datacard".format(
                            proc, expr))

        # get a list of datacard bins that are affected in some way
        bin_names = []
        for bin_name in signal_names.keys():
            for rule in rules:
                bin_pattern = rule[0]
                neg = bin_pattern.startswith("!")
                if multi_match(bin_name, bin_pattern[int(neg):]) != neg:
                    bin_names.append(bin_name)
                    break
        if not bin_names:
            logger.debug("removal rules do not match any datacard bin")
            return
        logger.info("going to process {} matched datacard bin(s)".format(len(bin_names)))

        # start a tfile cache for opening and updating shape files
        with TFileCache(logger=logger) as cache:
            # extract shape lines, sort them so that most specific ones (no wildcards) come first
            shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
            shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

            # get shapes (nominal and systematic ones) of all processes per bin and maintain a list
            # of current shape bin indices per datacard bin which is reduced later on
            shapes = {}
            bin_nums = {}
            for bin_name in bin_names:
                proc_names = signal_names.get(bin_name, []) + background_names.get(bin_name, [])
                proc_names.append("data_obs")
                for proc_name in proc_names:
                    # find the shape line that applies
                    for shape_line in shape_lines:
                        if shape_line.is_fake or not shape_line.nom_pattern:
                            continue
                        if not multi_match(bin_name, shape_line.bin):
                            continue
                        if not multi_match(proc_name, shape_line.process):
                            continue

                        # reject shapes in workspaces
                        if ":" in shape_line.nom_pattern:
                            raise Exception("shape line for bin {} and process {} refers to "
                                "workspace in nominal pattern {} which is not supported".format(
                                    bin_name, proc_name, shape_line.nom_pattern))

                        # open the file for writing
                        file_path = os.path.join(os.path.dirname(datacard), shape_line.file)
                        tfile = cache.open_tfile(file_path, "UPDATE")

                        # helper for reading shapes and validating the number of bins
                        def read_shape(name):
                            abs_name = name
                            owner = tfile
                            if "/" in name:
                                owner_name, name = name.rsplit("/", 1)
                                owner = owner.Get(owner_name)
                            shape = owner.Get(name)

                            # check or store bins
                            n = shape.GetNbinsX()
                            if bin_name not in bin_nums:
                                bin_nums[bin_name] = n
                            elif bin_nums[bin_name] != n:
                                raise Exception("the number of shape bins in datacard bin {} was "
                                    "set to {} before, but {} bins were found in shape {}".format(
                                        bin_name, bin_nums[bin_name], n, abs_name))

                            return tfile, owner, owner.Get(name), name

                        # get the nominal shape
                        variables = {"channel": bin_name, "process": proc_name, "mass": mass}
                        shape_name = expand_variables(shape_line.nom_pattern, **variables)
                        nom_shape = read_shape(shape_name)

                        # get all systematic shapes
                        syst_shapes = {}
                        if shape_line.syst_pattern:
                            for param in content["parameters"]:
                                if not multi_match(param["type"], "shape*"):
                                    continue
                                if bin_name not in param["spec"]:
                                    continue
                                if param["spec"][bin_name].get(proc_name, "-") in ["-", "0"]:
                                    continue
                                syst_shapes[param["name"]] = {
                                    "up": read_shape(expand_variables(shape_line.syst_pattern,
                                        systematic=param["name"] + "Up", **variables)),
                                    "down": read_shape(expand_variables(shape_line.syst_pattern,
                                        systematic=param["name"] + "Down", **variables)),
                                }

                        # store them
                        shapes.setdefault(bin_name, {})[proc_name] = (nom_shape, syst_shapes)
                        logger.debug("extracted {} shape(s) for bin {} and process {}".format(
                            1 + len(syst_shapes) * 2, bin_name, proc_name))
                        break

                logger.info("loaded shapes of {} process(es) in datacard bin {}".format(
                    len(shapes[bin_name]), bin_name))

            # keep sets of shape bin indices per datacard bin to remove
            remove_bin_indices = {bin_name: set() for bin_name in shapes}

            # apply rules to remove bin indices
            for rule in rules:
                bin_pattern, expressions = rule[0], rule[1:]
                for bin_name, _shapes in shapes.items():
                    neg = bin_pattern.startswith("!")
                    if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                        continue

                    # prepare nominal shapes mapped to process names
                    nom_hists = {
                        proc_name: nom_shape[2]
                        for proc_name, (nom_shape, _) in _shapes.items()
                    }

                    # helper to get bin contents for one or more processes
                    def get_bin_contents(procs):
                        return [
                            sum((nom_hists[p].GetBinContent(b) for p in procs if p in nom_hists), 0)
                            for b in range(1, bin_nums[bin_name] + 1)
                        ]

                    # determine the bin indices to remove per expression for AND concatenation
                    indices = []
                    for expr in expressions:
                        if expr[0] == INDICES:
                            _indices = expr[1]
                        elif expr[0] == FUNC:
                            _indices = expr[1](content, bin_name, nom_hists)
                        elif expr[0] == COMP:
                            procs, _, _, comp_fn = expr[1:]

                            # get bin values
                            if len(procs) == 1 and procs[0] in ["S", "B", "SB", "SOB", "STN"]:
                                # prepare signal and background values if needed
                                bin_values_s, bin_values_b = None, None
                                if procs[0] in ["S", "SB", "SOB", "STN"]:
                                    bin_values_s = get_bin_contents(signal_names[bin_name])
                                if procs[0] in ["B", "SB", "SOB", "STN"]:
                                    bin_values_b = get_bin_contents(background_names[bin_name])

                                # prepare special bin values
                                if procs[0] == "S":
                                    bin_values = bin_values_s
                                elif procs[0] == "B":
                                    bin_values = bin_values_b
                                elif procs[0] == "SB":
                                    bin_values = [
                                        s + b
                                        for s, b in zip(bin_values_s, bin_values_b)
                                    ]
                                elif procs[0] == "SOB":
                                    bin_values = [
                                        0. if s == 0 else (1.e7 if b == 0 else (s / b))
                                        for s, b in zip(bin_values_s, bin_values_b)
                                    ]
                                else:  # "STN"
                                    bin_values = [
                                        0. if s == 0 else (1.e7 if b == 0 else (s / b**0.5))
                                        for s, b in zip(bin_values_s, bin_values_b)
                                    ]
                            else:
                                bin_values = get_bin_contents(procs)

                            # apply the comparison function
                            _indices = [(i + 1) for i, v in enumerate(bin_values) if comp_fn(v)]

                        # limit and store the indices
                        indices.append({i for i in _indices if (1 <= i <= bin_nums[bin_name])})

                    # AND concatente indices to drop by finding those existing in all lists
                    joined_indices = set(
                        b for b in set.union(*indices)
                        if all(b in _indices for _indices in indices)
                    )

                    # OR concatenate with previous bins
                    remove_bin_indices[bin_name] |= joined_indices

            # remove bins dropped above and remember new observations and rates per process
            new_rates = {}
            new_observations = {}
            for bin_name, _shapes in shapes.items():
                indices = list(sorted({b for b in remove_bin_indices[bin_name] if b > 0}))
                if not indices:
                    continue

                logger.info("dropping {} shape bin(s) in datacard bin {}".format(
                    len(indices), bin_name))
                logger.info("shape bin indices to remove in bin {}: {}".format(
                    bin_name, ",".join(map(str, indices))))

                for proc_name, (nom_shape, syst_shapes) in _shapes.items():
                    # update the nominal hist
                    tfile, owner, hist, name = nom_shape
                    new_hist = drop_shape_bins(hist, name, indices, owner)
                    if new_hist:
                        cache.write_tobj(tfile, new_hist, towner=owner, name=name)

                        # remember rate or observation
                        if proc_name == "data_obs":
                            new_observations[bin_name] = new_hist.Integral()
                        else:
                            new_rates.setdefault(bin_name, {})[proc_name] = new_hist.Integral()

                    # update the all syst hists
                    for _syst_shapes in syst_shapes.values():
                        for tfile, owner, hist, name in _syst_shapes.values():
                            new_hist = drop_shape_bins(hist, name, indices, owner)
                            if new_hist:
                                cache.write_tobj(tfile, new_hist, towner=owner, name=name)

            # update observations
            obs_bin_names = blocks["observations"][0].split()[1:]
            obs_values = blocks["observations"][1].split()[1:]
            if len(obs_bin_names) != len(obs_values):
                raise Exception("the number of bin names ({}) and observations ({}) does not "
                    "match".format(len(obs_bin_names), len(obs_values)))
            new_obs_values = [
                (str(new_observations.get(bin_name, obs_value)) if obs_value != "-1" else obs_value)
                for bin_name, obs_value in zip(obs_bin_names, obs_values)
            ]
            blocks["observations"][1] = "observation " + " ".join(new_obs_values)
            logger.info("added new observation line with updated integrals in {} bin(s)".format(
                len(new_observations)))

            # update rates per process
            rates_bin_names = blocks["rates"][0].split()[1:]
            process_names = blocks["rates"][1].split()[1:]
            rate_values = blocks["rates"][3].split()[1:]
            if not (len(rates_bin_names) == len(process_names) == len(rate_values)):
                raise Exception("the number of bin names ({}), process names ({}) and rates ({}) "
                    "does not match".format(len(rates_bin_names), len(process_names),
                    len(rate_values)))
            new_rate_values = [
                (str(new_rates.get(bin_name, {}).get(proc_name, rate)) if rate != "-1" else rate)
                for bin_name, proc_name, rate in zip(rates_bin_names, process_names, rate_values)
            ]
            blocks["rates"][3] = "rate " + " ".join(new_rate_values)
            logger.info("added new rates line with updated integrals in {} bin(s)".format(
                len(new_rates)))


def drop_shape_bins(hist, name, drop_indices, owner):
    if not hist or not drop_indices or not owner:
        return None

    # create a mapping from new to old bins
    x_axis = hist.GetXaxis()
    y_axis = hist.GetYaxis()
    bin_mapping = OrderedDict()
    for b in range(1, x_axis.GetNbins() + 1):
        if b not in drop_indices:
            bin_mapping[len(bin_mapping) + 1] = b

    # prepare the title
    title = ";".join([hist.GetTitle(), x_axis.GetTitle(), y_axis.GetTitle()])

    # prepare the binning
    binning = (len(bin_mapping), 0., float(len(bin_mapping)))

    # create the new histogram with same type
    owner.cd()
    new_hist = hist.__class__(name, title, *binning)

    # register sumw2
    w2 = list(hist.GetSumw2())
    new_hist.Sumw2(bool(w2))

    # fill values
    for new_bin, old_bin in bin_mapping.items():
        # bin content
        new_hist.SetBinContent(new_bin, hist.GetBinContent(old_bin))

        # sumw2 error
        if w2:
            new_hist.SetBinError(new_bin, (w2[old_bin])**0.5)

    return new_hist


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("rules", nargs="+", metavar="BIN,EXPRESSION[,EXPRESSION]", help="removal "
        "rules for shape bins in a datacard bin 'BIN', which supports patterns; prepending '!' to "
        "a bin pattern negates its meaning; an 'EXPRESSION' can either be a list of "
        "colon-separated bin indices to remove (starting at 1) with values 'A-B' being interpreted "
        "as ranges from A to B (inclusive), a simple expression 'PROCESS(<|>)THRESHOLD' (with "
        "special processes 'S', 'B', 'SB', 'SOB' and 'STN' being interpreted as combined signal, "
        "background, signal+background, signal/background, and signal/sqrt(background)), or the "
        "location of a function in the format 'module.func_name' with signature (datacard_content, "
        "datacard_bin, histograms) that should return indices of bins to remove; mutliple rules "
        "passed in the same expression are AND concatenated; the rules of multiple arguments are "
        "OR concatenated; each argument can also be a file containing "
        "'BIN,EXPRESSION[,EXPRESSION]' values line by line")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not change process "
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
        remove_shape_bins(args.input, args.rules, directory=args.directory,
            skip_shapes=args.no_shapes, mass=args.mass)
