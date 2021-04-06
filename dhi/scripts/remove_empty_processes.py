#!/usr/bin/env python
# coding: utf-8

"""
Script to remove processes from datacard bins whose rate is below a certain
threshold. Bins, processes and the threshold value can be fully configured.
Example usage:

# remove a certain process from all bins where its rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt '*,tt,0.1' -d output_directory

# remove all processes in a certain bin whose rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt 'OS_2018,*,0.1' -d output_directory

# remove all processes except signal in a certain bin whose rate is below 0.1
# (note the quotes)
> remove_empty_processes.py datacard.txt 'OS_2018,*,0.1' -s -d output_directory

Note: The use of an output directory is recommended to keep input files
      unchanged.
"""

import os
from collections import OrderedDict

import six

from dhi.datacard_tools import read_datacard_structured, expand_variables, expand_file_lines
from dhi.util import TFileCache, create_console_logger, patch_object, multi_match
from dhi.scripts import remove_bin_process_pairs


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def remove_empty_processes(datacard, rules, skip_signal=False, directory=None, skip_shapes=False,
        mass="125"):
    """
    Reads a *datacard* and removes processes from certain bins depending on whether their rate is
    below a threshold value which can be configured through *rules*. A rule can be a 3-tuple
    containing a bin pattern, a process pattern and the threshold value. When a pattern starts with
    '!', its meaning is negated. When *skip_signal* is *True*, processes declared as signal in the
    datacard are skipped even if a process pattern matches. When multiple rules match the same bin
    process pair, the smallest threshold value is used.

    When *directory* is *None*, the input *datacard* and all shape files it refers to are updated
    in-place. Otherwise, both the changed datacard and its shape files are stored in the specified
    directory. For consistency, this will also update the location of shape files in the datacard.
    When *skip_shapes* is *True*, all shape files remain unchanged (the shape lines in the datacard
    itself are still changed).
    """
    # expand rules from files and parse them
    _rules = expand_file_lines(rules)
    rules = []
    for _rule in _rules:
        if isinstance(_rule, six.string_types):
            _rule = _rule.split(",")
        if not isinstance(_rule, (tuple, list)):
            raise TypeError("invalid rule '{}'".format(_rule))
        if len(_rule) != 3:
            raise ValueError("rule '{}' must have at exactly three values".format(_rule))
        rules.append([_rule[0], _rule[1], float(_rule[2])])

    # nothing to do when no rules exist
    if not rules:
        logger.debug("no rules found")
        return

    # maintain a list of bin process pairs to remove
    remove_pairs = []

    # read the datacard content
    content = read_datacard_structured(datacard)

    # get a list of signal processes per bin
    signal_names = [p["name"] for p in content["processes"] if p["id"] <= 0]

    # find all matched bin process pairs and the minimal threshold value that applies
    matched = OrderedDict()
    for bin_name, rates in content["rates"].items():
        for proc_name, rate in rates.items():
            # skip signal?
            if skip_signal and proc_name in signal_names:
                continue

            # check if at least one rule applies
            for bin_pattern, proc_pattern, threshold in rules:
                neg = bin_pattern.startswith("!")
                if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                    continue

                neg = proc_pattern.startswith("!")
                if multi_match(proc_name, proc_pattern[int(neg):]) == neg:
                    continue

                # remeber the minimal threshold
                key = (bin_name, proc_name)
                matched[key] = min(matched[key], threshold) if key in matched else threshold

    # get the rates of matched processes
    with TFileCache(logger=logger) as cache:
        for (bin_name, proc_name), threshold in matched.items():
            # get the rate directly from the correct shape
            for shape_data in content["shapes"]:
                if shape_data["bin"] != bin_name or shape_data["process"] != proc_name:
                    continue

                # open the file for reading
                file_path = os.path.join(os.path.dirname(datacard), shape_data["path"])
                tfile = cache.open_tfile(file_path, "READ")

                # get the nominal shape and get its rate
                rate = None
                variables = {"channel": bin_name, "process": proc_name, "mass": mass}
                shape_name = expand_variables(shape_data["nom_pattern"], **variables)
                if ":" in shape_name:
                    ws_name, arg_name = shape_name.split(":", 1)
                    ws = tfile.Get(ws_name)
                    if ws:
                        norm = ws.arg(arg_name + "_norm")
                        if norm:
                            rate = norm.getVal()
                else:
                    shape = tfile.Get(shape_name)
                    if shape:
                        rate = shape.Integral()

                if rate is not None:
                    logger.debug("extracted rate {} for process {} in bin {} from shape".format(
                        rate, proc_name, bin_name))
                else:
                    logger.error("no nominal shape {} found in file {} for process {} in "
                        "bin {}".format(shape_name, shape_data["path"], proc_name, bin_name))
                break
            else:
                # no shape line found, get the rate directly from the datacard
                rate = content["rates"][bin_name][proc_name]
                logger.debug("extracted rate for process {} in bin {} from datacard: {}".format(
                    proc_name, bin_name, rate))

                # special case
                if rate == -1:
                    rate = None
                    logger.warning("skipped process {} in bin {} as rate is -1 datacard".format(
                        proc_name, bin_name))

            # check the rate
            if rate is not None and rate < threshold:
                remove_pairs.append((bin_name, proc_name))
                logger.debug("rate {} of process {} in bin {} below threshold {}".format(
                    rate, proc_name, bin_name, threshold))

    logger.info("found {} bin-process pair(s) to remove".format(len(remove_pairs)))
    if remove_pairs:
        # just call remove_bin_process_pairs with our own logger
        with patch_object(remove_bin_process_pairs, "logger", logger):
            remove_bin_process_pairs.remove_bin_process_pairs(datacard, remove_pairs,
                directory=directory, skip_shapes=skip_shapes)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("rules", nargs="+", metavar="BIN,PROCESS,THRESHOLD", help="names of bins, "
        "processes and a threshold value below which processes are removed in the format "
        "'bin_name,process_name,threshold_value'; both names support patterns where a leading '!' "
        "negates their meaning; each argument can also be a file containing "
        "'BIN,PROCESS,THRESHOLD' values line by line")
    parser.add_argument("--skip-signal", "-s", action="store_true", help="skip signal processes, "
        "independent of whether they are matched by a process pattern")
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
        remove_empty_processes(args.input, args.rules, skip_signal=args.skip_signal,
            directory=args.directory, skip_shapes=args.no_shapes, mass=args.mass)
