#!/usr/bin/env python
# coding: utf-8

"""
Script to extract values from a RooFitResult object in a ROOT file into a json
file with configurable patterns for variable selection. Example usage:

# extract variables from a result 'fit_b' starting with 'CMS_'
# (note the quotes)
> extract_fit_result.py fit.root fit_b output.json --keep 'CMS_*'

# extract variables from a result 'fit_b' except thise starting with 'CMS_'
# (note the quotes)
> extract_fit_result.py fit.root fit_b output.json --skip 'CMS_*'
"""

import os
import json
from collections import OrderedDict

from dhi.util import import_ROOT, real_path, multi_match, create_console_logger, patch_object


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def extract_fit_result(fit_file, fit_name, output_file, keep_patterns=None, skip_patterns=None):
    """
    Extracts parameters from a RooFitResult named *fit_name* in a ROOT file *fit_file* and stores
    their values, errors, minima and maxima in a json *output_file*. The parameters to consider can
    be configured with sequences of *keep_patterns* and *skip_patterns*. When *keep_patterns* is
    *None*, all parameters are initially kept. The *skip_patterns* are evaluated in a second step,
    based on parameters that passed the *keep_patterns*.
    """
    ROOT = import_ROOT()

    # open the file at get the fit result object
    fit_file = real_path(fit_file)
    tfile = ROOT.TFile(fit_file, "READ")
    fit = tfile.Get(fit_name)
    if not fit:
        raise Exception("no object {} found in {}".format(fit_name, fit_file))
    if not isinstance(fit, ROOT.RooFitResult):
        raise Exception("object {} in {} is not a RooFitResult".format(fit_name, fit_file))
    logger.info("read RootFitResult {} from {}".format(fit_name, fit_file))

    # loop on parameters
    data = OrderedDict()
    params = fit.floatParsFinal()
    for i in range(params.getSize()):
        param = params.at(i)
        name = param.GetName()

        # check if we keep it
        keep = multi_match(name, keep_patterns) if keep_patterns else True
        if keep and skip_patterns:
            keep = not multi_match(name, skip_patterns)
        if not keep:
            logger.debug("skipping parameter {}".format(name))
            continue
        logger.debug("extracting parameter {}".format(name))

        # store data
        data[name] = {
            "value": param.getValV(),
            "error": param.getError(),
            "min": param.getMin(),
            "max": param.getMax(),
        }
    tfile.Close()
    logger.info("extracted {} parameters".format(len(data)))

    # save as json
    output_file = real_path(output_file)
    if os.path.exists(output_file):
        os.remove(output_file)
    elif not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("saved output file {}".format(output_file))


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input_file", metavar="INPUT", help="the input root file to read")
    parser.add_argument("fit_name", metavar="FIT", help="the name of the RootFitResult")
    parser.add_argument("output_file", metavar="OUTPUT", help="name of the output file to write")
    parser.add_argument("--keep", "-k", default=None, help="comma-separated patterns matching "
        "names of variables to keep")
    parser.add_argument("--skip", "-s", default=None, help="comma-separated patterns matching "
        "names of variables to skip")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # add the parameter
    with patch_object(logger, "name", args.log_name):
        extract_fit_result(args.input_file, args.fit_name, args.output_file,
            keep_patterns=args.keep and args.keep.split(","),
            skip_patterns=args.skip and args.skip.split(","))
