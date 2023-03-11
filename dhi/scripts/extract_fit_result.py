#!/usr/bin/env python
# coding: utf-8

"""
Script to extract values from a RooFitResult object or a RooWorkspace (snapshot)
in a ROOT file into a json file with configurable patterns for variable selection.
Example usage:

# extract variables from a fit result 'fit_b' starting with 'CMS_'
# (note the quotes)
> extract_fit_result.py fit.root fit_b output.json --keep 'CMS_*'

# extract variables from a fit result 'fit_b' except those starting with 'CMS_'
# (note the quotes)
> extract_fit_result.py fit.root fit_b output.json --skip 'CMS_*'

# extract variables from a workspace snapshot starting with 'CMS_'
# (note the quotes)
> extract_fit_result.py workspace.root w:MultiDimFit output.json --keep 'CMS_*'
"""

import os
import json
from collections import OrderedDict

from dhi.util import (
    import_ROOT, real_path, multi_match, create_console_logger, patch_object, prepare_output,
)


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def extract_fit_result(input_file, obj_name, output_file, keep_patterns=None, skip_patterns=None):
    """
    Extracts parameters from a RooFitResult or RooWorkspace named *obj_name* in a ROOT file
    *input_file* and stores values, errors, minima and maxima in a json *output_file*. When the
    object is a RooWorkspace, *obj_name* name can contain a snapshot name to load in the format
    ``"workspace_name:snapshot_name"``.

    The parameters to consider can be configured with sequences of *keep_patterns* and
    *skip_patterns*. When *keep_patterns* is *None*, all parameters are initially kept. The
    *skip_patterns* are evaluated in a second step, based on parameters that passed the
    *keep_patterns*.
    """
    ROOT = import_ROOT()

    # open the file and get the object
    tfile = ROOT.TFile(real_path(input_file), "READ")
    snapshot_name = None
    if ":" in obj_name:
        obj_name, snapshot_name = obj_name.split(":", 1)
    tobj = tfile.Get(obj_name)
    if not tobj:
        raise Exception("no object {} found in {}".format(obj_name, input_file))
    if not isinstance(tobj, (ROOT.RooFitResult, ROOT.RooWorkspace)):
        raise Exception(
            "object {} in {} is neither a RooFitResult nor a RooWorkspace".format(
                obj_name, input_file,
            ),
        )
    if isinstance(tobj, ROOT.RooWorkspace):
        logger.info("read RooWorkspace {} from {}".format(obj_name, input_file))
        if snapshot_name:
            tobj.loadSnapshot(snapshot_name)
            logger.info("loaded snapshot {}".format(snapshot_name))
    else:
        logger.info("read RootFitResult {} from {}".format(obj_name, input_file))

    # create a parameter iterator
    if isinstance(tobj, ROOT.RooWorkspace):
        it = _argset_iter(tobj.allVars())
    else:
        it = _arglist_iter(tobj.floatParsFinal())

    # loop on parameters
    data = OrderedDict()
    for param in it:
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
            "min": param.getMin(),
            "max": param.getMax(),
            "error": param.getError(),
            "error_hi": param.getErrorHi(),
            "error_lo": param.getErrorLo(),
        }
    logger.info("extracted {} parameters".format(len(data)))

    # close the input file
    tfile.Close()

    # save as json
    output_file = prepare_output(output_file)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("saved output file {}".format(output_file))


def _arglist_iter(arglist):
    for i in range(arglist.getSize()):
        yield arglist.at(i)


def _argset_iter(argset):
    it = argset.createIterator()
    while True:
        param = it.Next()
        if not param:
            break
        yield param


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input_file",
        metavar="INPUT",
        help="the input root file to read",
    )
    parser.add_argument(
        "obj_name",
        metavar="NAME",
        help="name of the object to read values from",
    )
    parser.add_argument(
        "output_file",
        metavar="OUTPUT",
        help="name of the output file to write",
    )
    parser.add_argument(
        "--keep",
        "-k",
        default=None,
        help="comma-separated patterns matching names of variables to keep",
    )
    parser.add_argument(
        "--skip",
        "-s",
        default=None,
        help="comma-separated patterns matching names of variables to skip",
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

    # add the parameter
    with patch_object(logger, "name", args.log_name):
        extract_fit_result(
            args.input_file,
            args.obj_name,
            args.output_file,
            keep_patterns=args.keep and args.keep.split(","),
            skip_patterns=args.skip and args.skip.split(","),
        )
