#!/usr/bin/env python
# coding: utf-8

"""
Script to inject nuisance values from a json file into a RooFit workspace. The
json file must follow the same structure as produced by extract_fit_result.py.
Example usage:

# inject variables into a workspace "w"
> inject_fit_result.py input.json workspace.root w
"""

import os
import json

from dhi.util import import_ROOT, TFileCache, real_path, create_console_logger, patch_object


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def inject_fit_result(input_file, workspace_file, workspace_name):
    """
    Injects values, errors, minima and maxima of parameters stored in a json *input_file* into a
    workspace named *workspace_name* in a *workspace_file*.
    """
    ROOT = import_ROOT()

    # read the inputs
    input_file = real_path(input_file)
    with open(input_file, "r") as f:
        data = json.load(f)
    logger.info("read input file {}".format(input_file))

    # open the root file at get the workspace object
    with TFileCache(logger=logger) as cache:
        workspace_file = real_path(workspace_file)
        tfile = cache.open_tfile(workspace_file, "UPDATE")
        w = tfile.Get(workspace_name)
        if not w:
            raise Exception("no object {} found in {}".format(workspace_name, workspace_file))
        if not isinstance(w, ROOT.RooWorkspace):
            raise Exception("object {} in {} is not a RooWorkspace".format(
                workspace_name, workspace_file))
        cache.write_tobj(tfile, w)
        logger.info("read RooWorkspace {} from {}".format(workspace_name, workspace_file))

        # loop over data and inject values
        n = 0
        for name, d in data.items():
            v = w.var(name)
            if not v:
                logger.warning("workspace does not contain parameter {}".format(name))
                continue

            v.setVal(d["value"])
            v.setError(d["error"])
            v.setMin(d["min"])
            v.setMax(d["max"])
            logger.debug("injected parameter {}".format(name))
            n += 1
        logger.info("injected {} parameters into workspace".format(n))


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input_file", metavar="INPUT", help="the input json file to read")
    parser.add_argument("workspace_file", metavar="OUTPUT", help="the workspace file")
    parser.add_argument("workspace_name", metavar="WORKSPACE", help="name of the workspace")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # add the parameter
    with patch_object(logger, "name", args.log_name):
        inject_fit_result(args.input_file, args.workspace_file, args.workspace_name)
