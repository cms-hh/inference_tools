#!/usr/bin/env python
# coding: utf-8

"""
Script to extract datacard content into a json file in the structure:

{
    "bins": [{"name": bin_name}],
    "processes": [{"name": process_name, "id": process_id}],
    "rates": {bin_name: {process_name: float}},
    "observations": {bin_name: float},
    "parameters": [{"name": string, "type": string, "columnar": bool, "spec": object}],
}

Example usage:

# extract content and save to a certain file
> extract_datacard_content.py datacard.txt -o data.json
"""

import os
import json

from dhi.datacard_tools import read_datacard_structured
from dhi.util import real_path, create_console_logger, patch_object


script_name = os.path.splitext(os.path.basename(__file__))[0]
logger = create_console_logger(script_name)


def extract_datacard_content(datacard, output_file=None):
    """
    Reads the contents of a *datacard* and stores information about bins, processes, rates,
    observations, shape files and parameters to a json file at *output_file*. When not set, the same
    base name is used with a json file extension.
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # default output file
    if not output_file:
        output_file = os.path.splitext(datacard)[0] + ".json"

    # simply read the structured datacard content
    data = read_datacard_structured(datacard)

    # save the data
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("saved structured datacard content in {}".format(output_file))


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        metavar="DATACARD",
        help="the datacard to read",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="location of the json output file; default: DATACARD.json",
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

    # run the removing
    with patch_object(logger, "name", args.log_name):
        extract_datacard_content(args.input, output_file=args.output)
