#!/usr/bin/env python
# coding: utf-8

"""
Script to get yield tables from a datacard.
Example usage:

> yield_table.py datacard.txt --table-fmt latex
"""

import os
import re
import tabulate
from scinum import Number

from dhi.datacard_tools import create_datacard_instance
from dhi.util import real_path, create_console_logger, import_ROOT


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def yield_table(datacard, bins, tablefmt, data_name):
    ROOT = import_ROOT()
    datacard = real_path(datacard)
    dc, sb = create_datacard_instance(datacard, create_shape_builder=True)
    groups = {regex: [b for b in dc.bins if re.compile(regex).match(b)] for regex in bins}
    for group, cats in groups.items():
        logger.info("Selected {} from {} with regex: '{}'".format(cats, dc.bins, group))
    content = []
    use_latex = "latex" in tablefmt
    style = "latex" if use_latex else "plain"
    tmpl = "${}$" if use_latex else "{}"
    if use_latex:
        tabulate.LATEX_ESCAPE_RULES.pop(u"\\")
        tabulate.LATEX_ESCAPE_RULES.pop(u"$")
    for p in dc.processes + [data_name]:
        prate = [str(p)]
        for g, cats in groups.items():
            r = Number(0.0, 0.0)
            for c in cats:
                try:
                    shape = sb.getShape(c, p)
                    error = ROOT.double(0.0)
                    rate = shape.IntegralAndError(0, shape.GetNbinsX() + 1, error)
                except RuntimeError:
                    rate = 0.0
                    error = 0.0
                r += Number(rate, error)
            prate.append(tmpl.format(r.str("%.2f", style=style)))
        content.append(prate)
    print(
        tabulate.tabulate(
            content,
            headers=["Processes"] + groups.keys(),
            tablefmt=tablefmt,
            colalign=("left",) + ("right",) * len(groups),
        )
    )


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "input",
        metavar="DATACARD",
        help="the datacard to read",
    )
    parser.add_argument(
        "--bins",
        nargs="+",
        help="Regex to group bins. Default: '.+'.",
        default=".+",
    )
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--table-fmt", default="github", help="Table format. Default: 'github'.")
    parser.add_argument(
        "--data-name", default="data_obs", help="Name of observation. Default: 'data_obs'."
    )
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    if not isinstance(args.bins, (list, tuple)):
        bins = [args.bins]
    else:
        bins = args.bins
    # add the parameter
    yield_table(args.input, bins, args.table_fmt, args.data_name)
