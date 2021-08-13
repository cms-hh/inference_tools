#!/usr/bin/env python
# coding: utf-8

"""
Script to plot histogram shapes of a datacard using configurable rules.
Shapes stored in workspaces are not supported at the moment. Example usage:

# plot all nominal shapes in a certain datacard bin
# (note the quotes)
> plot_datacard_shapes.py datacard.txt 'ee_2018,*'

# plot all nominal shapes of a certain process in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,ttbar'

# plot all systematic shapes of a certain process in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,ttbar,*'

# plot all systematic shapes of all signals in all datacard bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt '*,S,*'

# plot all systematic shapes of two stacked processes in all bins
# (note the quotes)
> plot_datacard_shapes.py datacard.txt --stack '*,ttbar+singlet,*'
"""

import os
import math

import six

from dhi.datacard_tools import ShapeLine, manipulate_datacard, expand_variables, expand_file_lines
from dhi.util import (
    TFileCache, import_ROOT, create_console_logger, patch_object, multi_match, make_unique,
    real_path, to_root_latex,
)
from dhi.plots.util import use_style
from dhi.config import colors


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def plot_datacard_shapes(datacard, rules, stack=False, directory=".",
        nom_format="{bin}__{process}.pdf", syst_format="{bin}__{process}__{syst}.pdf", mass="125",
        **plot_kwargs):
    """
    Reads a *datacard* and plots its histogram shapes according to certain *rules*. A rule should
    be a tuple consisting of a datacard bin, a datacard process and an optional name of a systematic
    uncertainty. When the latter is missing, only the nominal shape is plotted. All elements support
    patterns, where a prepended '!' negates their meaning. Multiple process patterns can be
    concatenated with '+'. By default, a separate plot is created per process matched by any of the
    process name patterns of a rule. When *stack* is *True*, the distributions for all matched
    processes are stacked and a single plot is created.

    Certain values of process and systematic names are interpreted in a special way. Processes 'S',
    'B' and 'SB' are interpreted as combined signal, background and signal+background (using the
    *stack* feature). Systematics 'S' and 'R' denote all shape changing (type 'shape*') and all rate
    changing nuisances (types 'lnN' and 'lnU').

    The binning strategy can be adjust using the *binning* argument:
    - ``"original"``: Use original bin edges.
    - ``"numbers"``: Use equidistant edges using bin numbers.
    - ``"numbers_width"``: Use equidistant edges using bin numbers and divide by bin widths.

    Plots are stored in a specificed *directory* using *nom_format* when plotting only nominal
    shapes, and *syst_format* otherwise. The *mass* hypothesis is used to expand the '$MASS' field
    in shape line patterns. All additional *plot_kwargs* are forwarded to
    :py:func:`create_shape_plot`.

    .. note::

        This function currently only supports shapes stored as histograms in standard ROOT files and
        does not work for RooFit workspaces.
    """
    # expand rules from files and parse them
    _rules = expand_file_lines(rules)
    rules = []
    for rule in _rules:
        if isinstance(rule, six.string_types):
            rule = rule.split(",")
        if not isinstance(rule, (tuple, list)):
            raise TypeError("invalid shape rule '{}'".format(rule))
        rule = list(rule)
        if len(rule) == 2:
            rule.append(None)
        elif len(rule) != 3:
            raise ValueError("shape rule '{}' must have two or three values".format(rule))
        rules.append(rule)

    # nothing to do when no rules exist
    if not rules:
        logger.debug("no rules found")
        return

    # prepare the output directory
    directory = real_path(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        logger.debug("found a total of {} signal and {} background processes".format(
            len(all_signal_names), len(all_background_names)))

        # expand bin and process patterns in rules
        expanded_rules = []
        for bin_pattern, proc_patterns, syst_pattern in rules:
            for bin_name in content["rates"]:
                neg = bin_pattern.startswith("!")
                if multi_match(bin_name, bin_pattern[int(neg):]) == neg:
                    continue

                # strategy: collect matching processes and check if they enter the expanded rules
                # separately or stacked
                matched_procs = []
                for proc_pattern in proc_patterns.split("+"):
                    # handle special process names
                    if proc_pattern == "S":
                        matched_procs.append(("Signal", signal_names[bin_name]))
                        continue
                    if proc_pattern == "B":
                        matched_procs.append(("Background", background_names[bin_name]))
                        continue
                    if proc_pattern == "SB":
                        matched_procs.append(("Signal+Background",
                            signal_names[bin_name] + background_names[bin_name]))
                        continue

                    # match process patterns
                    for proc_name in signal_names[bin_name] + background_names[bin_name]:
                        neg = proc_pattern.startswith("!")
                        if multi_match(proc_name, proc_pattern[int(neg):]) != neg:
                            matched_procs.append((proc_name, [proc_name]))

                # add to expanded rules
                if stack:
                    expanded_rules.append((
                        bin_name,
                        (proc_patterns, list(set(sum((procs for _, procs in matched_procs), [])))),
                        syst_pattern,
                    ))
                else:
                    expanded_rules.extend([
                        (bin_name, (proc_label, procs), syst_pattern)
                        for proc_label, procs in matched_procs
                    ])

        # expand systematic patterns in rules
        _expanded_rules = []
        for bin_name, proc_data, syst_pattern in expanded_rules:
            if not syst_pattern:
                _expanded_rules.append([bin_name, proc_data, None])
                continue

            for param in content["parameters"]:
                # skip unsupported types
                if not multi_match(param["type"], ["lnN", "lnU", "shape*"]):
                    continue

                # handle special systematic names
                if syst_pattern == "S":  # all shapes
                    if multi_match(param["type"], "shape*"):
                        _expanded_rules.append([bin_name, proc_data, param])
                    continue
                if syst_pattern == "R":  # all rates
                    if param["type"] in ["lnN", "lnU"]:
                        _expanded_rules.append([bin_name, proc_data, param])
                    continue

                # match pattern
                neg = syst_pattern.startswith("!")
                if multi_match(param["name"], syst_pattern[int(neg):]) == neg:
                    continue

                # check if it applies to the bin
                if bin_name not in param["spec"]:
                    continue

                # check if it applies to any process in that bin
                if all(param["spec"][bin_name].get(proc, "-") == "-" for proc in proc_data[1]):
                    continue

                # store the rule
                _expanded_rules.append([bin_name, proc_data, param])
        expanded_rules = _expanded_rules
        logger.info("going to produce {} shape plot(s)".format(len(expanded_rules)))

        # extract shape lines, sort them so that most specific ones (no wildcards) come first
        shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks["shapes"])]
        shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

        # start a tfile cache for opening files and reading shapes
        with TFileCache(logger=logger) as cache:
            # traverse expanded rules
            for bin_name, (proc_label, proc_names), param in expanded_rules:
                # per process, determine the nominal shape, and optionally shifted ones
                proc_shapes = {}
                for proc_name in proc_names:
                    # get the first matching shape line
                    for sl in shape_lines:
                        if sl.is_fake:
                            continue
                        if multi_match(bin_name, sl.bin) and multi_match(proc_name, sl.process):
                            break
                    else:
                        logger.warning("no shape line found matching bin {} and process {}".format(
                            bin_name, proc_name))
                        continue

                    # reject shapes in workspaces
                    if ":" in sl.nom_pattern:
                        logger.warning("shape line for bin {} and process {} refers to workspace "
                            "in nominal pattern {} which is not supported".format(
                                bin_name, proc_name, sl.nom_pattern))
                        continue

                    # open the file for writing
                    file_path = os.path.join(os.path.dirname(datacard), sl.file)
                    tfile = cache.open_tfile(file_path, "READ")

                    # get the nominal shape
                    shape_name = expand_variables(sl.nom_pattern, process=proc_name,
                        channel=bin_name, mass=mass)
                    nom_shape = tfile.Get(shape_name)
                    if not nom_shape:
                        logger.warning("nominal shape named {} not found in file {} for bin {} and "
                            "process {}".format(shape_name, file_path, bin_name, proc_name))
                        continue

                    # get the systematic shapes when param has type shape, otherwise use rate effect
                    d, u = None, None
                    effect = param["spec"][bin_name].get(proc_name, "-") if param else "-"
                    if effect != "-":
                        if param["type"] in ["lnN", "lnU"]:
                            if "/" in effect:
                                d, u = map(float, effect.split("/", 1))
                            else:
                                u = float(effect)
                                d = 2 - u
                        else:  # shape*
                            if sl.syst_pattern:
                                shape_name = expand_variables(sl.syst_pattern, process=proc_name,
                                    channel=bin_name, systematic=param["name"], mass=mass)
                                u_shape = tfile.Get(shape_name + "Up")
                                d_shape = tfile.Get(shape_name + "Down")
                                if u_shape and d_shape:
                                    u, d = u_shape, d_shape
                                else:
                                    logger.warning("incomplete systematic shape named {}(Up|Down) "
                                        "in file {} for bin {} and process {}".format(
                                            shape_name, file_path, bin_name, proc_name))
                            else:
                                logger.warning("shape line for bin {} and process {} does not "
                                    "contain a systematic pattern".format(bin_name, proc_name))

                    # store the shape info
                    proc_shapes[proc_name] = (nom_shape, d, u)

                # do nothing when a parameter is set, but no shape was found (likely a misconfig)
                if param and all(d is None for _, d, _ in proc_shapes.values()):
                    continue

                # draw the shape
                create_shape_plot(bin_name, proc_label, proc_shapes, param, directory, nom_format,
                    syst_format, **plot_kwargs)


@use_style("dhi_default")
def create_shape_plot(bin_name, proc_label, proc_shapes, param, directory, nom_format, syst_format,
        binning="original", x_title="Datacard shape", y_min=None, y_max=None, y_min2=None,
        y_max2=None, y_log=False, campaign_label=None):
    import plotlib.root as r
    ROOT = import_ROOT()

    # check if systematic shifts are to be plotted, determine the plot path
    plot_syst = param is not None and any(d is not None for _, d, _ in proc_shapes.values())
    if plot_syst:
        path = syst_format.format(bin=bin_name, process=proc_label, syst=param["name"])
    else:
        path = nom_format.format(bin=bin_name, process=proc_label)
    path = path.replace("*", "X").replace("?", "Y").replace("!", "N")
    path = os.path.join(directory, path)
    logger.debug("going to create plot at {} for shapes in bin {} and process {}, stacking {} "
        "processes".format(path, bin_name, proc_label, len(proc_shapes)))

    # combine histograms
    hist_n, hist_d, hist_u = None, None, None
    for n, d, u in proc_shapes.values():
        if not hist_n:
            hist_n = n.Clone("h__{}__{}__{}".format(bin_name, proc_label, param and param["name"]))
        else:
            hist_n.Add(n)

        if plot_syst:
            if d is None:
                d = n
            elif isinstance(d, float):
                _d = n.Clone(n.GetName() + "Down")
                _d.Scale(d)
                d = _d
            if not hist_d:
                hist_d = d.Clone(hist_n.GetName() + "_down")
            else:
                hist_d.Add(d)

            if u is None:
                u = n
            elif isinstance(u, float):
                _u = n.Clone(n.GetName() + "Up")
                _u.Scale(u)
                u = _u
            if not hist_u:
                hist_u = u.Clone(hist_n.GetName() + "_up")
            else:
                hist_u.Add(u)

    # apply the binning strategy, adjust axis titles and properties
    if binning == "original":
        hist_n_trans = hist_n
        hist_d_trans = hist_d
        hist_u_trans = hist_u
        y_title = "Events"
        x_props = {}
    elif binning in ["numbers", "numbers_width"]:
        hist_n_trans = transform_binning(hist_n, binning)
        if plot_syst:
            hist_d_trans = transform_binning(hist_d, binning)
            hist_u_trans = transform_binning(hist_u, binning)
        x_title = "Bin number"
        y_title = "Events" if binning == "numbers" else "Events / bin width"
        x_props = {"Ndivisions": max(2, hist_n_trans.GetNbinsX())}
    else:
        raise ValueError("unknown binning strategy '{}'".format(binning))

    # get axis ranges
    hists = [hist_n_trans]
    if plot_syst:
        hists.extend([hist_d_trans, hist_u_trans])
    x_min = hist_n_trans.GetXaxis().GetXmin()
    x_max = hist_n_trans.GetXaxis().GetXmax()
    y_min_value = min(
        min(h.GetBinContent(b) - h.GetBinErrorLow(b) for b in range(1, h.GetNbinsX() + 1))
        for h in hists
    )
    y_max_value = max(
        max(h.GetBinContent(b) + h.GetBinErrorUp(b) for b in range(1, h.GetNbinsX() + 1))
        for h in hists
    )
    if y_log:
        if y_min is None:
            y_min = (0.75 * y_min_value) if y_min_value > 0 else 1e-3
        if y_max is None:
            y_max = y_min * 10**(math.log10(y_max_value / y_min) * 1.38)
    else:
        if y_min is None:
            y_min = 0.
        if y_max is None:
            y_max = (y_max_value - y_min) * 1.38

    # start plotting
    r.setup_style()
    if plot_syst:
        canvas, (pad1, pad2) = r.routines.create_canvas(divide=(1, 2))
        r.setup_pad(pad1, props={"Logy": y_log, "BottomMargin": 0.3})
        r.setup_pad(pad2, props={"TopMargin": 0.7, "Gridy": 1})
    else:
        canvas, (pad1,) = r.routines.create_canvas()
    pad1.cd()
    draw_objs1 = []
    draw_objs2 = []
    legend_entries = []

    # dummy histograms for both pads to control axes
    h_dummy1 = ROOT.TH1F("dummy1", ";{};{}".format(x_title, y_title), 1, x_min, x_max)
    r.setup_hist(h_dummy1, pad=pad1, props={"LineWidth": 0, "Minimum": y_min, "Maximum": y_max})
    r.setup_x_axis(h_dummy1.GetXaxis(), pad=pad1, props=x_props)
    r.setup_y_axis(h_dummy1.GetYaxis(), pad=pad1, props={"TitleOffset": 1.6})
    draw_objs1.append((h_dummy1, "HIST"))
    if plot_syst:
        r.setup_x_axis(h_dummy1.GetXaxis(), pad1, props={"Title": "", "LabelSize": 0})

        h_dummy2 = ROOT.TH1F("dummy2", ";{};Change / %".format(x_title), 1, x_min, x_max)
        r.setup_hist(h_dummy2, pad=pad2, props={"LineWidth": 0})
        r.setup_x_axis(h_dummy2.GetXaxis(), pad=pad2, props=x_props)
        r.setup_x_axis(h_dummy2.GetXaxis(), pad=pad2, props={"TitleOffset": 1.23})
        r.setup_y_axis(h_dummy2.GetYaxis(), pad=pad2, props={"Ndivisions": 6, "CenterTitle": True,
            "LabelSize": 20, "TitleOffset": 1.6})
        draw_objs2.append((h_dummy2, "HIST"))

    # add top histograms
    r.setup_hist(hist_n_trans, props={"LineWidth": 2, "LineColor": colors.root.black})
    draw_objs1.append((hist_n_trans, "SAME,HIST,E"))
    legend_entries.append((hist_n_trans, "Nominal", "L"))
    if plot_syst:
        label_d, label_u = "Down", "Up"
        if hist_n.Integral():
            def fmt(v):
                sign = 1. if v > 0 else -1.
                prefix = "" if round(v, 2) else {1: "< ", -1: "> "}[sign]
                return prefix + "{:+.2f}".format({1: max, -1: min}[sign](v, 0.01 * sign))
            change_d = 100 * (hist_d.Integral() - hist_n.Integral()) / hist_n.Integral()
            change_u = 100 * (hist_u.Integral() - hist_n.Integral()) / hist_n.Integral()
            label_d += "   ({}%)".format(fmt(change_d))
            label_u += "      #scale[0.775]{{ }}({}%)".format(fmt(change_u))
        r.setup_hist(hist_u_trans, props={"LineWidth": 2, "LineColor": colors.root.green})
        draw_objs1.insert(-1, (hist_u_trans, "SAME,HIST,E"))
        legend_entries.append((hist_u_trans, label_u, "L"))
        r.setup_hist(hist_d_trans, props={"LineWidth": 2, "LineColor": colors.root.red})
        draw_objs1.insert(-2, (hist_d_trans, "SAME,HIST,E"))
        legend_entries.append((hist_d_trans, label_d, "L"))

    # bin, process and systematic labels
    def create_top_left_labels(key, value, i):
        common = {"pad": pad1, "props": {"TextSize": 18}, "y_offset": 44 + 24 * i}
        return [
            r.routines.create_top_left_label(key + ":", x_offset=20, **common),
            r.routines.create_top_left_label(value, x_offset=120, **common),
        ]

    draw_objs1.extend(create_top_left_labels("Category", bin_name, 0))
    draw_objs1.extend(create_top_left_labels("Process", proc_label, 1))
    if plot_syst:
        draw_objs1.extend(create_top_left_labels("Systematic", param["name"], 2))

    # draw the ratio plot
    if plot_syst:
        # clone histograms and normalize
        def norm(h):
            h.Add(hist_n_trans, -1.)
            h.Divide(hist_n_trans)
            h.Scale(100.)

        hist_n_trans2 = hist_n_trans.Clone(hist_n_trans.GetName() + "2")
        hist_d_trans2 = hist_d_trans.Clone(hist_d_trans.GetName() + "2")
        hist_u_trans2 = hist_u_trans.Clone(hist_u_trans.GetName() + "2")
        norm(hist_n_trans2)
        norm(hist_d_trans2)
        norm(hist_u_trans2)

        # set y limits
        no_yrange2_set = y_min2 is None and y_max2 is None
        if y_min2 is None:
            y_min2 = min(hist_d_trans2.GetMinimum(), hist_u_trans2.GetMinimum())
            y_min2 = min(-0.059, max(-59., y_min2 * 1.5))
        if y_max2 is None:
            y_max2 = max(hist_d_trans2.GetMaximum(), hist_u_trans2.GetMaximum())
            y_max2 = max(0.059, min(59., y_max2 * 1.5))
        # when no limit was requested, ensure it is symmetric
        if no_yrange2_set:
            y_min2 = min(y_min2, -y_max2)
            y_max2 = max(y_max2, -y_min2)
        h_dummy2.SetMinimum(y_min2)
        h_dummy2.SetMaximum(y_max2)

        # add to plots
        draw_objs2.append((hist_d_trans2, "SAME,HIST"))
        draw_objs2.append((hist_u_trans2, "SAME,HIST"))
        draw_objs2.append((hist_n_trans2, "SAME,HIST"))

    # legend
    legend = r.routines.create_legend(pad=pad1, width=250, n=len(legend_entries))
    r.fill_legend(legend, legend_entries)
    draw_objs1.append(legend)
    legend_box = r.routines.create_legend_box(legend, pad1, "tr",
        props={"LineWidth": 0, "FillColor": colors.root.white_trans_70})
    draw_objs1.insert(-1, legend_box)

    # cms label
    cms_labels = r.routines.create_cms_labels(pad=pad1, layout="outside_horizontal")
    draw_objs1.extend(cms_labels)

    # campaign label
    if campaign_label:
        campaign_label = to_root_latex(campaign_label)
        campaign_label = r.routines.create_top_right_label(campaign_label, pad=pad1)
        draw_objs1.append(campaign_label)

    # draw all objects
    pad1.cd()
    r.routines.draw_objects(draw_objs1)
    if plot_syst:
        pad2.cd()
        r.routines.draw_objects(draw_objs2)

    # save
    r.update_canvas(canvas)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    canvas.SaveAs(path)


def transform_binning(hist, binning):
    def clone_hist(*args, **kwargs):
        clone = hist.__class__(
            hist.GetName() + "__transformed",
            ";".join([hist.GetTitle(), hist.GetXaxis().GetTitle(), hist.GetYaxis().GetTitle()]),
            *args, **kwargs
        )
        clone.Sumw2()
        return clone

    if binning == "numbers":
        n_bins = hist.GetNbinsX()
        x_min = 0.5
        x_max = n_bins + 0.5
        clone = clone_hist(n_bins, x_min, x_max)

        # fill values
        for b in range(1, n_bins + 1):
            clone.SetBinContent(b, hist.GetBinContent(b))
            clone.SetBinError(b, hist.GetBinError(b))

    elif binning == "numbers_width":
        n_bins = hist.GetNbinsX()
        x_min = 0.5
        x_max = n_bins + 0.5
        clone = clone_hist(n_bins, x_min, x_max)

        # fill values
        for b in range(1, n_bins + 1):
            clone.SetBinContent(b, hist.GetBinContent(b) / hist.GetBinWidth(b))
            clone.SetBinError(b, hist.GetBinError(b) / hist.GetBinWidth(b))

    else:
        raise NotImplementedError

    return clone


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read")
    parser.add_argument("rules", nargs="+", metavar="BIN,PROCESS[,SYSTEMATIC]", help="rules "
        "defining which shapes to plot; 'BIN', 'PROCESS' and 'SYSTEMATIC' support patterns where a "
        "prepended '!' negates their meaning; special process names are 'S', 'B', and 'SB' which "
        "are interpreted as combined signal, background, and signal+background; multiple process "
        "patterns can be concatenated with '+'; when no 'SYSTEMATIC' is given, only nominal shapes "
        "are plotted; special systematic names 'S' and 'R' are interpreted as all shape and all "
        "rate systematics; this parameter also supports files that contain the rules in the "
        "described format line by line")
    parser.add_argument("--stack", "-s", action="store_true", help="instead of creating separate "
        "plots per process machted by a rule, stack distributions and create a single plot")
    parser.add_argument("--directory", "-d", default=".", help="directory in which produced plots "
        "are saved; defaults to the current directory")
    parser.add_argument("--nom-format", default="{bin}__{process}.pdf", help="format for created "
        "files when creating only nominal shapes; default: {bin}__{process}.pdf")
    parser.add_argument("--syst-format", default="{bin}__{process}__{syst}.pdf", help="format for "
        "created files when creating systematic shapes; default: {bin}__{process}__{syst}.pdf")
    parser.add_argument("--mass", "-m", default="125", help="mass hypothesis; default: 125")
    parser.add_argument("--binning", "-b", default="original", choices=["original", "numbers",
        "numbers_width"], help="the binning strategy; 'original': use original bin edges; "
        "'numbers': equidistant edges using bin numbers; 'numbers_width': same as 'numbers' and "
        "divide by bin widths")
    parser.add_argument("--x-title", default="Datacard shape", help="x-axis label; default: "
        "'Datacard shape'")
    parser.add_argument("--y-min", type=float, help="min y value of the top pad; no default")
    parser.add_argument("--y-max", type=float, help="max y value of the top pad; no default")
    parser.add_argument("--y-min2", type=float, help="min y value of the bottom pad; no default")
    parser.add_argument("--y-max2", type=float, help="max y value of the bottom pad; no default")
    parser.add_argument("--y-log", action="store_true", help="transform y-axis to log scale")
    parser.add_argument("--campaign", help="label to be shown at the top right; no default")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    parser.add_argument("--log-name", default=logger.name, help="name of the logger on the command "
        "line; default: {}".format(logger.name))
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # run the renaming
    with patch_object(logger, "name", args.log_name):
        plot_datacard_shapes(args.input, args.rules, stack=args.stack, directory=args.directory,
            nom_format=args.nom_format, syst_format=args.syst_format, mass=args.mass,
            binning=args.binning, x_title=args.x_title, y_min=args.y_min, y_max=args.y_max,
            y_min2=args.y_min2, y_max2=args.y_max2, y_log=args.y_log, campaign_label=args.campaign)
