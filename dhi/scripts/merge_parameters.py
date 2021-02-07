#!/usr/bin/env python
# coding: utf-8

"""
Script to merge multiple (nuisance) parameters of the same type into a new,
single one. Currently, only parameters with columnar type "lnN", "lnU" and
"shape" are supported. Example usage:

# merge two parameters
> merge_parameters.py datacard.txt CMS_eff_m CMS_eff_m_iso CMS_eff_m_id -d output_directory

# merge parameters via fnmatch wildcards (note the quotes)
> merge_parameters.py datacard.txt CMS_eff_m 'CMS_eff_m_*' -d output_directory

Note 1: The use of an output directory is recommended to keep input files
        unchanged.

Note 2: This script is not intended to be used to merge incompatible systematic
        uncertainties. Its only purpose is to reduce the number of parameters by
        merging the effect of (probably small) uncertainties that are related at
        analysis level, e.g. multiple types of lepton efficiency corrections.
        Please refer the doc string of "merge_parameters()" for more info.
"""

import os

from dhi.datacard_tools import (
    columnar_parameter_directives, bundle_datacard, manipulate_datacard, update_datacard_count,
    expand_variables, expand_file_lines, ShapeLine,
)
from dhi.util import import_ROOT, real_path, multi_match, create_console_logger, TFileCache


logger = create_console_logger(os.path.splitext(os.path.basename(__file__))[0])


def merge_parameters(datacard, new_name, patterns, directory=None, skip_shapes=False,
        flip_parameters=None, auto_rate_flip=False, auto_rate_max=False, auto_rate_envelope=False,
        auto_shape_average=False, auto_shape_envelope=False, digits=4, mass="125"):
    """
    Reads a *datacard* and merges parameters given by a list of *patterns* into a new, single
    parameter *new_name*. A pattern can be a parameter name, a pattern that is matched via fnmatch,
    or a file containing patterns. The matched parameters are required to have the exact same type.
    Currently, only parameters with columnar type "lnN", "lnU" and "shape" are supported.
    *flip_parameters* can be a list of names or patterns matching parameters whose effect should be
    flipped, i.e., the effect of its up and down variation will be interchanged.

    The combination of uncertainties is usually not trivial and situations where no standard recipe
    exists occur quite often. In these cases, the default behavior of this function is to raise an
    exception unless explicitly configured not to do so. There are three cases handled for "lnN" and
    "lnU", and two cases for "shape" parameters.

    1. lnN/lnU: Multiple asymmetric rate uncertainties are not allowed to have effects pointing in
       different directions, leading to situations where a positive and a negative uncertainty are
       to be combined. When *auto_rate_flip* is *True*, the effects are automatically interchanged
       so that a combination makes sense.

    2. lnN/lnU: Asymmetric rate uncertainty with both variations pointing in the same direction
       (one-sided effects) are not allowed. When *auto_rate_max* is *True*, the maximum value will
       be used in the combination as it is likely to be preferred by the likelihood.

    3. lnN/lnU: When *auto_rate_envelope* is *True*, uncertainties are not combined by means of
       uncorrelated error prpagation, but the envelope is constructed instead. This intrinsically
       prevents errors described in 1. and 2. above.

    1. shape: The merging of shapes is done bin-wise with the uncertainty defined by each bin being
       propagated by addition in quadrature. When the input shapes contain both negative and
       positive effects in the same bin, there is no consistent way for combining errors and an
       error is raised. When *auto_shape_average* is *True*, positive and negative merged components
       will be determined first, and averaged afterwards. This approach is highly experimental and
       should not be used extensively to merge incompatible systematics which would rather need more
       insight on analysis level.

    2. shape: When *auto_shape_envelope* is *True*, shapes are not combined by means of uncorrelated
       error propagation, but the envelopes for up and down variations are constructed instead. This
       intrinsically prevents errors described in 1. above.

    The merged effects of "lnN" and "lnU" parameters are rounded using *digits*. When *directory* is
    *None*, the input *datacard* is updated in-place. Otherwise, both the changed datacard and all
    the shape files it refers to are stored in the specified directory. For consistency, this will
    also update the location of shape files in the datacard. When *skip_shapes* is *True*, all shape
    files remain unchanged (the shape lines in the datacard itself are still changed).
    """
    ROOT = import_ROOT()

    # store supported types, which must always be a subset of all columnar types
    supported_types = ["lnN", "lnU", "shape"]
    assert(all(multi_match(t, columnar_parameter_directives) for t in supported_types))

    # prepare the datacard path
    datacard = real_path(datacard)

    # expand patterns from files
    patterns = expand_file_lines(patterns)

    # when a directory is given, copy the datacard (and all its shape files when not skipping them)
    # into that directory and continue working on copies
    if directory:
        logger.info("bundle datacard files into directory {}".format(directory))
        datacard = bundle_datacard(datacard, directory, skip_shapes=skip_shapes)

    # start removing
    with manipulate_datacard(datacard) as content:
        # keep track of the filll lines of parameters to be merged as well as their type
        removed_param_lines = []
        new_type = None

        # find parameters to be merged
        if content.get("parameters"):
            to_remove = []
            for i, param_line in enumerate(content["parameters"]):
                param_line = param_line.split()

                # the name must not exist yet
                if param_line and param_line[0] == new_name:
                    raise Exception("a parameter named {} already exists in the datacard".format(
                        new_name))

                # cannot process with less than two line elements
                if len(param_line) < 2:
                    continue

                param_name, param_type = param_line[:2]
                if multi_match(param_name, patterns):
                    if not new_type:
                        if not multi_match(param_type, supported_types):
                            raise Exception("matched parameter {} has type {} which is currently "
                                "not supported".format(param_name, param_type))
                        logger.info("determined type of new parameter {} to be {}".format(new_name,
                            param_type))
                        new_type = param_type
                    elif param_type != new_type:
                        raise Exception("matched parameter {} has type {} which is different than "
                            "the already determined type {}".format(param_name, param_type,
                            new_type))

                    logger.info("found parameter {} to be merged".format(param_name))
                    removed_param_lines.append(param_line)
                    to_remove.append(i)

            # change lines in-place
            lines = [line for i, line in enumerate(content["parameters"]) if i not in to_remove]
            del content["parameters"][:]
            content["parameters"].extend(lines)

        # nothing to do when no parameter was found, this is likely is misconfiguration
        if not removed_param_lines:
            logger.info("patterns {} did not match any parameters".format(patterns))
            return
        removed_param_names = [line[0] for line in removed_param_lines]

        # when the new type is "shape", verify that shape lines are given and sort them
        shape_lines = None
        if new_type == "shape":
            if content.get("shapes"):
                # prepare shape lines that have a systematic pattern and sort them so that most
                # specific ones (i.e. without wildcards) come first
                shape_lines = [ShapeLine(line, j) for j, line in enumerate(content["shapes"])]
                shape_lines = [shape_line for shape_line in shape_lines if shape_line.syst_pattern]
                shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)

            if not shape_lines:
                raise Exception("cannot create merged parameter {} of type {} when datacard does "
                    "not contain shape lines".format(new_name, new_type))

        # check if all param lines have the same length
        unique_lengths = set(len(line) for line in removed_param_lines)
        if len(unique_lengths) != 1:
            raise Exception("the new parameter type {} is columnar, but the found lines have "
                "unequal lengths {}".format(new_type, unique_lengths))
        n_cols = list(unique_lengths)[0] - 2

        # get all bins and processes
        bin_names = content["rates"][0].split()[1:]
        process_names = content["rates"][1].split()[1:]

        # quick check if all lists have the same lengths
        if not (len(bin_names) == len(process_names) == n_cols):
            raise Exception("the number of bin names ({}), process names ({}), and removed "
                "columns {} does not match".format(len(bin_names), len(process_names), n_cols))

        # helper to parse a nuisance effect
        # "-"         -> (0.0,)
        # "1"         -> (1.0,),
        # "0.833/1.2" -> (0.833, 1.2)
        def parse_effect(s, empty_value):
            if s == "-":
                return (empty_value,)
            elif s.count("/") == 1:
                s1, s2 = s.split("/")
                return (float(s1), float(s2))
            else:
                return (float(s),)

        # parse effects
        empty_value = {
            "lnN": 1.,
            "lnU": 1.,
            "shape": 0.,
        }[new_type]
        effects = [
            [parse_effect(line[col + 2], empty_value) for line in removed_param_lines]
            for col in range(n_cols)
        ]

        # loop though bin, process and effects within the context of a tfile cache to handle files
        with TFileCache(logger=logger) as tcache:
            merged_effects = []
            for bin_name, process_name, eff in zip(bin_names, process_names, effects):
                merged_effect = None

                if new_type == "shape":
                    # get the most specific shape line that applies to the bin process pair
                    for shape_line in shape_lines:
                        if multi_match(bin_name, shape_line.bin) and \
                                multi_match(process_name, shape_line.process):
                            break
                    else:
                        raise Exception("no shape line found that matches bin {} and process {} to "
                            "extract shapes".format(bin_name, process_name))

                    # get the shape file and open it
                    shape_file_path = os.path.join(os.path.dirname(datacard), shape_line.file)
                    shape_file = tcache.open_tfile(shape_file_path, "UPDATE")

                    # helper to get a shape
                    def get_shape(name):
                        shape = shape_file.Get(name)
                        if not shape or not isinstance(shape, ROOT.TH1):
                            raise Exception("shape {} not found in shape file {} or is not a "
                                "histogram".format(name, shape_file_path))
                        return shape

                    # get the nominal shape
                    nom_shape_name = expand_variables(shape_line.nom_pattern, process=process_name,
                        channel=bin_name, mass=mass)
                    shape_n = get_shape(nom_shape_name)

                    # get shapes to merge
                    comb_d = []
                    comb_u = []
                    for name, f in zip(removed_param_names, eff):
                        f = f[0]

                        # skip when the effect is zero
                        if not f:
                            continue

                        # get shapes
                        syst_shape_name = expand_variables(shape_line.syst_pattern,
                            process=process_name, channel=bin_name, systematic=name, mass=mass)
                        shape_d = get_shape(syst_shape_name + "Down").Clone()
                        shape_u = get_shape(syst_shape_name + "Up").Clone()

                        # subtract the nominal shape
                        shape_d.Add(shape_n, -1.)
                        shape_u.Add(shape_n, -1.)

                        # possibly scale the effect
                        if f != 1:
                            shape_d.Scale(f)
                            shape_u.Scale(f)

                        # flip variations
                        if flip_parameters and multi_match(name, flip_parameters):
                            shape_d, shape_u = shape_u, shape_d
                            logger.info("manually flipped down and up shapes of parameter {} in "
                                "bin {} and process {}".format(name, bin_name, process_name))

                        # store the shapes
                        comb_d.append(shape_d)
                        comb_u.append(shape_u)

                    # predict the new shape name using the systematic pattern and
                    new_shape_name = expand_variables(shape_line.syst_pattern, process=process_name,
                        channel=bin_name, systematic=new_name, mass=mass)

                    # create the merged shape variations
                    towner = shape_file
                    if "/" in new_shape_name:
                        towner_name, new_shape_name = new_shape_name.rsplit("/", 1)
                        towner = shape_file.Get(towner_name)
                    towner.cd()
                    merged_d = shape_n.Clone(new_shape_name + "Down")
                    merged_u = shape_n.Clone(new_shape_name + "Up")
                    merged_d.Sumw2(False)
                    merged_u.Sumw2(False)

                    for b in range(1, shape_n.GetNbinsX() + 1):
                        # get bin contents, stop when all values are zero
                        diffs_d = [shape.GetBinContent(b) for shape in comb_d]
                        diffs_u = [shape.GetBinContent(b) for shape in comb_u]
                        if all(v == 0 for v in diffs_d + diffs_u):
                            continue

                        # determine the merged effect that is to be added on top of the nominal one
                        diff_d = 0.
                        diff_u = 0.

                        if auto_shape_envelope:
                            # when building the envelope, just pick the maximum / minimum values
                            for v in diffs_d + diffs_u:
                                if v < diff_d <= 0:
                                    diff_d = v
                                elif v > diff_u >= 0:
                                    diff_u = v

                        else:
                            # merge in quadrature, separately for positive and negative components
                            # TODO: maybe go with something like 2.4 in
                            # https://www.slac.stanford.edu/econf/C030908/papers/WEMT002.pdf
                            diffs_d_n = [d for d in diffs_d if d < 0]
                            diffs_d_p = [d for d in diffs_d if d > 0]
                            diffs_u_n = [u for u in diffs_u if u < 0]
                            diffs_u_p = [u for u in diffs_u if u > 0]

                            # abort or warn when signs of effects are mixed and the relative effect
                            # strength is larger than a threshold value
                            tmpl = "found both positive ({}) and negative ({}) effects in bin {} " \
                                "of {} variation for bin {} and process {}"
                            thresh = 0.001  # 0.1 %
                            if min(diffs_d_n + [0]) < -thresh and max(diffs_d_p + [0]) > thresh:
                                msg = tmpl.format(diffs_d_p, diffs_d_n, b, "down", bin_name,
                                    process_name)
                                if not auto_shape_average:
                                    raise Exception(msg)
                                logger.warning(msg + ", averaging is experimental")
                            if min(diffs_u_n + [0]) < -thresh and max(diffs_u_p + [0]) > thresh:
                                msg = tmpl.format(diffs_u_p, diffs_u_n, b, "up", bin_name,
                                    process_name)
                                if not auto_shape_average:
                                    raise Exception(msg)
                                logger.warning(msg + ", averaging is experimental")

                            # merge components separately
                            diff_d_n = -sum((v**2. for v in diffs_d_n), 0)**0.5
                            diff_d_p = sum((v**2. for v in diffs_d_p), 0)**0.5
                            diff_u_n = -sum((v**2. for v in diffs_u_n), 0)**0.5
                            diff_u_p = sum((v**2. for v in diffs_u_p), 0)**0.5

                            # combine components by averaging if necessary
                            diff_d = (
                                (diff_d_n * len(diffs_d_n) + diff_d_p * len(diffs_d_p)) /
                                (len(diffs_d_n) + len(diffs_d_p))
                            )
                            diff_u = (
                                (diff_u_n * len(diffs_u_n) + diff_u_p * len(diffs_u_p)) /
                                (len(diffs_u_n) + len(diffs_u_p))
                            )

                        # add the difference
                        merged_d.SetBinContent(b, merged_d.GetBinContent(b) + diff_d)
                        merged_u.SetBinContent(b, merged_u.GetBinContent(b) + diff_u)
                        # logger.debug("computed down and up variations of {:.3e} and {:.3e} in "
                        #     "bin {} of new shape in bin {} and process {}".format(diff_d, diff_u,
                        #     b, bin_name, process_name))

                    # write them to the file
                    merged_d.Write()
                    merged_u.Write()

                    # set the merged effect
                    merged_effect = 1 if sum(sum(eff, ())) else "-"

                elif new_type in ("lnN", "lnU"):
                    # helpers to convert a value in lnN/U format to a signed uncertainty and back
                    ln2unc = lambda v: v - 1.
                    unc2ln = lambda v: 1. + v
                    rnd = lambda v: "{{:.{}f}}".format(digits).format(v)

                    # consider the merged effect to be symmetric when all effets have only one entry
                    sym = all(len(f) == 1 for f in eff)
                    if sym:
                        # get single uncertainty values
                        uncs = [ln2unc(f[0]) for f in eff]
                        merged_effect = rnd(unc2ln(sum(v**2. for v in uncs)**0.5))
                    else:
                        # get both sets of uncertainties
                        uncs_d = [(ln2unc(f[0]) if len(f) == 2 else -ln2unc(f[0])) for f in eff]
                        uncs_u = [(ln2unc(f[1]) if len(f) == 2 else ln2unc(f[0])) for f in eff]

                        # go through variations and create groups of up and down effects to merge,
                        # under consideration of the auto_rate_* settings
                        comb_d = []
                        comb_u = []
                        for name, d, u in zip(removed_param_names, uncs_d, uncs_u):
                            # manually flip variations
                            if flip_parameters and multi_match(name, flip_parameters):
                                d, u = u, d
                                logger.info("manually flipped down ({}) and up ({}) variations of "
                                    "parameter {} in bin {} and process {}".format(u, d, name,
                                    bin_name, process_name))

                            # when the envelope is constructed, only store the maximum / minimum
                            # value in the corresponding direction
                            if auto_rate_envelope:
                                for v in u, d:
                                    comb, fn = (comb_d, min) if v < 0 else (comb_u, max)
                                    if not comb or v == fn(v, comb[0]):
                                        del comb[:]
                                        comb.append(v)
                                        logger.info("set new {} value of envelope to {} in bin {} "
                                            "and process {}".format(fn.__name__, v, bin_name,
                                            process_name))
                                continue

                            # check the orientation of effects for later combination in quadrature
                            if d <= 0 and u >= 0:
                                # default case, no further action required
                                comb_d.append(d)
                                comb_u.append(u)

                            elif d >= 0 and u <= 0:
                                # flipped case
                                if not auto_rate_flip:
                                    raise Exception("the signs of down ({}) and up ({}) variations "
                                        "of parameter {} in bin {} and process {} are mixed and "
                                        "automatic flipping is not allowed".format(d, u, name,
                                        bin_name, process_name))

                                comb_d.append(u)
                                comb_u.append(d)
                                logger.warning("automatically flipped down ({}) and up ({}) "
                                    "variations of parameter {} in bin {} and process {}".format(d,
                                    u, name, bin_name, process_name))

                            else:
                                # both effects are either negative or positive
                                if not auto_rate_max:
                                    raise Exception("the down ({}) and up ({}) variations of "
                                        "parameter {} in bin {} and process {} are one-sided and "
                                        "automatic maximum selection is not allowed".format(d, u,
                                        name, bin_name, process_name))

                                max_value = d if max(abs(d), abs(u)) == abs(d) else u
                                if max_value > 0:
                                    comb_d.append(0.)
                                    comb_u.append(max_value)
                                else:
                                    comb_d.append(max_value)
                                    comb_u.append(0.)
                                logger.warning("automatically built envelope of down ({}) and up "
                                    "({}) variations of parameter {} in bin {} and process "
                                    "{}".format(d, u, name, bin_name, process_name))

                        # create the merged effect
                        unc_d = -sum(d**2. for d in comb_d)**0.5
                        unc_u = sum(u**2. for u in comb_u)**0.5
                        merged_effect = "{}/{}".format(rnd(unc2ln(unc_d)), rnd(unc2ln(unc_u)))
                else:
                    # this should never happen
                    assert(False)

                # store the merged effect
                merged_effects.append(str(merged_effect))
                logger.debug("computed merged effect for bin {} and process {} as {}".format(
                    bin_name, process_name, merged_effect))

        # add the merged line
        content["parameters"].append(" ".join([new_name, new_type] + merged_effects))
        logger.debug("added merged parameter line for bin {} and process {}".format(bin_name,
            process_name))

        # decrease kmax in counts
        if removed_param_lines:
            # decrement kmax
            update_datacard_count(content, "kmax", 1 - len(removed_param_lines), diff=True,
                logger=logger)


if __name__ == "__main__":
    import argparse

    # setup argument parsing
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("input", metavar="DATACARD", help="the datacard to read and possibly "
        "update (see --directory)")
    parser.add_argument("merged", metavar="MERGED_NAME", help="name of the newly merged parameter")
    parser.add_argument("names", nargs="+", help="names of parameters or files containing "
        "names of parameters line by line to merge; supports patterns")
    parser.add_argument("--directory", "-d", nargs="?", help="directory in which the updated "
        "datacard and shape files are stored; when not set, the input files are changed in-place")
    parser.add_argument("--no-shapes", "-n", action="store_true", help="do not copy shape files to "
        "the output directory when --directory is set")
    parser.add_argument("--flip-parameters", help="comma-separated list of parameters whose effect "
        "should be flipped, i.e., flips effects of up and down variations; supports patterns")
    parser.add_argument("--auto-rate-flip", action="store_true", help="only for lnN and lnU; when "
        "set, up and down variations of a parameter are swapped when they change the rate in the "
        "relative opposite directions; otherwise, an error is raised")
    parser.add_argument("--auto-rate-max", action="store_true", help="only for lnN and lnU; when "
        "set, the maximum effect of a parameter is used when both up and down variation change the "
        "rate in the same direction; otherwise, an error is raised")
    parser.add_argument("--auto-rate-envelope", action="store_true", help="only for lnN and lnU; "
        "when set, the effect on the new parameter is constructed as the envelope of effects of "
        "parameters to merge")
    parser.add_argument("--auto-shape-average", action="store_true", help="only for shape; when "
        "set and shapes to merge contain both positive negative effects in the same bin, propagate "
        "errors separately and then use their average; otherwise, an error is raised")
    parser.add_argument("--auto-shape-envelope", action="store_true", help="only for shape; when "
        "set, the merged shape variations of the new parameter are constructed as the envelopes of "
        "shapes of parameters to merge")
    parser.add_argument("--digits", type=int, default=4, help="the amount of digits for rounding "
        "merged parameters; defaults to 4")
    parser.add_argument("--mass", "-m", default="125", help="mass hypothesis; default: 125")
    parser.add_argument("--log-level", "-l", default="INFO", help="python log level; default: INFO")
    args = parser.parse_args()

    # configure the logger
    logger.setLevel(args.log_level.upper())

    # prepare flipped parameters
    flip_parameters = (args.flip_parameters and args.flip_parameters.split(",")) or []

    # run the merging
    merge_parameters(args.input, args.merged, args.names, directory=args.directory,
        skip_shapes=args.no_shapes, flip_parameters=flip_parameters,
        auto_rate_flip=args.auto_rate_flip, auto_rate_max=args.auto_rate_max,
        auto_rate_envelope=args.auto_rate_envelope, auto_shape_average=args.auto_shape_average,
         auto_shape_envelope=args.auto_shape_envelope, digits=args.digits, mass=args.mass)
