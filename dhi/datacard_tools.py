# coding: utf-8

"""
Helpers to manipulate and work with datacards and shape files.
"""

import os
import shutil
import contextlib
from collections import OrderedDict

import law

from dhi.util import import_ROOT, multi_match


def read_datacard_blocks(datacard):
    """
    Reads the content of a *datacard*, divides the lines into blocks named "preamble", "counts",
    "shapes", "observation", "rates", "parameters", "groups", "mc_stats", "nuisance_edits", and
    "unknown". These blocks are returned in an ordered dictionary for further inspection.
    """
    # create the returned mapping
    fields = [
        "preamble", "counts", "shapes", "observation", "rates", "parameters", "groups", "mc_stats",
        "nuisance_edits", "unknown",
    ]
    blocks = OrderedDict((field, []) for field in fields)

    # read lines
    datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(datacard)))
    with open(datacard, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip().lstrip("- ")
            if line and not line.startswith("#"):
                lines.append(line)

    # store and remove preamble, i.e., everything before {i,j,k}max
    for preamble_offset, line in enumerate(lines):
        if line.startswith(("imax", "jmax", "kmax")):
            break
    else:
        raise Exception("datacard {} contains no counter section (imax|jmax|kmax)".format(datacard))
    blocks["preamble"].extend(lines[:preamble_offset])
    del lines[:preamble_offset]

    # trace interdependent lines describing observations
    for obs_offset in range(len(lines) - 1):
        line = lines[obs_offset]
        if not line.startswith("bin "):
            continue
        next_line = lines[obs_offset + 1]
        if next_line.startswith("observation "):
            blocks["observation"].extend([line, next_line])
            del lines[obs_offset:obs_offset + 2]
            break

    # trace interdependent lines describing process rates
    for rate_offset in range(len(lines) - 3):
        line = lines[rate_offset]
        if not line.startswith("bin "):
            continue
        next_lines = lines[rate_offset + 1:rate_offset + 4]
        if next_lines[0].startswith("process ") and next_lines[1].startswith("process ") \
                and next_lines[2].startswith("rate "):
            blocks["rates"].extend([line] + next_lines)
            del lines[rate_offset:rate_offset + 4]
            break

    # go through lines one by one and assign to blocks based on directive names
    param_directives = ["lnN", "lnU", "gmN", "shape*", "discrete", "param", "rateParam", "extArgs"]
    for line in lines:
        words = line.split()
        if words[0] in ("imax", "jmax", "kmax"):
            field = "counts"
        elif words[0] == "shapes":
            field = "shapes"
        elif words[0] == "autoMCStats":
            field = "mc_stats"
        elif len(words) >= 2 and words[0] == "nuisance" and words[1] == "edit":
            field = "nuisance_edits"
        elif words[1] == "group":
            field = "groups"
        elif multi_match(words[1], param_directives):
            field = "parameters"
        else:
            # the above cases should match every line but still store unknown cases
            field = "unknown"
        blocks[field].append(line)

    return blocks


@contextlib.contextmanager
def manipulate_datacard(datacard, target_datacard=None, read_only=False):
    """
    Context manager that opens a *datacard* and yields its contents as a dictionary of specific
    content blocks as returned by :py:func:`read_datacard_blocks`. Each block is a list of lines
    which can be updated in-place to make changes to the datacard. When a *target_datacard* is
    defined, the changes are saved in a new datacard at this location and the original datacard
    remains unchanged. When no changes are to be made to the datacard, you may set *read_only* to
    *True* to disable the tracking of changes. Just note that the *target_datacard* is still written
    when given. Example:

    .. code-block:: python

        # add a new (rate) parameter to the datacard and remove autoMCStats
        with manipulate_datacard("datacard.txt") as content:
            content["parameters"].append("beta rateParam B bkg 50")
            del content["mc_stats"]
    """
    # read the datacard content in blocks
    datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(datacard)))
    blocks = read_datacard_blocks(datacard)

    # yield blocks and keep track of changes via hashes
    hash_before = None if read_only else law.util.create_hash(blocks)
    yield blocks
    hash_after = None if read_only else law.util.create_hash(blocks)
    has_changes = hash_after != hash_before

    # prepare the target location when given
    if target_datacard:
        target_datacard = os.path.abspath(os.path.expandvars(os.path.expanduser(target_datacard)))
        target_dirname = os.path.dirname(target_datacard)
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)

    # handle saving of changes
    if has_changes:
        with open(target_datacard or datacard, "w") as f:
            for field, lines in blocks.items():
                if lines:
                    f.write("\n".join(lines) + "\n")
                    if field in ["counts", "shapes", "observation", "rates"]:
                        f.write(80 * "-" + "\n")

    elif target_datacard:
        # no changes, just copy the original file
        shutil.copy2(datacard, target_datacard)


def extract_shape_files(datacard, absolute=True, resolve=True, skip=("FAKE",)):
    """
    Extracts all unique paths declared as shape files in a *datacard* and returns them in a list.
    When *absolute* is *True*, the extracted paths are made absolute based on the location of the
    datacard itself. When both *absolute* and *resolve* are *True*, symbolic links are resolved.
    *skip* can be a sequence of shape file paths or patterns to skip.
    """
    # read shape lines and extract file paths
    shape_files = []
    with manipulate_datacard(datacard) as content:
        for line in content["shapes"]:
            # the shape file is the 4th part
            parts = line.split()
            if len(parts) < 4:
                continue
            shape_file = parts[3]
            # skip fake files
            if multi_match(shape_file, skip):
                continue
            shape_files.append(shape_file)

    # convert to absolute paths (when paths are already absolute, os.path.join will not change them)
    if absolute:
        dirname = os.path.dirname(datacard)
        shape_files = [os.path.join(dirname, f) for f in shape_files]

        if resolve:
            shape_files = [os.path.realpath(f) for f in shape_files]

    # make them unique
    shape_files = law.util.make_unique(shape_files)

    return shape_files


def update_shape_files(func, datacard, target_datacard=None, skip=("FAKE",)):
    """
    Updates the shape files in a *datacard* according to a configurable function *func* that should
    accept the shape file location, the process name, the channel name and optionally the nominal
    and systematic histogram extraction patterns that are usually defined in each shape line. When
    a *target_datacard* is given, the updated datacard is stored at this path rather than the
    original one. *skip* can be a sequence of shape file paths or patterns to skip. Example:

    .. code-block:: python

        def func(shape_file, process, channel, *patterns):
            return shape_file.replace(".root", "_new.root")

        update_shape_files(func, "datacard.txt")
    """
    # use the content manipulation helper
    with manipulate_datacard(datacard, target_datacard=target_datacard) as content:
        # iterate through shape lines and change them in-place
        for i, line in enumerate(content["shapes"]):
            parts = line.split()
            if len(parts) < 4:
                continue

            # extract fields
            prefix = parts.pop(0)  # usually "shapes"
            process = parts.pop(0)
            channel = parts.pop(0)
            shape_file = parts.pop(0)
            patterns = parts

            # skip certain files
            if multi_match(shape_file, skip):
                continue

            # run the func
            new_shape_file = func(shape_file, process, channel, *patterns)

            # update if needed
            if new_shape_file != shape_file:
                new_line = " ".join([prefix, process, channel, new_shape_file] + patterns)
                content["shapes"][i] = new_line


def get_workspace_parameters(workspace, workspace_name="w", config_name="ModelConfig"):
    """
    Takes a workspace stored in a ROOT file *workspace* with the name *workspace_name* and gathers
    information on all non-constant parameters. The return value is an ordered dictionary that maps
    parameter names to dicts with fields ``name``, ``type``, ``groups`` and ``prefit``. The
    functionality is loosely based on ``CombineHarvester.CombineTools.combine.utils``.
    """
    ROOT = import_ROOT()

    # read the workspace
    workspace = os.path.expandvars(os.path.expanduser(workspace))
    f = ROOT.TFile.Open(workspace)
    w = f.Get(workspace_name)

    # get all model parameters
    config = w.genobj(config_name)
    all_params = config.GetPdf().getParameters(config.GetObservables())

    # iteration helper since ROO parameter lists yield no iterator through bindings
    def iterate(iterable):
        it = iterable.createIterator()
        while True:
            obj = it.Next()
            if obj:
                yield obj
            else:
                break

    # loop through parameters and select
    params = OrderedDict()
    for param in iterate(all_params):
        if not isinstance(param, ROOT.RooRealVar) or param.isConstant():
            continue

        # get the type of the pdf
        pdf = w.pdf("{}_Pdf".format(param.GetName()))
        if pdf is None or isinstance(pdf, ROOT.RooUniform):
            pdf_type = "Unconstrained"
        elif isinstance(pdf, ROOT.RooGaussian):
            pdf_type = "Gaussian"
        elif isinstance(pdf, ROOT.RooPoisson):
            pdf_type = "Poisson"
        elif isinstance(pdf, ROOT.RooBifurGauss):
            pdf_type = "AsymmetricGaussian"
        else:
            pdf_type = "Unrecognised"

        # get groups
        start = "group_"
        groups = [attr.replace(start, "") for attr in param.attributes() if attr.startswith(start)]

        # prefit values
        nom = param.getVal()
        if pdf_type == "Unconstrained":
            prefit = [nom, nom, nom]
        else:
            prefit = [nom + param.getErrorLo(), nom, nom + param.getErrorHi()]

        # store it
        params[param.GetName()] = {
            "name": param.GetName(),
            "type": pdf_type,
            "groups": groups,
            "prefit": prefit,
        }

    # cleanup
    f.Close()

    return params
