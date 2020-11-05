# coding: utf-8

"""
Helpers to manipulate and work with datacards and shape files.
"""

import os
import shutil
import contextlib
import logging
import tempfile
import copy
from optparse import OptionParser
from collections import OrderedDict, defaultdict

import law

from dhi.util import import_ROOT, real_path, multi_match, copy_no_collisions


#: Parameter directives excluding groups, autoMCStats and nuisace edit lines.
parameter_directives = [
    "lnN", "lnU", "gmN", "trG", "unif", "dFD", "dFD2", "constr", "shape*", "discrete", "param",
    "rateParam", "flatParam", "extArgs",
]

#: Parameter directives that are configured per bin and process column.
columnar_parameter_directives = ["lnN", "lnU", "gmN", "shape*"]


class DatacardRenamer(object):

    @classmethod
    def parse_rules(cls, rules):
        """
        Parses a list of translation *rules* (strings) which should either have the format
        ``old=new`` or refer to files containing rules in this format line by line. Returns a list
        of 2-tuples ``(old, new)``.
        """
        pairs = []

        for rule_or_path in rules:
            # first try to interpret it as a file
            path = real_path(rule_or_path)
            if not os.path.exists(path):
                # not a file, use as is
                _rules = [rule_or_path]
            else:
                # read the file line by line, accounting for empty lines and comments
                _rules = []
                with open(path, "r") as f:
                    for line in f.readlines():
                        rule = line.split("#", 1)[0].strip()
                        if rule:
                            _rules.append(rule)
            # split rules
            for rule in _rules:
                if rule.count("=") != 1:
                    raise ValueError("invalid rule {}, must contain exactly one '='".format(rule))
                pairs.append(rule.strip().split("=", 1))

        return pairs

    def __init__(self, datacard, rules, directory=None, skip_shapes=False, logger=None):
        super(DatacardRenamer, self).__init__()

        # store attributes
        self._datacard_orig = datacard
        self.datacard = real_path(datacard)
        self._rules_orig = rules
        self.rules = None
        self.skip_shapes = skip_shapes
        self.logger = logger or logging.getLogger("{}_{}".format(
            self.__class__.__name__, hex(id(self))))

        # datacard object if required
        self._dc = None

        # setup file and object caches
        self._tmpfile_cache = {}
        self._tfile_cache = {}
        self._tobj_input_cache = defaultdict(dict)
        self._tobj_output_cache = defaultdict(dict)

        # validate renaming rules right away
        self._validate_rules()

        # when a directory is given, run the bundling
        if directory:
            self._bundle_files(directory)

    def _clear_caches(self):
        self._tmpfile_cache.clear()
        self._tfile_cache.clear()
        self._tobj_input_cache.clear()
        self._tobj_output_cache.clear()

    def _validate_rules(self):
        rules = self.parse_rules(self._rules_orig)
        old_names = [name for name, _ in rules]
        for rule in rules:
            # rules must have length 2
            if len(rule) != 2:
                raise ValueError("translation rule {} invalid, must have length 2".format(rule))
            old_name, new_name = rule

            # old names must be unique
            if old_names.count(old_name) > 1:
                raise ValueError("old process name {} not unique in translationion rules".format(
                    old_name))

            # new name must not be in old names
            if new_name in old_names:
                raise ValueError("new process name {} must not be in old process names".format(
                    new_name))

        # store in the dictionary
        self.rules = OrderedDict(map(tuple, rules))

    def _bundle_files(self, directory):
        self.logger.info("bundle datacard files into directory {}".format(directory))
        self.datacard = bundle_datacard(self.datacard, directory, skip_shapes=self.skip_shapes)

    @property
    def dc(self):
        if self._dc is None:
            self._dc = create_datacard_instance(self.datacard)
        return self._dc

    def has_rule(self, name):
        return name in self.rules

    def translate(self, name):
        return self.rules[name]

    def get_bin_process_pairs(self):
        pairs = []
        for bin_name in self.dc.exp:
            for process_name in self.dc.exp[bin_name]:
                pairs.append((bin_name, process_name))

        return pairs

    def get_bin_process_to_systs_mapping(self):
        shape_syst_names = defaultdict(list)

        # determine shape systematic names per (bin, process) pair
        for syst_name, _, syst_type, _, syst_data in self.dc.systs:
            if not syst_type.startswith("shape"):
                continue
            for bin_name, bin_syst_data in syst_data.items():
                for process_name, syst_effect in bin_syst_data.items():
                    if syst_effect:
                        key = (bin_name, process_name)
                        if syst_name in shape_syst_names[key]:
                            self.logger.warning("shape systematic {} appears more than once for "
                                "bin {} and process {}".format(syst_name, *key))
                        else:
                            shape_syst_names[key].append(syst_name)

        return shape_syst_names

    def make_tmpfile(self, abs_path):
        abs_path = real_path(abs_path)

        if abs_path not in self._tmpfile_cache:
            suffix = "_" + os.path.basename(abs_path)
            self._tmpfile_cache[abs_path] = tempfile.mkstemp(suffix=suffix)[1]

            self.logger.debug("creating tmp file {} from {}".format(
                self._tmpfile_cache[abs_path], abs_path))
            shutil.copy2(abs_path, self._tmpfile_cache[abs_path])

        return self._tmpfile_cache[abs_path]

    def open_tfile(self, abs_path, *args):
        ROOT = import_ROOT()

        abs_path = real_path(abs_path)

        if abs_path not in self._tfile_cache:
            self.logger.debug("opening file {}".format(abs_path))
            self._tfile_cache[abs_path] = ROOT.TFile(abs_path, *args)

        return self._tfile_cache[abs_path]

    def get_tobj(self, abs_path, obj_name, write=False, *args):
        ROOT = import_ROOT()

        cache = self._tobj_output_cache if write else self._tobj_input_cache
        tfile = abs_path if isinstance(abs_path, ROOT.TFile) else self.open_tfile(abs_path, *args)

        if obj_name not in cache[tfile]:
            self.logger.debug("loading object {} from file {}".format(obj_name, tfile.GetPath()))
            cache[tfile][obj_name] = tfile.Get(obj_name)

        return cache[tfile][obj_name]

    @contextlib.contextmanager
    def start(self):
        ROOT = import_ROOT()

        # clear all caches
        self._clear_caches()

        # yield the context and handle errors
        error = True
        try:
            with manipulate_datacard(self.datacard) as content:
                yield content
            error = False
        except BaseException as e:
            self.logger.error("an exception of type {} occurred while renaming the datacard".format(
                e.__class__.__name__))
            raise
        finally:
            # write all output tobjs
            n_tobjs = sum(len(v) for v in self._tobj_output_cache.values())
            if not error and n_tobjs:
                self.logger.info("writing {} output object(s)".format(n_tobjs))
                ignore_level_orig = ROOT.gROOT.ProcessLine("gErrorIgnoreLevel;")
                ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kFatal;")
                for f in self._tobj_output_cache:
                    f.cd()
                    for tobj in self._tobj_output_cache[f].values():
                        self.logger.debug("writing object {} to file {}".format(
                            tobj.GetName(), f.GetPath()))
                        tobj.Write()
                ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = {};".format(ignore_level_orig))

            # close all open files
            if self._tfile_cache:
                self.logger.info("closing {} file(s)".format(len(self._tfile_cache)))
                for f in self._tfile_cache.values():
                    if f and f.IsOpen():
                        self.logger.debug("closing file {}".format(f.GetPath()))
                        f.Close()

            # move tmp files back to initial locations
            if not error and self._tmpfile_cache:
                self.logger.info("moving {} temporary file(s)".format(len(self._tmpfile_cache)))
                for path, tmp_path in self._tmpfile_cache.items():
                    self.logger.debug("moving back tmp file {} to {}".format(tmp_path, path))
                    shutil.move(tmp_path, path)

            # clear caches again
            self._clear_caches()


class ShapeLine(object):

    @classmethod
    def parse(cls, line):
        parts = line.split("#")[0].split()
        if len(parts) < 5:
            raise Exception("invalid shape line format: {}".format(line))

        return (parts + [None])[1:6]

    def __init__(self, line, i):
        super(ShapeLine, self).__init__()

        # parse the line
        p, b, f, n, s = self.parse(line)

        # set attributes
        self.i = i
        self.process = p
        self.bin = b
        self.file = f
        self.nom_pattern = n
        self.syst_pattern = s

    def __str__(self):
        parts = ["shapes", self.process, self.bin, self.file, self.nom_pattern, self.syst_pattern]
        return " ".join(filter(bool, parts))

    @property
    def sorting_weight(self):
        w = self.i
        if self.bin == "*":
            w += 100000
        if self.process == "*":
            w += 1000000
        return w

    def copy(self):
        return copy.copy(self)


def create_datacard_instance(datacard, create_shape_builder=False):
    """
    Parses a *datacard* using ``HiggsAnalysis.CombinedLimit.DatacardParser.parseCard`` and returns a
    ``HiggsAnalysis.CombinedLimit.Datacard.Datacard`` instance. When *create_shape_builder* is
    *True*, a ``HiggsAnalysis.CombinedLimit.ShapeTools.ShapeBuilder`` is built as well and returned
    as the second item in a 2-tuple.
    """
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard, addDatacardParserOptions
    from HiggsAnalysis.CombinedLimit.ShapeTools import ShapeBuilder

    # create a dummy option parser
    parser = OptionParser()
    addDatacardParserOptions(parser)
    options = parser.parse_args([])[0]

    # patch some options
    options.fileName = datacard

    # create the datacard object
    datacard = real_path(datacard)
    with open(datacard, "r") as f:
        dc = parseCard(f, options)

    # optionally create the shape builder
    if create_shape_builder:
        sb = ShapeBuilder(dc, options)

    return (dc, sb) if create_shape_builder else dc


def read_datacard_blocks(datacard):
    """
    Reads the content of a *datacard* and divides the lines into blocks named "preamble", "counts",
    "shapes", "observation", "rates", "parameters", "groups", "auto_mc_stats", "nuisance_edits", and
    "unknown". These blocks are returned in an ordered dictionary for further inspection.
    """
    # create the returned mapping
    fields = [
        "preamble", "counts", "shapes", "observation", "rates", "parameters", "groups",
        "auto_mc_stats", "nuisance_edits", "unknown",
    ]
    blocks = OrderedDict((field, []) for field in fields)

    # read lines
    datacard = real_path(datacard)
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
    for line in lines:
        words = line.split()
        field = "unknown"
        if words[0] in ("imax", "jmax", "kmax"):
            field = "counts"
        elif words[0] == "shapes":
            field = "shapes"
        elif len(words) >= 2:
            if words[1] == "autoMCStats":
                field = "auto_mc_stats"
            elif words[1] == "group":
                field = "groups"
            elif words[1] == "edit" and words[0] == "nuisance":
                field = "nuisance_edits"
            elif multi_match(words[1], parameter_directives):
                field = "parameters"
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
    *True* to disable the tracking of changes. However, please note that the *target_datacard* is
    still written when given. Example:

    .. code-block:: python

        # add a new (rate) parameter to the datacard and remove autoMCStats
        with manipulate_datacard("datacard.txt") as content:
            content["parameters"].append("beta rateParam B bkg 50")
            del content["auto_mc_stats"][:]
    """
    # read the datacard content in blocks
    datacard = real_path(datacard)
    blocks = read_datacard_blocks(datacard)

    # yield blocks and keep track of changes via hashes
    hash_before = None if read_only else law.util.create_hash(blocks)
    yield blocks
    hash_after = None if read_only else law.util.create_hash(blocks)
    has_changes = hash_after != hash_before

    # prepare the target location when given
    if target_datacard:
        target_datacard = real_path(target_datacard)
        target_dirname = os.path.dirname(target_datacard)
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)

    # handle saving of changes
    if has_changes:
        with open(target_datacard or datacard, "w") as f:
            for field, lines in blocks.items():
                if not lines:
                    continue
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
            shape_files = [real_path(f) for f in shape_files]

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


def bundle_datacard(datacard, directory, skip_shapes=False):
    """
    Takes a *datacard* given by its path, copies it as well as the shape files it refers to a new
    location given by *directory* and updates the shape lines accordingly to be relative paths.
    When *skip_shapes* is *True*, only the datacard is copied. The path to the new datacard is
    returned.
    """
    # prepare the directory
    directory = real_path(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # copy the card itself
    src_datacard = real_path(datacard)
    dst_datacard = os.path.join(directory, os.path.basename(src_datacard))
    shutil.copy2(src_datacard, dst_datacard)

    # copy shape files and update the datacard shape lines
    copied_files = {}
    def update_and_copy(shape_file, *_):
        abs_shape_file = os.path.join(os.path.dirname(src_datacard), shape_file)
        if abs_shape_file not in copied_files:
            if skip_shapes:
                copied_files[abs_shape_file] = abs_shape_file
            else:
                copied_files[abs_shape_file] = copy_no_collisions(abs_shape_file, directory)
        return os.path.basename(copied_files[abs_shape_file])

    update_shape_files(update_and_copy, dst_datacard)

    return dst_datacard


def update_shape_name(towner, old_name, new_name):
    """
    Renames a shape object (a ROOT histogram or RooFit PDF) contained in an owner object
    *towner* (a ROOT file or ROOFit workspace) from *old_name* to *new_name*. When the object to
    rename is a RooFit PDF, its normalization formula is renamed also as required by combine.
    """
    if towner.InheritsFrom("TDirectoryFile"):
        # strategy: get the object, make a copy with the new name, delete all cycles of the old
        # object and write the new one
        tobj_orig = towner.Get(old_name)
        if not tobj_orig:
            raise Exception("no object named {} found in {}".format(old_name, towner))

        towner.cd()
        tobj_clone = tobj_orig.Clone(new_name)
        tobj_clone.SetTitle(tobj_clone.GetTitle().replace(old_name, new_name))
        towner.Delete(old_name + ";*")
        tobj_clone.Write(new_name)

    elif towner.InheritsFrom("RooWorkspace"):
        # strategy: get the pdf and optional norm object, simply rename them
        pdf = towner.pdf(old_name)
        if not pdf:
            if not pdf:
                raise Exception("no pdf named {} found in {}".format(old_name, towner))
        pdf.SetName(new_name)
        pdf.SetTitle(pdf.GetTitle().replace(old_name, new_name))

        old_norm_name = old_name + "_norm"
        new_norm_name = new_name + "_norm"
        norm = towner.arg(old_norm_name)
        if norm:
            norm.SetName(new_norm_name)
            norm.SetTitle(norm.GetTitle().replace(old_norm_name, new_norm_name))

    else:
        raise NotImplementedError("cannot extract shape from {} object for updating".format(
            towner.ClassName()))


def expand_variables(s, process=None, channel=None, systematic=None, mass=None):
    """
    Expands variables $PROCESS, $CHANNEL, $SYSTEMATIC, $MASS in a string *s* and returns it.
    """
    if process is not None:
        s = s.replace("$PROCESS", process)
    if channel is not None:
        s = s.replace("$CHANNEL", channel)
    if systematic is not None:
        s = s.replace("$SYSTEMATIC", systematic)
    if mass is not None:
        s = s.replace("$MASS", mass)
    return s


def get_workspace_parameters(workspace, workspace_name="w", config_name="ModelConfig"):
    """
    Takes a workspace stored in a ROOT file *workspace* with the name *workspace_name* and gathers
    information on all non-constant parameters. The return value is an ordered dictionary that maps
    parameter names to dicts with fields ``name``, ``type``, ``groups`` and ``prefit``. The
    functionality is loosely based on ``CombineHarvester.CombineTools.combine.utils``.
    """
    ROOT = import_ROOT()

    # read the workspace
    workspace = real_path(workspace)
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
