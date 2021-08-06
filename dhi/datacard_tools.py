# coding: utf-8

"""
Helpers to manipulate and work with datacards and shape files.
"""

import os
import re
import shutil
import contextlib
import logging
import copy
from optparse import OptionParser
from collections import OrderedDict, defaultdict

import law
import six

from dhi.util import import_ROOT, real_path, multi_match, copy_no_collisions, TFileCache


#: Parameter directives excluding groups, autoMCStats and nuisace edit lines.
parameter_directives = [
    "lnN",
    "lnU",
    "gmN",
    "trG",
    "unif",
    "dFD",
    "dFD2",
    "constr",
    "shape*",
    "discrete",
    "param",
    "rateParam",
    "flatParam",
    "extArg",
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

        # expand lines in possible files
        rules = expand_file_lines(rules)

        # parse and fill pairs
        pairs = []
        for rule in rules:
            if rule.count("=") != 1:
                raise ValueError("invalid rule {}, must contain exactly one '='".format(rule))
            pairs.append(rule.strip().split("=", 1))

        return pairs

    def __init__(self, datacard, rules=None, directory=None, skip_shapes=False, logger=None):
        super(DatacardRenamer, self).__init__()

        # store attributes
        self._datacard_orig = datacard
        self.datacard = real_path(datacard)
        self.rules = rules
        self.skip_shapes = skip_shapes
        self.logger = logger or logging.getLogger(
            "{}_{}".format(self.__class__.__name__, hex(id(self))))

        # datacard object if required
        self._dc = None

        # setup file and object caches
        self._tfile_cache = TFileCache(logger=self.logger)
        self._tobj_input_cache = defaultdict(dict)
        self._tobj_output_cache = defaultdict(dict)

        # when a directory is given, run the bundling
        if directory:
            self._bundle_files(directory)

    def _clear_caches(self):
        self._tobj_input_cache.clear()
        self._tobj_output_cache.clear()
        self._tfile_cache._clear()

    def _expand_and_validate_rules(self, rules, old_names=None):
        # check rule formats
        rules = self.parse_rules(rules)

        # expand patterns
        if old_names:
            _rules = []
            for old_name, new_name in rules:
                is_pattern = old_name.startswith("^") and old_name.endswith("$")
                if is_pattern:
                    # compare to all existing old names and store the result as the new name
                    for _old_name in old_names:
                        if re.match(old_name, _old_name):
                            _new_name = re.sub(old_name, new_name, _old_name)
                            _rules.append((_old_name, _new_name))
                else:
                    _rules.append((old_name, new_name))
            rules = _rules

        # check uniqueness
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

        # sort such that the longest name is first to prevent replacing only parts of names
        rules.sort(key=lambda rule: -len(rule[0]))

        # store in the dictionary
        return OrderedDict(map(tuple, rules))

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

    def open_tfile(self, path, mode):
        return self._tfile_cache.open_tfile(path, mode)

    def get_tobj(self, path, obj_name, mode):
        ROOT = import_ROOT()
        tfile = path if isinstance(path, ROOT.TFile) else self.open_tfile(path, mode)

        cache = self._tobj_input_cache if mode == "READ" else self._tobj_output_cache
        if obj_name not in cache[tfile]:
            self.logger.debug("loading object {} from file {}".format(
                obj_name, tfile.GetPath().rstrip("/")))
            cache[tfile][obj_name] = tfile.Get(obj_name)

        return cache[tfile][obj_name]

    @contextlib.contextmanager
    def start(self, expand=None):
        # clear all caches
        self._clear_caches()

        # yield the context and handle errors
        try:
            with self._tfile_cache:
                with manipulate_datacard(self.datacard, read_structured=True) as (blocks, content):
                    old_names = None
                    if expand == "processes":
                        old_names = set(p["name"] for p in content["processes"])
                    elif expand == "parameters":
                        old_names = set(p["name"] for p in content["parameters"])
                    elif expand:
                        raise Exception("unknown expand type '{}'".format(expand))

                    # expand and validate rules
                    self.rules = self._expand_and_validate_rules(self.rules, old_names=old_names)

                    yield blocks

                # add all output objects to the tfile cache for writing
                for f in self._tobj_output_cache:
                    for tobj in self._tobj_output_cache[f].values():
                        self._tfile_cache.write_tobj(f, tobj)

        except BaseException as e:
            self.logger.error("an exception of type {} occurred while renaming the datacard".format(
                e.__class__.__name__))
            raise

        finally:
            self._clear_caches()


class ShapeLine(object):

    @classmethod
    def parse_line(cls, line):
        return cls.parse_tuple(line.strip().split())

    @classmethod
    def parse_dict(cls, d):
        required = ["process_pattern", "bin_pattern", "file", "nom_pattern"]
        optional = ["syst_pattern"]
        for key in required:
            if key not in d:
                raise Exception("shape line dict misses field '{}'".format(key))

        return cls.parse_tuple([d.get(key) for key in required + optional])

    @classmethod
    def parse_tuple(cls, tpl):
        tpl = tuple(tpl)
        if len(tpl) < 4 or (len(tpl) == 4 and tpl[-1] != "FAKE"):
            raise Exception("invalid shape line tuple {}".format(tpl))

        return (tpl + (None, None))[1:6]

    @classmethod
    def parse(cls, value):
        if isinstance(value, six.string_types):
            return cls.parse_line(value)
        elif isinstance(value, dict):
            return cls.parse_dict(value)
        else:
            return cls.parse_tuple(value)

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
        if self.is_fake:
            w += 10000
        if self.bin == "*":
            w += 1000000
        if self.process == "*":
            w += 100000000
        return w

    @property
    def is_fake(self):
        return self.file.lower() == "fake"

    def copy(self):
        return copy.copy(self)


def create_datacard_instance(datacard, create_shape_builder=False, **kwargs):
    """
    Parses a *datacard* using ``HiggsAnalysis.CombinedLimit.DatacardParser.parseCard`` and returns a
    ``HiggsAnalysis.CombinedLimit.Datacard.Datacard`` instance. When *create_shape_builder* is
    *True*, a ``HiggsAnalysis.CombinedLimit.ShapeTools.ShapeBuilder`` is built as well and returned
    as the second item in a 2-tuple. All *kwargs* are forwarded as options to both the datacard and
    the shape builder initialization.
    """
    from HiggsAnalysis.CombinedLimit.DatacardParser import parseCard, addDatacardParserOptions
    from HiggsAnalysis.CombinedLimit.ShapeTools import ShapeBuilder

    # create a dummy option parser
    parser = OptionParser()
    addDatacardParserOptions(parser)
    options = parser.parse_args([])[0]

    # forward kwargs as options with useful defaults
    kwargs.setdefault("mass", 125.)
    for key, value in kwargs.items():
        setattr(options, key, value)

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
    "shapes", "observations", "rates", "parameters", "groups", "auto_mc_stats", "nuisance_edits",
    and "unknown". These blocks are returned in an ordered dictionary for further inspection.
    """
    # create the returned mapping
    fields = [
        "preamble",
        "counts",
        "shapes",
        "observations",
        "rates",
        "parameters",
        "groups",
        "auto_mc_stats",
        "nuisance_edits",
        "unknown",
    ]
    blocks = OrderedDict((field, []) for field in fields)

    # read lines
    datacard = real_path(datacard)
    with open(datacard, "r") as f:
        lines = []
        for line in f.readlines():
            line = line.strip().lstrip("- ")
            if line and not line.startswith(("#", "//")):
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
            blocks["observations"].extend([line, next_line])
            del lines[obs_offset:obs_offset + 2]
            break

    # trace interdependent lines describing process rates
    for rate_offset in range(len(lines) - 3):
        line = lines[rate_offset]
        if not line.startswith("bin "):
            continue
        next_lines = lines[rate_offset + 1:rate_offset + 4]
        if next_lines[0].startswith("process ") and next_lines[1].startswith("process ") and \
                next_lines[2].startswith("rate "):
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


def read_datacard_structured(datacard):
    """
    Reads a *datacard* and returns a structured, nested object with its content.
    """
    # prepare the datacard path
    datacard = real_path(datacard)

    # prepare the output data
    data = OrderedDict()
    data["bins"] = []  # {name: string}
    data["processes"] = []  # {name: string, id: int}
    data["rates"] = OrderedDict()  # {bin: {process: float}
    data["observations"] = OrderedDict()  # {bin: float}
    data["shapes"] = []  # {index: int, bin: string, bin_pattern: string process: string, process_pattern: string, path: string, nom_pattern: string, syst_pattern: string}
    data["parameters"] = []  # {name: string, type: string, columnar: bool, spec: ...}

    # read the content
    blocks = read_datacard_blocks(datacard)

    # get bin and process name pairs
    bin_names = blocks["rates"][0].split()[1:]
    process_names = blocks["rates"][1].split()[1:]
    process_ids = blocks["rates"][2].split()[1:]
    rates = blocks["rates"][3].split()[1:]

    # check if all lists have the same lengths
    if not (len(bin_names) == len(process_names) == len(process_ids) == len(rates)):
        raise Exception("the number of bin names ({}), process names ({}), process ids "
            "({}) and rates ({}) does not match".format(len(bin_names), len(process_names),
            len(process_ids), len(rates)))

    # store data
    for bin_name, process_name, process_id, rate in zip(bin_names, process_names, process_ids,
            rates):
        if not any(d["name"] == bin_name for d in data["bins"]):
            data["bins"].append({"name": bin_name})

        if not any(d["name"] == process_name for d in data["processes"]):
            data["processes"].append(OrderedDict([("name", process_name), ("id", int(process_id))]))

        data["rates"].setdefault(bin_name, OrderedDict())[process_name] = float(rate)

    # get observations
    bin_names_obs = blocks["observations"][0].split()[1:]
    observations = blocks["observations"][1].split()[1:]

    # check if the bin names are the same
    if set(bin_names) != set(bin_names_obs):
        raise Exception("the bins defined in observations and rates do not match")

    # store data
    for bin_name, obs in zip(bin_names_obs, observations):
        if bin_name not in data["observations"]:
            data["observations"][bin_name] = float(obs)

    # read shape file data
    # sort them so that most specific ones (i.e. without wildcards) come first
    shape_lines = [ShapeLine(line, j) for j, line in enumerate(blocks.get("shapes", []))]
    shape_lines.sort(key=lambda shape_line: shape_line.sorting_weight)
    for bin_name, process_name in zip(bin_names, process_names):
        # get the shape line that applies
        for sl in shape_lines:
            if multi_match(bin_name, sl.bin) and multi_match(process_name, sl.process):
                # store the entry
                data["shapes"].append(OrderedDict([
                    ("index", sl.i),
                    ("bin", bin_name),
                    ("bin_pattern", sl.bin),
                    ("process", process_name),
                    ("process_pattern", sl.process),
                    ("path", sl.file),
                    ("nom_pattern", sl.nom_pattern),
                    ("syst_pattern", sl.syst_pattern),
                ]))
                break

    # get parameters
    for line in blocks.get("parameters", []):
        parts = line.split()

        # skip certain lines
        if len(parts) < 3 or parts[0] == "nuisance":
            continue

        (param_name, param_type), param_spec = parts[:2], parts[2:]

        if not multi_match(param_type, columnar_parameter_directives):
            # when the type is not columnar, store the param_spec as is
            data["parameters"].append(OrderedDict([
                ("name", param_name),
                ("type", param_type),
                ("columnar", False),
                ("spec", param_spec),
            ]))
        else:
            # when it is columnar, store effects per bin and process pair
            if len(param_spec) != len(bin_names):
                raise Exception("numbef of columns of parameter {} ({}) does not match number of "
                    "bin-process pairs ({})".format(param_name, len(param_spec), len(bin_names)))
            _param_spec = OrderedDict()
            for bin_name, process_name, spec in zip(bin_names, process_names, param_spec):
                _param_spec.setdefault(bin_name, OrderedDict())[process_name] = spec
            data["parameters"].append(OrderedDict([
                ("name", param_name),
                ("type", param_type),
                ("columnar", True),
                ("spec", _param_spec),
            ]))

    return data


@contextlib.contextmanager
def manipulate_datacard(datacard, target_datacard=None, read_only=False, read_structured=False,
        writer="pretty"):
    """
    Context manager that opens a *datacard* and yields its contents as a dictionary of specific
    content blocks as returned by :py:func:`read_datacard_blocks`. Each block is a list of lines
    which can be updated in-place to make changes to the datacard. When a *target_datacard* is
    defined, the changes are saved in a new datacard at this location and the original datacard
    remains unchanged. When no changes are to be made to the datacard, you may set *read_only* to
    *True* to disable the tracking of changes. However, please note that the *target_datacard* is
    still written when given. When *read_structured* is *True*, the context yields not only the
    blocks of lines, but also a structured, nexted object, obtained from
    :py:func:`read_datacard_structured`. However, changes to this object are not propagated to the
    manipulated datacard. Example:

    .. code-block:: python

        # add a new (rate) parameter to the datacard and remove autoMCStats
        with manipulate_datacard("datacard.txt") as content:
            content["parameters"].append("beta rateParam B bkg 50")
            del content["auto_mc_stats"][:]

        # also yield a structured object with its content
        with manipulate_datacard("datacard.txt", read_structured=True) as (content, struct):
            ...

    *writer* should be a function receiving a file object and the changed datacard blocks to write
    the contents of the new datacard. When its value is ``"simple"`` or ``"pretty"`` (strings),
    :py:meth:`write_datacard_simple` or :py:meth:`write_datacard_pretty`, respectively, are used.
    """
    # read the datacard content in blocks
    datacard = real_path(datacard)
    blocks = read_datacard_blocks(datacard)

    # define the object to yield, and potentially extract stuctured data
    yield_obj = blocks
    if read_structured:
        struct = read_datacard_structured(datacard)
        yield_obj = (blocks, struct)

    # yield blocks and keep track of changes via hashes
    hash_before = None if read_only else law.util.create_hash(blocks)
    yield yield_obj
    hash_after = None if read_only else law.util.create_hash(blocks)
    has_changes = hash_after != hash_before

    # prepare the target location when given
    if target_datacard:
        target_datacard = real_path(target_datacard)
        target_dirname = os.path.dirname(target_datacard)
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)

    # prepare the writer
    if writer == "simple":
        writer = write_datacard_simple
    elif writer == "pretty":
        writer = write_datacard_pretty

    # handle saving of changes
    if has_changes:
        with open(target_datacard or datacard, "w") as f:
            writer(f, blocks)

    elif target_datacard:
        # no changes, just copy the original file
        shutil.copy2(datacard, target_datacard)


def write_datacard_simple(f, blocks, skip_fields=None):
    """
    Writes the contents of a datacard given in *blocks* (in the format returned by
    :py:meth:`read_datacard_blocks`) into a open file objected *f* the most simple way possible.
    *skip_fields* can be a sequence of names of fields whose lines are not written.
    """
    for field, lines in blocks.items():
        # skip empty lines
        if not lines:
            continue

        # skip certain filds
        if skip_fields and field in skip_fields:
            continue

        # simply write all lines of the block
        f.write("\n".join(lines) + "\n")

        # add a separator after certain fields
        if field in ["preamble", "counts", "shapes", "observations", "rates"]:
            f.write(80 * "-" + "\n")


def write_datacard_pretty(f, blocks, skip_fields=False):
    """
    Writes the contents of a datacard given in *blocks* (in the format returned by
    :py:meth:`read_datacard_blocks`) into a open file objected *f* in a pretty way, i.e., with
    proper offsets between values across columnar lines. *skip_fields* can be a sequence of names of
    fields whose lines are not written.
    """
    skip_fields = skip_fields or []

    # default spacing and block separator
    spacing = 2 * " "
    sep = 80 * "-"

    # helper for writing lines
    def write(l):
        if not isinstance(l, six.string_types):
            l = spacing.join(map(str, l))
        f.write(l.strip() + "\n")

    # helper to write a block of lines with aligned columns
    def align(lines, n_cols=None):
        # split into columns
        rows = [
            (line.strip().split() if isinstance(line, six.string_types) else line) for line in lines
        ]
        # add or remove columns
        if not n_cols:
            n_cols = max([len(row) for row in rows] + [0])
        for row in rows:
            diff = n_cols - len(row)
            if diff > 0:
                row.extend(diff * [""])
            elif diff < 0:
                del row[n_cols:]
        # get the maximum width per column
        widths = [max([len(row[i]) for row in rows] + [0]) for i in range(n_cols)]
        # combine to lines again and return
        return [
            spacing.join(value.ljust(width) for value, width in zip(row, widths)) for row in rows
        ]

    # print the premble as is when existing
    if "preamble" not in skip_fields and blocks.get("preamble"):
        write("\n".join(blocks["preamble"]))
        write(sep)

    # print "*max" counts without subsequent comments
    if "counts" not in skip_fields:
        for line in blocks["counts"]:
            write(line.split()[:2])
        write(sep)

    # write shape lines
    if "shapes" not in skip_fields and blocks.get("shapes"):
        for line in align(blocks["shapes"], n_cols=6):
            write(line)
        write(sep)

    # write observations
    if "observations" not in skip_fields:
        for line in align(blocks["observations"]):
            write(line)
        write(sep)

    # align process rates and columnar parameters combined
    parameter_lines = blocks.get("parameters") if "parameters" not in skip_fields else []
    columnar_parameter_lines = []
    other_parameter_lines = defaultdict(list)
    if parameter_lines:
        for line in parameter_lines:
            parts = line.strip().split()
            param_type = "missing" if len(parts) < 2 else parts[1]
            if multi_match(param_type, columnar_parameter_directives):
                columnar_parameter_lines.append(parts)
            else:
                other_parameter_lines[param_type].append(parts)

    rate_lines = []
    if "rates" not in skip_fields:
        for rate_line in blocks["rates"]:
            parts = rate_line.strip().split()
            # insert an empty space when columnar parameter lines exist as they have an additional
            # column for the parameter type before columnar values start
            if columnar_parameter_lines:
                parts = parts[:1] + [""] + parts[1:]
            rate_lines.append(parts)

    # align lines and split into rate and parameters again
    aligned_lines = align(rate_lines + columnar_parameter_lines)
    rate_lines = aligned_lines[: len(rate_lines)]
    columnar_parameter_lines = aligned_lines[len(rate_lines):]

    # write rates, already aligned
    if rate_lines:
        for line in rate_lines:
            write(line)
        write(sep)

    # write columnar parameters, already aligned
    for line in columnar_parameter_lines:
        write(line)

    # write non-columnar parameters, sorted and aligned per directive
    for lines in other_parameter_lines.values():
        for line in align(lines):
            write(line)

    # write auto mc stats aligned
    for field in ["auto_mc_stats"]:
        if field not in skip_fields and blocks.get(field):
            for line in align(blocks[field]):
                write(line)

    # write groups, nuisance edits and unknown lines with proper spacing
    for field in ["groups", "nuisance_edits", "unknown"]:
        if field not in skip_fields and blocks.get(field):
            for line in blocks[field]:
                write(line.strip().split())


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
        new_shape_files = {}
        for i, line in enumerate(list(content["shapes"])):
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
                new_shape_files[shape_file] = new_shape_file

        # replace shape files in extArg's
        for i, line in enumerate(list(content["parameters"])):
            parts = line.split()
            if len(parts) < 3 or parts[1] != "extArg":
                continue

            # get the shape file
            shape_file, rest = parts[2].split(":", 1) if ":" in parts[2] else (parts[2], "")
            if rest:
                rest = ":" + rest

            # update if needed
            if shape_file in new_shape_files:
                new_line = " ".join(parts[:2] + [new_shape_files[shape_file] + rest] + parts[3:])
                content["parameters"][i] = new_line


def bundle_datacard(datacard, directory, shapes_directory=".", skip_shapes=False):
    """
    Takes a *datacard* given by its path, copies it as well as the shape files it refers to a new
    location given by *directory* and updates the shape lines accordingly. When a *shapes_directory*
    is given, the shape files are copied to this directory instead, which can be relative to
    *directory*. When *skip_shapes* is *True*, only the datacard is copied. The path to the new
    datacard is returned.
    """
    # prepare the directories
    directory = real_path(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    shapes_directory_relative = not shapes_directory.startswith("/")
    shapes_directory = real_path(os.path.join(directory, shapes_directory or "."))
    if not os.path.exists(shapes_directory):
        os.makedirs(shapes_directory)

    # copy the card itself
    src_datacard = real_path(datacard)
    dst_datacard = copy_no_collisions(src_datacard, directory)

    # copy shape files and update the datacard shape lines
    copied_files = {}

    def update_and_copy(shape_file, *_):
        abs_shape_file = os.path.join(os.path.dirname(src_datacard), shape_file)
        if abs_shape_file not in copied_files:
            if skip_shapes:
                copied_files[abs_shape_file] = abs_shape_file
            else:
                copied_files[abs_shape_file] = copy_no_collisions(abs_shape_file, shapes_directory)
        if shapes_directory_relative:
            return os.path.relpath(copied_files[abs_shape_file], directory)
        else:
            return copied_files[abs_shape_file]

    update_shape_files(update_and_copy, dst_datacard)

    return dst_datacard


def update_shape_name(towner, old_name, new_name):
    """
    Renames a shape object (a ROOT histogram or RooFit PDF) contained in an owner object
    *towner* (a ROOT file, directory, or ROOFit workspace) from *old_name* to *new_name*. When the
    object to rename is a RooFit PDF, its normalization formula is renamed also as required by
    combine.
    """
    if not towner:
        raise Exception("owner object is null pointer, cannot rename shape {} to {}".format(
            old_name, new_name))

    elif towner.InheritsFrom("TDirectoryFile"):
        # strategy: get the object, make a copy with the new name, delete all cycles of the old
        # object and write the new one

        # also consider intermediate tdirectories
        if old_name.count("/") != new_name.count("/"):
            raise Exception(
                "when renamening shapes in a TDirectoryFile, the old name ({}) and new name ({}) "
                "must have to same amount of '/' characters for the object to remain at the same "
                "depth".format(old_name, new_name)
            )

        if "/" in old_name:
            # get the next, intermediate directory and check if the renaming affects it
            old_owner_name, old_rest = old_name.split("/", 1)
            new_owner_name, new_rest = new_name.split("/", 1)

            if old_owner_name == new_owner_name:
                # the directory name is not changed, just get it
                towner = towner.Get(old_owner_name)
            else:
                # the directory name is changed, use recursion
                # TODO: when there is already an owner with the new name, an exception will be
                # raised by the lines below in the recursion, but one could actually try to
                # effectively merge the two objects (usually directories), given that names of their
                # contained objects do not collide
                towner = update_shape_name(towner, old_owner_name, new_owner_name)

            # do the actual renaming of the rest
            return update_shape_name(towner, old_rest, new_rest)

        # get the object and check if it's valid
        tobj_orig = towner.Get(old_name)
        if not tobj_orig:
            raise Exception("no object named {} found in {}".format(old_name, towner))

        # stop here when the name does not change at all
        if new_name == old_name:
            return tobj_orig

        # check if there is already an object with the new name
        tobj_clone = towner.Get(new_name)
        if tobj_clone:
            raise Exception("object named {} already present in {}".format(new_name, towner))

        # go ahead and rename
        towner.cd()
        tobj_clone = tobj_orig.Clone(new_name)
        tobj_clone.SetTitle(tobj_clone.GetTitle().replace(old_name, new_name))
        towner.Delete(old_name + ";*")
        tobj_clone.Write(new_name)

        return tobj_clone

    elif towner.InheritsFrom("RooWorkspace"):
        # strategy: get the data or pdf, and optional norm objects, simply rename them
        tdata = towner.data(old_name)
        tpdf = towner.pdf(old_name)
        if not tdata and not tpdf:
            raise Exception("no pdf or data named {} found in {}".format(old_name, towner))

        # stop here when the name does not change at all
        if new_name == old_name:
            return tdata or tpdf

        # go ahead and rename
        if tdata:
            tdata.SetName(new_name)
            tdata.SetTitle(tdata.GetTitle().replace(old_name, new_name))
        if tpdf:
            tpdf.SetName(new_name)
            tpdf.SetTitle(tpdf.GetTitle().replace(old_name, new_name))

        # also rename the norm object when existing
        old_norm_name = old_name + "_norm"
        new_norm_name = new_name + "_norm"
        tnorm = towner.arg(old_norm_name)
        if tnorm:
            tnorm.SetName(new_norm_name)
            tnorm.SetTitle(tnorm.GetTitle().replace(old_norm_name, new_norm_name))

        return tdata or tpdf

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


def update_datacard_count(blocks, key, value, diff=False, logger=None):
    """
    Update the count *key* (e.g. imax, jmax, or kmax) of a datacard given by *blocks*, as returned
    by :py:func:`manipulate_datacard`, to a *value*. When *diff* is *True* and the current value is
    not a wildcard, *value* is added to that value. When a *logger* is defined, an info-level log
    is produced.
    """
    if blocks.get("counts"):
        for i, count_line in enumerate(list(blocks["counts"])):
            parts = count_line.split()
            if len(parts) >= 2 and parts[0] == key:
                new_value = None
                if not diff:
                    new_value = value
                    logger.info("set {} to {}".format(key, new_value))
                elif parts[1] != "*":
                    new_value = value
                    old_value = int(parts[1])
                    new_value = old_value + value
                    logger.info("set {} from {} to {}".format(key, old_value, new_value))
                if new_value is not None:
                    parts[1] = str(new_value)
                    blocks["counts"][i] = " ".join(parts)
                break


def drop_datacard_lines(blocks, field, indices):
    """
    Drops lines with *indices* inplace from a datacard block given by *blocks*, as returned
    by :py:func:`manipulate_datacard`, and *field*. Returns *True* when at least one line was
    removed, and *False* otherwise. Example:

    .. code-block:: python

        # drop the first four line of the "parameters" block
        drop_datacard_lines(blocks, "parameters", [0, 1, 2, 3])
    """
    if field not in blocks or not any(i < len(blocks[field]) for i in indices):
        return False

    # change lines in-place
    lines = [line for i, line in enumerate(blocks[field]) if i not in indices]
    del blocks[field][:]
    blocks[field].extend(lines)

    return True


def expand_file_lines(paths, skip_comments=True):
    """
    Returns a concatenated list of lines in files given by *paths*. When *skip_comments* is *True*,
    lines starting with "#" or "/" are skipped. When a path is not a string or does not point to an
    existing file, the value is added to the returned list as is.
    """
    lines = []
    for path in paths:
        # first try to interpret it as a file
        is_pattern = isinstance(path, six.string_types) and (path.startswith("^") or "*" in path)
        _path = real_path(path) if isinstance(path, six.string_types) and not is_pattern else ""
        if not os.path.isfile(_path):
            # not a file, use as is
            lines.append(path)
        else:
            # read the file line by line, accounting for empty lines and comments
            with open(_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line and (not skip_comments or not line.startswith(("#", "//"))):
                        lines.append(line)
    return lines


def prepare_prefit_var(var, pdf, epsilon=0.001):
    """
    Prepares a RooRealVar *var* for the extraction of its prefit values using the corresponding
    *pdf* object. Internally, this is done using a made-up fit with precision *epsilon* following a
    recipe in the CombineHarvester (see https://github.com/cms-analysis/CombineHarvester/blob/f1029e160701140ce3a1c1f44a991315fd272886/CombineTools/python/combine/utils.py#L87-L95).
    *var* is returned.
    """
    ROOT = import_ROOT()

    if var and pdf:
        nll = ROOT.RooConstraintSum("NLL", "", ROOT.RooArgSet(pdf), ROOT.RooArgSet(var))
        minimizer = ROOT.RooMinimizer(nll)
        minimizer.setEps(epsilon)
        minimizer.setErrorLevel(0.5)
        minimizer.setPrintLevel(-1)
        minimizer.setVerbose(False)
        minimizer.minimize("Minuit2", "migrad")
        minimizer.minos(ROOT.RooArgSet(var))

    return var


def get_workspace_parameters(workspace, workspace_name="w", config_name="ModelConfig"):
    """
    Takes a workspace stored in a ROOT file *workspace* with the name *workspace_name* and gathers
    information on all non-constant parameters. The return value is an ordered dictionary that maps
    parameter names to dicts with fields ``name``, ``type``, ``groups`` and ``prefit``. The
    functionality is loosely based on ``CombineHarvester.CombineTools.combine.utils``.
    """
    ROOT = import_ROOT()
    from HiggsAnalysis.CombinedLimit.RooAddPdfFixer import FixAll

    # read the workspace
    workspace = real_path(workspace)
    f = ROOT.TFile.Open(workspace)
    w = f.Get(workspace_name)
    FixAll(w)

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
        var = prepare_prefit_var(param, pdf)
        nom = var.getVal()
        if pdf_type == "Unconstrained":
            prefit = [nom, nom, nom]
        else:
            prefit = [nom + var.getErrorLo(), nom, nom + var.getErrorHi()]

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
