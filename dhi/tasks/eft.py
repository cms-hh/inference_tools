# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

import os
import re
from collections import OrderedDict, defaultdict

import law
import luigi
import json
import glob

from dhi.tasks.base import HTCondorWorkflow, PlotTask, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits
from dhi.eft_tools import sort_eft_benchmark_names, sort_eft_scan_names, extract_eft_scan_parameter
from dhi.config import br_hh


class EFTBase(MultiDatacardTask):

    datacard_pattern = law.CSVParameter(
        default=(r"datacard_(.+)\.txt",),
        description="one or multiple comma-separated regular expressions for selecting datacards "
        "from each of the sequences passed in --multi-datacards, and for extracting information "
        "with a single regex group; when set on the command line, single quotes should be used; "
        r"default: ('datacard_(.+)\.txt',)",
    )
    frozen_parameters = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of parameters to be frozen in addition to non-POI and "
        "scan parameters",
    )
    frozen_groups = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of groups of parameters to be frozen",
    )
    unblinded = luigi.BoolParameter(
        default=False,
        description="unblinded computation and plotting of results; default: False",
    )
    datacard_pattern_matches = law.CSVParameter(
        default=(),
        significant=False,
        description="internal parameter, do not use"
    )

    exclude_params_index = {"datacard_names", "datacard_order", "datacard_pattern_matches"}

    hh_model = law.NO_STR
    datacard_names = None
    datacard_order = None
    allow_empty_hh_model = True

    poi = "r_gghh"

    @classmethod
    def modify_param_values(cls, params):
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)

        # re-group multi datacards by basenames, filter with datacard_pattern and store matches
        if "multi_datacards" in params and "datacard_pattern" in params \
                and not params.get("datacard_pattern_matches"):
            # when there is one pattern and multiple datacards or vice versa, expand the former
            patterns = params["datacard_pattern"]
            multi_datacards = params["multi_datacards"]
            if len(patterns) == 1 and len(multi_datacards) > 1:
                patterns *= len(params["multi_datacards"])
            elif len(patterns) > 1 and len(multi_datacards) == 1:
                multi_datacards *= len(patterns)
            elif len(patterns) != len(multi_datacards):
                raise ValueError("the number of patterns in --datacard-pattern ({}) does not "
                    "match the number of datacard sequences in --multi-datacards ({})".format(
                        len(patterns), len(params["multi_datacards"])))

            # assign datacards to groups, based on the matched group
            groups = defaultdict(set)
            for datacards, pattern in zip(multi_datacards, patterns):
                n_matches = 0
                for datacard in datacards:
                    # apply the pattern to the basename
                    m = re.match(pattern, os.path.basename(datacard))
                    if m:
                        groups[m.group(1)].add(datacard)
                        n_matches += 1
                if not n_matches:
                    raise Exception("the datacard pattern '{}' did not match any of the selected "
                        "datacards\n  {}".format(pattern, "\n  ".join(datacards)))

            # sort cards, assign back to multi_datacards and store the pattern matches
            params["multi_datacards"] = tuple(tuple(sorted(cards)) for cards in groups.values())
            params["datacard_pattern_matches"] = tuple(groups.keys())
            params["datacard_pattern"] = tuple(patterns)

        # sort frozen parameters
        if "frozen_parameters" in params:
            params["frozen_parameters"] = tuple(sorted(params["frozen_parameters"]))

        # sort frozen groups
        if "frozen_groups" in params:
            params["frozen_groups"] = tuple(sorted(params["frozen_groups"]))

        return params

    def __init__(self, *args, **kwargs):
        super(EFTBase, self).__init__(*args, **kwargs)

        # create a map of datacard names (e.g. benchmark number or EFT parameters) to datacard paths
        self.eft_datacards = dict(zip(self.datacard_pattern_matches, self.multi_datacards))

    def store_parts(self):
        parts = super(EFTBase, self).store_parts()
        parts["poi"] = "poi_{}".format(self.poi)
        return parts

    def get_output_postfix(self, join=True):
        parts = []

        # add the unblinded flag
        if self.unblinded:
            parts.append(["unblinded"])

        # add the poi
        parts.append(["poi", self.poi])

        # add frozen paramaters
        if self.frozen_parameters:
            parts.append(["fzp"] + list(self.frozen_parameters))

        # add frozen groups
        if self.frozen_groups:
            parts.append(["fzg"] + list(self.frozen_groups))

        return self.join_postfix(parts) if join else parts

    @property
    def joined_frozen_parameters(self):
        return ",".join(self.frozen_parameters) or '""'

    @property
    def joined_frozen_groups(self):
        return ",".join(self.frozen_groups) or '""'


class EFTBenchmarkBase(EFTBase):

    def __init__(self, *args, **kwargs):
        super(EFTBenchmarkBase, self).__init__(*args, **kwargs)

        # sort EFT datacards according to benchmark names
        names = sort_eft_benchmark_names(self.eft_datacards.keys())
        self.benchmark_datacards = OrderedDict((name, self.eft_datacards[name]) for name in names)


# the EFTScanBase is currently not needed as we have a didicated model
# so expect the commented lines to be removed soon
# class EFTScanBase(EFTBase):

#     scan_range = law.CSVParameter(
#         default=(-100., 100.),
#         cls=luigi.FloatParameter,
#         min_len=2,
#         max_len=2,
#         sort=True,
#         description="the range of the scan parameter extracted from the datacards in the format "
#         "'min,max'; when empty, the full range is used; no default",
#     )

#     def __init__(self, *args, **kwargs):
#         super(EFTScanBase, self).__init__(*args, **kwargs)

#         # get the name of the parameter to scan
#         scan_parameters = set(map(extract_eft_scan_parameter, self.eft_datacards.keys()))
#         if len(scan_parameters) != 1:
#             raise Exception("datacards belong to more than one EFT scan parameter: {}".format(
#                 ",".join(scan_parameters)))
#         self.scan_parameter = list(scan_parameters)[0]

#         # sort EFT datacards according to scan parameter values
#         values = sort_eft_scan_names(self.scan_parameter, self.eft_datacards.keys())

#         # apply the requested scan range
#         scan_min = max(self.scan_range[0], min(v for _, v in values))
#         scan_max = min(self.scan_range[1], max(v for _, v in values))
#         self.scan_range = (scan_min, scan_max)

#         # store a mapping of scan value to datacards
#         self.scan_datacards = OrderedDict(
#             (v, self.eft_datacards[name])
#             for name, v in values
#             if scan_min <= v <= scan_max
#         )

#     def get_output_postfix(self, join=True):
#         parts = super(EFTScanBase, self).get_output_postfix(join=False)

#         # insert the scan parameter value when this is a workflow branch, and the range otherwise
#         if isinstance(self, law.BaseWorkflow) and self.is_branch():
#             scan_part = [self.scan_parameter, self.branch_data]
#         else:
#             scan_part = ["scan", self.scan_parameter] + list(self.scan_range)
#         parts.insert(2 if self.unblinded else 1, scan_part)

#         return self.join_postfix(parts) if join else parts


class EFTLimitBase(CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def workflow_requires(self):
        reqs = super(EFTLimitBase, self).workflow_requires()

        # workspaces of all datacard sequences
        if not self.pilot:
            # require the requirements of all branch tasks when not in pilot mode
            reqs["workspace"] = {
                b: CreateWorkspace.req(self, datacards=self.benchmark_datacards[data],
                    hh_model=law.NO_STR)
                for b, data in self.branch_map.items()
            }

        return reqs

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)
        else:
            return "--seed {self.branch} --toys {self.toys} --run expected --noFitAsimov".format(
                self=self)

    def build_command(self):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.AsymptoticLimits.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class EFTBenchmarkLimits(EFTBenchmarkBase, EFTLimitBase):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.benchmark_datacards.keys())

    def requires(self):
        return CreateWorkspace.req(self, datacards=self.benchmark_datacards[self.branch_data],
            hh_model=law.NO_STR, branch=0)

    def output(self):
        parts = self.get_output_postfix(join=False)
        parts.append("bm{}".format(self.branch_data))

        return self.local_target("eftlimit__{}.root".format(self.join_postfix(parts)))


class MergeEFTBenchmarkLimits(EFTBenchmarkBase):

    def requires(self):
        return EFTBenchmarkLimits.req(self)

    def output(self):
        parts = self.get_output_postfix(join=False)
        postfix = ("__" + self.join_postfix(parts)) if parts else ""

        return self.local_target("eftlimits{}.npz".format(postfix))

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        records = []
        dtype = [
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        if self.unblinded:
            dtype.append(("observed", np.float32))

        for branch, inp in self.input()["collection"].targets.items():
            limits = UpperLimits.load_limits(inp, unblinded=self.unblinded)
            records.append(limits)

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class PlotEFTBenchmarkLimits(EFTBenchmarkBase, PlotTask):

    xsec = luigi.ChoiceParameter(
        default="fb",
        choices=["pb", "fb"],
        description="convert limits to cross sections in this unit; choices: pb,fb; default: fb",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: '',{}; no default".format(",".join(br_hh.keys())),
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    plot_type = luigi.ChoiceParameter(
        default="shape_BM_all",
        choices=["shape_BM_all", "shape_BM_JHEP04", "shape_BM_JHEP03"],
        description="Which bm points to plot, all + SM, jhep03/04 ones + SM ?",
    )
    x_min = None
    x_max = None
    z_min = None
    z_max = None
    save_hep_data = None

    def requires(self):
        return MergeEFTBenchmarkLimits.req(self)

    def output(self):
        # additional postfix
        parts = []
        parts.append(self.xsec)
        if self.br and self.br != law.NO_STR:
            parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["eftbenchmarks", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]  + [self.local_target(name.replace(".pdf", ".json")) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        limit_values = self.input().load(formatter="numpy")["data"]

        # prepare conversion scale
        scale = br_hh.get(self.br, 1.) * {"fb": 1., "pb": 0.001}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        for name, record in zip(self.benchmark_datacards.keys(), limit_values):
            entry = {
                "name": name,
                "expected": [v * scale for v in record.tolist()[:5]],
            }
            if self.unblinded:
                entry["observed"] = float(record[5]) * scale
            data.append(entry)

            # some printing
            self.publish_message("BM {} -> {}".format(name, record["limit"]))
        # output as ordered dict, to be read as ordered dictionary from the next tast
        # fill data entries as expected by the output dict
        to_output = OrderedDict()
        with open(outputs[1].path, 'w') as fp:
            for record in data :
                to_output.update({record["name"] :
                    OrderedDict([
                    ("limit", record["expected"][0]),
                    ("limit_p1", record["expected"][1]),
                    ("limit_m1", record["expected"][2]),
                    ("limit_p2", record["expected"][3]),
                    ("limit_m2", record["expected"][4]),
                    ("observed", record["observed"] if self.unblinded else record["expected"][0])
                     ])})
            json.dump(to_output, fp, indent=4)

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_benchmark_limits",
            paths=outputs[0].path, #[outp.path for outp in outputs],
            data=data,
            poi=self.poi,
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            paper=self.paper,
            type_plot=self.plot_type
        )

class PlotMultiEFTBenchmarkLimits(PlotTask):
    multi_datacards = law.MultiCSVParameter(
        description="multiple path sequnces to input datacards separated by a colon; supported "
        "formats are '[bin=]path', '[bin=]paths@store_directory for the last datacard in the "
        "sequence, and '@store_directory' for easier configuration; supports globbing and brace "
        "expansion",
        brace_expand=True,
    )

    xsec = luigi.ChoiceParameter(
        default="pb",
        choices=["pb", "fb"],
        description="convert limits to cross sections in this unit; choices: pb,fb; default: fb",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: '',{}; no default".format(",".join(br_hh.keys())),
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    datacard_labels = law.CSVParameter(
        default=(r"bm",),
        description="one or multiple comma-separated regular expressions for selecting datacards "
        "from each of the sequences passed in --multi-datacards, and for extracting information "
        "with a single regex group; when set on the command line, single quotes should be used; "
        r"default: ('bm',)",
    )
    unblinded = luigi.BoolParameter(
        default=False,
        description="unblinded computation and plotting of results; default: False",
    )

    plot_type = luigi.ChoiceParameter(
        default="shape_BM_all",
        choices=["shape_BM_all", "shape_BM_JHEP04", "shape_BM_JHEP03"],
        description="Which bm points to plot, all + SM, jhep03/04 ones + SM ?",
    )

    def requires(self):
        for mm, datacards in enumerate(self.multi_datacards) :
           cards = []
           for c in sorted(set(glob.glob(datacards[0]))):
               cards.append(tuple([c]))
           yield PlotEFTBenchmarkLimits(multi_datacards=tuple(cards), unblinded=self.unblinded, xsec=self.xsec, plot_type=self.plot_type, version = self.version, br=self.br)

    def output(self):
        # additional postfix
        parts = []
        parts.append(self.xsec)
        if self.br and self.br != law.NO_STR:
            parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["multi_benchmarks", parts])
        if self.unblinded:
            parts.append(["unblinded"])
        names2 = self.create_plot_names(["multi_benchmarks", parts])
        return [self.local_target(name) for name in names] + [self.local_target(name) for name in names2]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        outputs = self.output()
        outputs[0].parent.touch()
        data_all = {}
        for ll, label in enumerate(self.datacard_labels) :
             with open(self.input()[ll][1].path, 'r') as fp:
                 tempdata = json.load(fp, object_pairs_hook=OrderedDict)
                 limits = []
                 data_keys_sorted = tempdata.keys()
                 for dd in data_keys_sorted:
                     if self.unblinded:
                         limits.append({
                             "expected": [tempdata[str(dd)]["limit"], tempdata[str(dd)]["limit_p1"],tempdata[str(dd)]["limit_m1"],tempdata[str(dd)]["limit_m2"],tempdata[str(dd)]["limit_p2"]],
                             "observed": tempdata[str(dd)]["observed"],
                             "name": dd
                         })
                     else:
                         limits.append({
                             "expected": [tempdata[str(dd)]["limit"], tempdata[str(dd)]["limit_p1"],tempdata[str(dd)]["limit_m1"],tempdata[str(dd)]["limit_m2"],tempdata[str(dd)]["limit_p2"]],
                             "name": dd
                         })
                 data_all[label]=limits

        self.call_plot_func(
            "dhi.plots.limits.plot_multi_benchmark_limits",
            paths=outputs[0].path,
            data=data_all,
            poi = "r_gghh",
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            paper=self.paper,
            type_plot=self.plot_type
        )
