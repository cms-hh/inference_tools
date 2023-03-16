# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

import os
import re
from functools import reduce
from collections import OrderedDict, defaultdict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    POITask,
    POIPlotTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits
from dhi.eft_tools import sort_eft_benchmark_names
from dhi.config import br_hh
from dhi.util import common_leading_substring


class EFTBase(POITask, MultiDatacardTask):

    datacard_pattern = law.CSVParameter(
        default=(),
        description="one or multiple comma-separated regular expressions for selecting datacards "
        "from each of the sequences passed in --multi-datacards, and for extracting information "
        "with a single regex group; when set on the command line, single quotes should be used; "
        "when empty, a common pattern is extracted per datacard sequence; default: empty",
    )
    datacard_pattern_matches = law.CSVParameter(
        default=(),
        significant=False,
        description="internal parameter, do not use manually",
    )

    exclude_params_index = {"datacard_names", "datacard_order", "datacard_pattern_matches"}
    exclude_params_repr = {"datacard_pattern", "datacard_pattern_matches"}

    hh_model = law.NO_STR
    datacard_names = None
    datacard_order = None
    allow_empty_hh_model = True

    poi = "r_gghh"

    @classmethod
    def modify_param_values(cls, params):
        params = POITask.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)

        # re-group multi datacards by basenames, filter with datacard_pattern and store matches
        if (
            "multi_datacards" in params and
            "datacard_pattern" in params and
            not params.get("datacard_pattern_matches")
        ):
            patterns = params["datacard_pattern"]
            multi_datacards = params["multi_datacards"]

            # infer a common pattern automatically per sequence
            if not patterns:
                # find the common leading substring of datacard bases and built a pattern from that
                patterns = []
                for datacards in multi_datacards:
                    basenames = [os.path.basename(datacard) for datacard in datacards]
                    common_basename = reduce(common_leading_substring, basenames)
                    patterns.append(common_basename + r"(.+)\.txt")

            # when there is one pattern and multiple datacards or vice versa, expand the former
            if len(patterns) == 1 and len(multi_datacards) > 1:
                patterns *= len(params["multi_datacards"])
            elif len(patterns) > 1 and len(multi_datacards) == 1:
                multi_datacards *= len(patterns)
            elif len(patterns) != len(multi_datacards):
                raise ValueError(
                    "the number of patterns in --datacard-pattern ({}) does not "
                    "match the number of datacard sequences in --multi-datacards ({})".format(
                        len(patterns), len(params["multi_datacards"]),
                    ),
                )

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
                    raise Exception(
                        "the datacard pattern '{}' did not match any of the selected "
                        "datacards\n  {}".format(pattern, "\n  ".join(datacards)),
                    )

            # sort cards, assign back to multi_datacards and store the pattern matches
            params["multi_datacards"] = tuple(tuple(sorted(cards)) for cards in groups.values())
            params["datacard_pattern_matches"] = tuple(groups.keys())
            params["datacard_pattern"] = tuple(patterns)

        return params

    def __init__(self, *args, **kwargs):
        super(EFTBase, self).__init__(*args, **kwargs)

        # create a map of datacard names (e.g. benchmark number or EFT parameters) to datacard paths
        self.eft_datacards = dict(zip(self.datacard_pattern_matches, self.multi_datacards))

    def store_parts(self):
        parts = super(EFTBase, self).store_parts()
        parts["poi"] = "poi_{}".format(self.poi)
        return parts


class EFTBenchmarkBase(EFTBase):

    def __init__(self, *args, **kwargs):
        super(EFTBenchmarkBase, self).__init__(*args, **kwargs)

        # sort EFT datacards according to benchmark names
        names = sort_eft_benchmark_names(self.eft_datacards.keys())
        self.benchmark_datacards = OrderedDict((name, self.eft_datacards[name]) for name in names)


class EFTLimitBase(CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def workflow_requires(self):
        reqs = super(EFTLimitBase, self).workflow_requires()

        # workspaces of all datacard sequences
        if not self.pilot:
            # require the requirements of all branch tasks when not in pilot mode
            reqs["workspace"] = {
                b: CreateWorkspace.req(
                    self,
                    datacards=self.benchmark_datacards[data],
                    hh_model=law.NO_STR,
                )
                for b, data in self.branch_map.items()
            }

        return reqs

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)

        return "--seed {self.branch} --toys {self.toys} --run expected --noFitAsimov".format(
            self=self,
        )

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
        return CreateWorkspace.req(
            self,
            datacards=self.benchmark_datacards[self.branch_data],
            hh_model=law.NO_STR,
            branch=0,
        )

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


class PlotEFTBenchmarkLimits(EFTBenchmarkBase, POIPlotTask):

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
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["benchmarks", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        bm_names = list(self.benchmark_datacards.keys())
        limit_values = dict(zip(bm_names, self.input().load(formatter="numpy")["data"]))

        # prepare conversion scale
        scale = br_hh.get(self.br, 1.) * {"fb": 1., "pb": 0.001}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        for bm_name in bm_names:
            record = limit_values[bm_name]
            entry = {
                "name": bm_name,
                "expected": [v * scale for v in record.tolist()[:5]],
            }
            if self.unblinded:
                entry["observed"] = float(record[5]) * scale
            data.append(entry)

            # some printing
            self.publish_message("BM {} -> {}".format(bm_name, record["limit"]))

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_benchmark_limits",
            paths=[outp.path for outp in outputs],
            data=data,
            poi=self.poi,
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style if self.style != law.NO_STR else None,
        )


class PlotMultiEFTBenchmarkLimits(PlotEFTBenchmarkLimits):

    datacard_names = MultiDatacardTask.datacard_names
    force_equal_sequence_lengths = True
    compare_sequence_length = True

    def requires(self):
        return [
            MergeEFTBenchmarkLimits.req(
                self,
                multi_datacards=tuple((datacards[i],) for datacards in self.multi_datacards),
            )
            for i in range(len(self.multi_datacards[0]))
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["multi_benchmarks", self.get_output_postfix(), parts])
        outputs = [self.local_target(name) for name in names]

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        bm_names = list(self.benchmark_datacards.keys())
        limit_values = [
            dict(zip(bm_names, list(inp.load(formatter="numpy")["data"])))
            for inp in self.input()
        ]

        # prepare conversion scale
        scale = br_hh.get(self.br, 1.) * {"fb": 1., "pb": 0.001}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        for bm_name in bm_names:
            entry = {
                "name": bm_name,
                "expected": [
                    [v * scale for v in records[bm_name].tolist()[:5]]
                    for records in limit_values
                ],
            }
            if self.unblinded:
                entry["observed"] = [
                    float(records[bm_name][5]) * scale
                    for records in limit_values
                ]
            data.append(entry)

        # datacard names
        names = (
            list(self.datacard_names)
            if self.datacard_names
            else ["datacards {}".format(i + 1) for i in range(len(limit_values))]
        )

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_multi_benchmark_limits",
            paths=[outp.path for outp in outputs],
            data=data,
            names=names,
            poi=self.poi,
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style if self.style != law.NO_STR else None,
        )


PlotMultiEFTBenchmarkLimits.exclude_params_index -= {"datacard_names"}
