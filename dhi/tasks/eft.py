# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    MultiDatacardPatternTask,
    POITask,
    POIPlotTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits
from dhi.eft_tools import sort_eft_benchmark_names
from dhi.config import br_hh


class EFTBase(POITask, MultiDatacardPatternTask):

    hh_model = law.NO_STR
    allow_empty_hh_model = True

    poi = "r_gghh"

    @classmethod
    def modify_param_values(cls, params):
        params = POITask.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardPatternTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def __init__(self, *args, **kwargs):
        super(EFTBase, self).__init__(*args, **kwargs)

        # sort EFT datacards according to benchmark names
        cards = dict(zip(self.datacard_pattern_matches, self.multi_datacards))
        names = sort_eft_benchmark_names(cards.keys())
        self.benchmark_datacards = OrderedDict((name, cards[name]) for name in names)

    @property
    def other_pois(self):
        return []

    def store_parts(self):
        parts = super(EFTBase, self).store_parts()
        parts["poi"] = "poi_{}".format(self.poi)
        return parts


class EFTBenchmarkLimits(EFTBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.benchmark_datacards.keys())

    def workflow_requires(self):
        reqs = super(EFTBenchmarkLimits, self).workflow_requires()

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

    def control_output_postfix(self):
        return "{}__{}".format(
            super(EFTBenchmarkLimits, self).control_output_postfix(),
            self.get_output_postfix(),
        )


class MergeEFTBenchmarkLimits(EFTBase):

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


class PlotEFTBenchmarkLimits(EFTBase, POIPlotTask):

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

    default_plot_function = "dhi.plots.eft.plot_benchmark_limits"

    def requires(self):
        return MergeEFTBenchmarkLimits.req(self)

    def output(self):
        # additional postfix
        parts = []
        parts.append(self.xsec)
        if self.br != law.NO_STR:
            parts.append(self.br)
        if self.y_log:
            parts.append("log")

        outputs = {}

        names = self.create_plot_names(["benchmarks", self.get_output_postfix(), parts])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load limit values
        bm_names = list(self.benchmark_datacards.keys())
        limit_values = dict(zip(bm_names, self.input().load(formatter="numpy")["data"]))

        # prepare conversion scale, default is expected to be pb!
        scale = br_hh.get(self.br, 1.0) * {"fb": 1000.0, "pb": 1.0}[self.xsec]

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
            paths=[outp.path for outp in outputs["plots"]],
            data=data,
            poi=self.poi,
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )


class PlotMultipleEFTBenchmarkLimits(PlotEFTBenchmarkLimits):

    datacard_names = MultiDatacardTask.datacard_names
    datacard_order = MultiDatacardTask.datacard_order
    force_equal_sequence_lengths = True
    compare_sequence_length = True

    default_plot_function = "dhi.plots.eft.plot_multi_benchmark_limits"

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
        parts.append(self.xsec)
        if self.br != law.NO_STR:
            parts.append(self.br)
        if self.y_log:
            parts.append("log")

        outputs = {}

        names = self.create_plot_names(["multi_benchmarks", self.get_output_postfix(), parts])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load limit values
        bm_names = list(self.benchmark_datacards.keys())
        limit_values = [
            dict(zip(bm_names, list(inp.load(formatter="numpy")["data"])))
            for inp in self.input()
        ]

        # prepare conversion scale, default is expected to be pb!
        scale = br_hh.get(self.br, 1.0) * {"fb": 1000.0, "pb": 1.0}[self.xsec]

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

        # reorder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
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
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )


PlotMultipleEFTBenchmarkLimits.exclude_params_index -= {"datacard_names", "datacard_order"}
