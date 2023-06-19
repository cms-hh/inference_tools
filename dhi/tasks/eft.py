# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import view_output_plots
from dhi.tasks.remote import HTCondorWorkflow
from dhi.tasks.combine import (
    MultiDatacardTask,
    MultiDatacardTransposedTask,
    POITask,
    POIPlotTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits
from dhi.eft_tools import sort_eft_benchmark_names
from dhi.config import br_hh


class EFTBase(POITask, MultiDatacardTransposedTask):

    hh_model = law.NO_STR
    allow_empty_hh_model = True

    poi = "r_gghh"

    @classmethod
    def modify_param_values(cls, params):
        params = POITask.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTransposedTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def __init__(self, *args, **kwargs):
        super(EFTBase, self).__init__(*args, **kwargs)

        # sort EFT datacards according to benchmark names
        names = sort_eft_benchmark_names(self.multi_datacards_transposed.keys())
        self.benchmark_datacards = OrderedDict(
            (name, self.multi_datacards_transposed[name])
            for name in names
        )

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
        branch_map = []
        for name, cards in self.benchmark_datacards.items():
            for _cards in cards:
                branch_map.append({"benchmark": name, "cards": _cards})
        return branch_map

    def workflow_requires(self):
        reqs = super(EFTBenchmarkLimits, self).workflow_requires()

        # workspaces of all datacard sequences
        if not self.pilot:
            # require the requirements of all branch tasks when not in pilot mode
            reqs["workspace"] = {
                b: CreateWorkspace.req_different_branching(
                    self,
                    datacards=tuple(branch_data["cards"]),
                    hh_model=law.NO_STR,
                )
                for b, branch_data in self.branch_map.items()
            }

        return reqs

    def requires(self):
        return CreateWorkspace.req(
            self,
            datacards=tuple(self.branch_data["cards"]),
            hh_model=law.NO_STR,
            branch=0,
        )

    def output(self):
        parts = self.get_output_postfix(join=False)
        parts.append("bm{}".format(self.branch_data["benchmark"]))

        return self.target("eftlimit__{}.root".format(self.join_postfix(parts)))

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)

        return "--seed {self.branch} --toys {self.toys} --run expected --noFitAsimov".format(
            self=self,
        )

    def build_command(self, fallback_level):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
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

        return self.target("eftlimits{}.npz".format(postfix))

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        records = []
        dtype = [
            ("benchmark", np.unicode_, max(len(n) for n in self.benchmark_datacards)),
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        if self.unblinded:
            dtype.append(("observed", np.float32))

        branch_map = self.requires().branch_map
        inputs = self.input()["collection"].targets
        for b, branch_data in branch_map.items():
            limits = UpperLimits.load_limits(inputs[b], unblinded=self.unblinded)
            records.append((branch_data["benchmark"],) + limits)

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
    save_hep_data = False

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
        outputs["plots"] = [self.target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.localize(input=False)
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load limit values
        limit_values = self.input().load(formatter="numpy")["data"]

        # prepare conversion scale, default is expected to be pb!
        scale = br_hh.get(self.br, 1.0) * {"fb": 1000.0, "pb": 1.0}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        value_mask = ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]
        if self.unblinded:
            value_mask.append("observed")
        for bm_name in limit_values["benchmark"]:
            record = limit_values[limit_values["benchmark"] == bm_name][value_mask][0]
            entry = {
                "name": str(bm_name),
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
    group_duplicate_cards = True

    default_plot_function = "dhi.plots.eft.plot_multi_benchmark_limits"

    def __init__(self, *args, **kwargs):
        super(PlotMultipleEFTBenchmarkLimits, self).__init__(*args, **kwargs)

        # check that each mass point has the same amount of cards
        n_entries = {len(cards) for cards in self.benchmark_datacards.values()}
        if len(n_entries) != 1:
            raise Exception("founds different amount of entries in input datacards: {}".format(
                ",".join(map(str, n_entries)),
            ))
        self.n_entries = list(n_entries)[0]

        # the lengths of names and order indices must match multi_datacards when given
        if self.datacard_names and len(self.datacard_names) != self.n_entries:
            raise Exception("found {} entries in datacard_names whereas {} are expected".format(
                len(self.datacard_names), self.n_entries,
            ))
        if self.datacard_order and len(self.datacard_order) != self.n_entries:
            raise Exception("found {} entries in datacard_order whereas {} are expected".format(
                len(self.datacard_order), self.n_entries,
            ))

    def requires(self):
        return [
            MergeEFTBenchmarkLimits.req(
                self,
                multi_datacards=tuple(
                    tuple(cards[i])
                    for cards in self.benchmark_datacards.values()
                ),
            )
            for i in range(self.n_entries)
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
        outputs["plots"] = [self.target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.localize(input=False)
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load limit values
        limit_values = [
            inp.load(formatter="numpy")["data"]
            for inp in self.input()
        ]

        # prepare conversion scale, default is expected to be pb!
        scale = br_hh.get(self.br, 1.0) * {"fb": 1000.0, "pb": 1.0}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        value_mask = ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]
        for bm_name in limit_values[0]["benchmark"]:
            entry = {
                "name": str(bm_name),
                "expected": [
                    [
                        v * scale
                        for v in vals[vals["benchmark"] == bm_name][0][value_mask].tolist()
                    ]
                    for vals in limit_values
                ],
            }
            if self.unblinded:
                entry["observed"] = [
                    float(vals[vals["benchmark"] == bm_name]["observed"]) * scale
                    for vals in limit_values
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
