# coding: utf-8

"""
Tasks related to significance calculation.
"""

import law
import luigi

from dhi.tasks.base import BoxPlotTask, view_output_plots
from dhi.tasks.remote import HTCondorWorkflow
from dhi.tasks.combine import (
    MultiDatacardTask,
    CombineCommandTask,
    POITask,
    POIMultiTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.tasks.snapshot import Snapshot, SnapshotUser


class GoodnessOfFitBase(POITask, SnapshotUser):

    toys = luigi.IntParameter(
        default=1,
        description="the positive number of toys to sample; default: 1",
    )
    toys_per_branch = luigi.IntParameter(
        default=1,
        description="the number of toys to generate per branch task; the number of tasks in this "
        "workflow is the number of total toys divided by this number; default: 1",
    )
    algorithm = luigi.ChoiceParameter(
        default="saturated",
        choices=["saturated", "KS", "AD"],
        description="the algorithm to use; possible choices are 'saturated', 'KS' and 'AD'; "
        "default: saturated",
    )
    frequentist_toys = luigi.BoolParameter(
        default=False,
        description="use frequentist toys (nuisance parameters set to nominal post-fit values); "
        "recommended for the 'saturated' algorithm; default: False",
    )

    unblinded = None
    allow_parameter_values_in_pois = True
    freeze_pois_with_parameter_values = True

    def store_parts(self):
        parts = super(GoodnessOfFitBase, self).store_parts()
        parts["gof"] = self.algorithm
        return parts

    def get_output_postfix(self, join=True):
        parts = super(GoodnessOfFitBase, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")
        if self.frequentist_toys:
            parts.append("freqtoys")

        return self.join_postfix(parts) if join else parts

    @property
    def toys_postfix(self):
        return "t{}_tpb{}".format(self.toys, self.toys_per_branch)


class GoodnessOfFit(GoodnessOfFitBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def __init__(self, *args, **kwargs):
        super(GoodnessOfFit, self).__init__(*args, **kwargs)

        # check that the number of toys is positive
        if self.toys <= 0:
            raise ValueError("{!r}: number of toys must be positive for GOF tests".format(self))

        # print a warning when the saturated algorithm is use without frequentist toys
        if self.algorithm == "saturated" and not self.frequentist_toys:
            self.logger.warning(
                "it is recommended for goodness-of-fit tests with the "
                "'saturated' algorithm to use frequentiest toys, so please consider adding "
                "--frequentist-toys to the {} task".format(self.__class__.__name__),
            )

    def create_branch_map(self):
        # the branch map refers to indices of toys in that branch, with 0 meaning the test on data
        all_toy_indices = list(range(1, self.toys + 1))
        toy_indices = [[0]] + list(law.util.iter_chunks(all_toy_indices, self.toys_per_branch))
        return dict(enumerate(toy_indices))

    def workflow_requires(self):
        reqs = super(GoodnessOfFit, self).workflow_requires()
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, _exclude={"toys"})
        else:
            reqs["workspace"] = CreateWorkspace.req(self)
        return reqs

    def requires(self):
        reqs = {}
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, branch=0, _exclude={"toys"})
        else:
            reqs["workspace"] = CreateWorkspace.req(self, branch=0)
        return reqs

    def output(self):
        parts = []
        if self.branch == 0:
            parts.append("b0_data")
        else:
            parts.append("b{}_toy{}To{}".format(
                self.branch, self.branch_data[0], self.branch_data[-1],
            ))

        name = self.join_postfix(["gof", self.get_output_postfix(), parts])
        return self.local_target(name + ".root")

    def build_command(self, fallback_level):
        # get the workspace to use and define snapshot args
        if self.use_snapshot:
            workspace = self.input()["snapshot"].path
            snapshot_args = " --snapshotName MultiDimFit"
        else:
            workspace = self.input()["workspace"].path
            snapshot_args = ""

        # toy options
        toy_opts = ""
        if self.branch > 0:
            toy_opts = "--toys {}".format(len(self.branch_data))
            if self.frequentist_toys:
                toy_opts += " --toysFrequentist"

        # build the command
        cmd = (
            "combine -M GoodnessOfFit {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {toy_opts}"
            " {snapshot_args}"
            " --seed {self.branch}"
            " --algo {self.algorithm}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.GoodnessOfFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=workspace,
            output=self.output().path,
            toy_opts=toy_opts,
            snapshot_args=snapshot_args,
        )

        return cmd

    def control_output_postfix(self):
        return "{}__{}".format(
            super(GoodnessOfFit, self).control_output_postfix(),
            self.toys_postfix,
        )


class MergeGoodnessOfFit(GoodnessOfFitBase):

    def requires(self):
        return GoodnessOfFit.req(self)

    def output(self):
        name = self.join_postfix(["gofs", self.get_output_postfix(), self.toys_postfix])
        return self.local_target(name + ".json")

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        # store gof values for the data measurement and for the toys
        data = {"data": None, "toys": []}

        # load values
        for branch, inp in self.input()["collection"].targets.items():
            if not inp.exists():
                self.logger.warning(
                    "input of branch {} at {} does not exist".format(branch, inp.path),
                )
                continue

            values = inp.load(formatter="uproot")["limit"].array("limit")
            if branch == 0:
                data["data"] = float(values[0])
            else:
                data["toys"].extend(values.tolist())

        # save the result as json
        self.output().dump(data, formatter="json")


class PlotGoodnessOfFit(GoodnessOfFitBase, POIPlotTask):

    n_bins = luigi.IntParameter(
        default=32,
        significant=False,
        description="number of bins in toy histograms for plotting; default: 32",
    )

    z_min = None
    z_max = None
    save_hep_data = None

    sort_pois = False

    default_plot_function = "dhi.plots.gof.plot_gof_distribution"

    def requires(self):
        return MergeGoodnessOfFit.req(self)

    def output(self):
        outputs = {}

        names = self.create_plot_names(["gofs", self.get_output_postfix(), self.toys_postfix])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix(), self.toys_postfix])
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

        # load input data
        gof_data = self.input().load(formatter="json")

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
            data=gof_data["data"],
            toys=gof_data["toys"],
            algorithm=self.algorithm,
            n_bins=self.n_bins,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )


class PlotMultipleGoodnessOfFits(PlotGoodnessOfFit, POIMultiTask, MultiDatacardTask, BoxPlotTask):

    toys = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1,),
        description="comma-separated list of positive amounts of toys per datacard sequence in "
        "--multi-datacards; when one value is given, it is used for all datacard sequences; "
        "default: (1,)",
    )
    toys_per_branch = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1,),
        description="comma-separated list of numbers per datacard sequence in --multi-datacards to "
        "define the amount of toys to generate per branch task; when one value is given, it is "
        "used for all datacard sequences; default: (1,)",
    )

    y_min = None
    y_max = None

    compare_multi_sequence = "multi_datacards"

    default_plot_function = "dhi.plots.gof.plot_gofs"

    def __init__(self, *args, **kwargs):
        super(PlotMultipleGoodnessOfFits, self).__init__(*args, **kwargs)

        # check toys and toys_per_branch
        n_seqs = len(self.multi_datacards)
        if len(self.toys) == 1:
            self.toys *= n_seqs
        elif len(self.toys) != n_seqs:
            raise ValueError(
                "{!r}: number of toy values must either be one or match the amount "
                "of datacard sequences in --multi-datacards ({}), but got {}".format(
                    self, n_seqs, len(self.toys),
                ),
            )
        if len(self.toys_per_branch) == 1:
            self.toys_per_branch *= n_seqs
        elif len(self.toys_per_branch) != n_seqs:
            raise ValueError(
                "{!r}: number of toys_per_branch values must either be one or match "
                "the amount of datacard sequences in --multi-datacards ({}), but got {}".format(
                    self, n_seqs, len(self.toys_per_branch),
                ),
            )

    @property
    def toys_postfix(self):
        def tpl_to_str(tpl):
            vals = list(map(str, tpl))
            s = "_".join(vals[:5])
            if len(vals) > 5:
                s += "_" + law.util.create_hash(vals)
            return s

        return "t{}_tpb{}".format(tpl_to_str(self.toys), tpl_to_str(self.toys_per_branch))

    def requires(self):
        return [
            MergeGoodnessOfFit.req(self, datacards=datacards, toys=t, toys_per_branch=tpb, **kwargs)
            for datacards, t, tpb, kwargs in zip(
                self.multi_datacards, self.toys, self.toys_per_branch, self.get_multi_task_kwargs(),
            )
        ]

    def output(self):
        outputs = {}

        names = self.create_plot_names(["multigofs", self.get_output_postfix(), self.toys_postfix])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix(), self.toys_postfix])
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

        # load input data
        data = []
        for i, inp in enumerate(self.input()):
            d = inp.load(formatter="json")

            # add a default name
            d["name"] = "Cards {}".format(i + 1)

            data.append(d)

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
            data=data,
            algorithm=self.algorithm,
            n_bins=self.n_bins,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            pad_width=None if self.pad_width == law.NO_INT else self.pad_width,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            right_margin=None if self.right_margin == law.NO_INT else self.right_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )
