# coding: utf-8

"""
Tasks related to significance calculation.
"""

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    CombineCommandTask,
    POITask,
    POIPlotTask,
    CreateWorkspace,
)


class GoodnessOfFit(POITask, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    toys = luigi.IntParameter(
        default=1,
        description="the positive number of toys to sample; default: 1",
    )
    toys_per_task = luigi.IntParameter(
        default=1,
        description="the number of toys to generate per task; the number of tasks in this workflow "
        "is the number of total toys divided by this number; default: 1"
    )
    algorithm = luigi.ChoiceParameter(
        default="saturated",
        choices=["saturated", "KS", "AD"],
        description="the algorithm to use; possible choices are 'saturated', 'KS' and 'AD'; "
        "default: saturated",
    )

    unblinded = None

    run_command_in_tmp = True

    def __init__(self, *args, **kwargs):
        super(GoodnessOfFit, self).__init__(*args, **kwargs)

        # check that the number of toys is positive
        if self.toys <= 0:
            raise ValueError("the number of toys must be positive for goodness-of-fit tests")

    def create_branch_map(self):
        # the branch map refers to indices of toys in that branch, with 0 meaning the test on data
        all_toy_indices = list(range(1, self.toys + 1))
        toy_indices = [[0]] + list(law.util.iter_chunks(all_toy_indices, self.toys_per_task))
        return dict(enumerate(toy_indices))

    def store_parts(self):
        parts = super(GoodnessOfFit, self).store_parts()
        parts["gof"] = self.algorithm
        return parts

    def workflow_requires(self):
        reqs = super(GoodnessOfFit, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        if self.branch == 0:
            postfix = "b0_data"
        else:
            postfix = "b{}_toy{}To{}".format(self.branch, self.branch_data[0], self.branch_data[-1])
        name = self.join_postfix(["gof", self.get_output_postfix(), postfix])
        return self.local_target(name + ".root")

    def build_command(self):
        # skip toys for the measurement (branch 0)
        branch_opts = "" if self.branch == 0 else "--toys {}".format(len(self.branch_data))

        return (
            "combine -M GoodnessOfFit {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {branch_opts}"
            " --seed {self.branch}"
            " --algo {self.algorithm}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.GoodnessOfFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            branch_opts=branch_opts,
        )

    def htcondor_output_postfix(self):
        postfix = super(GoodnessOfFit, self).htcondor_output_postfix()
        return "{}__t{}_p{}".format(postfix, self.toys, self.toys_per_task)


class MergeGoodnessOfFit(POITask):

    toys = GoodnessOfFit.toys
    toys_per_task = GoodnessOfFit.toys_per_task
    algorithm = GoodnessOfFit.algorithm

    def store_parts(self):
        parts = super(MergeGoodnessOfFit, self).store_parts()
        parts["gof"] = self.algorithm
        return parts

    def requires(self):
        return GoodnessOfFit.req(self)

    def output(self):
        toys_postfix = "t{}_pt{}".format(self.toys, self.toys_per_task)
        name = self.join_postfix(["gofs", self.get_output_postfix(), toys_postfix])
        return self.local_target(name + ".json")

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        # store gof values for the data measurement and for the toys
        data = {"data": None, "toys": []}

        # load values
        for b, inp in self.input()["collection"].targets.items():
            values = inp.load(formatter="uproot")["limit"].array("limit")
            if b == 0:
                data["data"] = float(values[0])
            else:
                data["toys"].extend(values.tolist())

        # save the result as json
        self.output().dump(data, formatter="json")


class PlotGoodnessOfFit(POIPlotTask):

    toys = GoodnessOfFit.toys
    toys_per_task = GoodnessOfFit.toys_per_task
    algorithm = GoodnessOfFit.algorithm

    z_min = None
    z_max = None

    def store_parts(self):
        parts = super(PlotGoodnessOfFit, self).store_parts()
        parts["gof"] = self.algorithm
        return parts

    def requires(self):
        return MergeGoodnessOfFit.req(self)

    def output(self):
        toys_postfix = "t{}_pt{}".format(self.toys, self.toys_per_task)
        name = self.create_plot_name(["gofs", self.get_output_postfix(), toys_postfix])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load input data
        gof_data = self.input().load(formatter="json")

        # call the plot function
        self.call_plot_func(
            "dhi.plots.gof.plot_gof_distribution",
            path=output.path,
            data=gof_data["data"],
            toys=gof_data["toys"],
            algorithm=self.algorithm,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotMultipleGoodnessOfFits(PlotGoodnessOfFit, MultiDatacardTask):

    toys = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1,),
        description="comma-separated list of positive amounts of toys per datacard sequence in "
        "--multi-datacards; when one value is given, it is used for all datacard sequences; "
        "default: (1,)",
    )
    toys_per_task = law.CSVParameter(
        cls=luigi.IntParameter,
        default=(1,),
        description="comma-separated list of numbers per datacard sequence in --multi-datacards to "
        "define the amount of toys to generate per task; when one value is given, it is used for "
        "all datacard sequences; default: (1,)",
    )

    y_min = None
    y_max = None

    def __init__(self, *args, **kwargs):
        super(PlotMultipleGoodnessOfFits, self).__init__(*args, **kwargs)

        # check toys and toys_per_task
        n_seqs = len(self.multi_datacards)
        if len(self.toys) == 1:
            self.toys *= n_seqs
        elif len(self.toys) != n_seqs:
            raise ValueError("the number of --toys values must either be one or match the amount "
                "of datacard sequences in --multi-datacards ({}), but got {}".format(
                n_seqs, len(self.toys)))
        if len(self.toys_per_task) == 1:
            self.toys_per_task *= n_seqs
        elif len(self.toys_per_task) != n_seqs:
            raise ValueError("the number of --toys-per-task values must either be one or match the "
                "amount of datacard sequences in --multi-datacards ({}), but got {}".format(
                n_seqs, len(self.toys_per_task)))

    def requires(self):
        return [
            MergeGoodnessOfFit.req(self, datacards=datacards, toys=t, toys_per_task=tpt)
            for datacards, t, tpt in zip(self.multi_datacards, self.toys, self.toys_per_task)
        ]

    def output(self):
        tpl_to_str = lambda tpl: "_".join(map(str, tpl))
        toys_postfix = "t{}_pt{}".format(tpl_to_str(self.toys), tpl_to_str(self.toys_per_task))
        name = self.create_plot_name(["multigofs", self.get_output_postfix(), toys_postfix])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

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
            "dhi.plots.gof.plot_gofs",
            path=output.path,
            data=data,
            algorithm=self.algorithm,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
