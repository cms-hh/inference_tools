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


class GoodnessOfFitBase(POITask):

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

    @property
    def toys_postfix(self):
        postfix = "t{}_pt{}".format(self.toys, self.toys_per_task)
        if self.frequentist_toys:
            postfix += "_freq"
        return postfix


class GoodnessOfFit(GoodnessOfFitBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def __init__(self, *args, **kwargs):
        super(GoodnessOfFit, self).__init__(*args, **kwargs)

        # check that the number of toys is positive
        if self.toys <= 0:
            raise ValueError("{!r}: number of toys must be positive for GOF tests".format(self))

        # print a warning when the saturated algorithm is use without frequentist toys
        if self.algorithm == "saturated" and not self.frequentist_toys:
            self.logger.warning("it is recommended for goodness-of-fit tests with the "
                "'saturated' algorithm to use frequentiest toys, so please consider adding "
                "--frequentist-toys to the {} task".format(self.__class__.__name__))

    def create_branch_map(self):
        # the branch map refers to indices of toys in that branch, with 0 meaning the test on data
        all_toy_indices = list(range(1, self.toys + 1))
        toy_indices = [[0]] + list(law.util.iter_chunks(all_toy_indices, self.toys_per_task))
        return dict(enumerate(toy_indices))

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
        # toy options
        toy_opts = ""
        if self.branch > 0:
            toy_opts = "--toys {}".format(len(self.branch_data))
            if self.frequentist_toys:
                toy_opts += " --toysFrequentist"

        return (
            "combine -M GoodnessOfFit {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {toy_opts}"
            " --seed {self.branch}"
            " --algo {self.algorithm}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
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
            toy_opts=toy_opts,
        )

    def htcondor_output_postfix(self):
        postfix = super(GoodnessOfFit, self).htcondor_output_postfix()
        return "{}__{}".format(postfix, self.toys_postfix)


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
                self.logger.warning("input of branch {} at {} does not exist".format(
                    branch, inp.path))
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

    sort_pois = False

    def requires(self):
        return MergeGoodnessOfFit.req(self)

    def output(self):
        name = self.create_plot_name(["gofs", self.get_output_postfix(), self.toys_postfix])
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
            n_bins=self.n_bins,
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
    left_margin = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="the left margin of the pad in pixels; uses the default of the plot when "
        "empty; no default"
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
            raise ValueError("{!r}: number of toy values must either be one or match the amount "
                "of datacard sequences in --multi-datacards ({}), but got {}".format(
                    self, n_seqs, len(self.toys)))
        if len(self.toys_per_task) == 1:
            self.toys_per_task *= n_seqs
        elif len(self.toys_per_task) != n_seqs:
            raise ValueError("{!r}: number of toys_per_task values must either be one or match the "
                "amount of datacard sequences in --multi-datacards ({}), but got {}".format(
                    self, n_seqs, len(self.toys_per_task)))

    @property
    def toys_postfix(self):
        tpl_to_str = lambda tpl: "_".join(map(str, tpl))
        postfix = "t{}_pt{}".format(tpl_to_str(self.toys), tpl_to_str(self.toys_per_task))
        if self.frequentist_toys:
            postfix += "_freq"
        return postfix

    def requires(self):
        return [
            MergeGoodnessOfFit.req(self, datacards=datacards, toys=t, toys_per_task=tpt)
            for datacards, t, tpt in zip(self.multi_datacards, self.toys, self.toys_per_task)
        ]

    def output(self):
        name = self.create_plot_name(["multigofs", self.get_output_postfix(), self.toys_postfix])
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
            n_bins=self.n_bins,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
