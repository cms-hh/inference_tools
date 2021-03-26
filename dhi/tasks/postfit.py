# coding: utf-8

"""
Tasks for working with postfit results.
"""

import copy

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import CombineCommandTask, POITask, POIPlotTask, CreateWorkspace


class FitDiagnostics(POITask, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    SAVE_FLAGS = ("Shapes", "WithUncertainties", "Normalizations", "Workspace", "Toys")

    pois = law.CSVParameter(
        default=("r",),
        unique=True,
        sort=True,
        choices=POITask.r_pois,
        description="names of POIs; choices: {}; default: (r,)".format(",".join(POITask.r_pois)),
    )
    skip_b_only = luigi.BoolParameter(
        default=True,
        description="when True, skip performing the background-only fit; default: True",
    )
    skip_save = law.CSVParameter(
        default=tuple(),
        choices=SAVE_FLAGS,
        description="comma-separated flags to skip passing to combine as '--save<flag>'; "
            "choices: {}; no default".format(",".join(SAVE_FLAGS)),
    )

    force_n_pois = 1
    allow_parameter_values_in_pois = True

    run_command_in_tmp = True

    def create_branch_map(self):
        return [""]  # single branch with empty data

    def workflow_requires(self):
        reqs = super(FitDiagnostics, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        parts = [self.get_output_postfix()]
        if not self.skip_b_only:
            parts.append("withBOnly")
        if self.skip_save:
            parts.append(map("no{}".format, self.skip_save))
        postfix = self.join_postfix(parts)

        return {
            "result": self.local_target("result__{}.root".format(postfix)),
            "diagnostics": self.local_target("fitdiagnostics__{}.root".format(postfix)),
        }

    @property
    def blinded_args(self):
        if self.unblinded:
            return ""
        else:
            return "--toys {self.toys}".format(self=self)

    def build_command(self):
        outputs = self.output()

        # prepare optional flags
        flags = []
        if self.skip_b_only:
            flags.append("--skipBOnlyFit")
        for save_flag in self.SAVE_FLAGS:
            if save_flag not in self.skip_save:
                flags.append("--save{}".format(save_flag))

        return (
            "combine -M FitDiagnostics {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {flags}"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.FitDiagnostics.mH{self.mass_int}.123456.root {output_result}"
            " && "
            "mv fitDiagnosticsTest.root {output_diagnostics}"
        ).format(
            self=self,
            workspace=self.input().path,
            output_result=outputs["result"].path,
            output_diagnostics=outputs["diagnostics"].path,
            flags=" ".join(flags),
        )


class PlotPostfitSOverB(POIPlotTask):

    pois = FitDiagnostics.pois
    bins = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(8,),
        significant=False,
        description="comma-separated list of bin edges to use; when a single number is passed, a "
        "automatic binning is applied with that number of bins; default: (8,)",
    )
    signal_scale = luigi.FloatParameter(
        default=100.0,
        significant=False,
        description="scale the postfit signal by this value; default: 100.",
    )
    ratio_min = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the lower y-axis limit of the ratio plot; no default",
    )
    ratio_max = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the upper y-axis limit of the ratio plot; no default",
    )
    x_min = None
    x_max = None
    z_max = None
    z_max = None

    force_n_pois = 1

    def requires(self):
        return FitDiagnostics.req(self)

    def output(self):
        name = self.create_plot_name(["postfitsoverb", self.get_output_postfix()])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # get the path to the fit diagnostics file
        fit_diagnostics_path = self.input()["collection"][0]["diagnostics"].path

        # call the plot function
        self.call_plot_func(
            "dhi.plots.postfit_shapes.plot_s_over_b",
            path=output.path,
            poi=self.pois[0],
            fit_diagnostics_path=fit_diagnostics_path,
            bins=self.bins if len(self.bins) > 1 else int(self.bins[0]),
            y1_min=self.get_axis_limit("y_min"),
            y1_max=self.get_axis_limit("y_max"),
            y2_min=self.get_axis_limit("ratio_min"),
            y2_max=self.get_axis_limit("ratio_max"),
            signal_scale=self.signal_scale,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotNuisanceLikelihoodScans(POIPlotTask):

    x_min = copy.copy(POIPlotTask.x_min)
    x_max = copy.copy(POIPlotTask.x_max)
    x_min._default = -2.
    x_max._default = 2.
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    only_parameters = law.CSVParameter(
        default=(),
        significant=False,
        description="patterns of parameters to include; skips all others; no default",
    )
    skip_parameters = law.CSVParameter(
        default=(),
        significant=False,
        description="patterns of parameters to skip; no default",
    )
    parameters_per_page = luigi.IntParameter(
        default=1,
        description="number of parameters per page; creates a single page when < 1; default: 1",
    )

    file_type = "pdf"
    y_min = None
    y_max = None
    z_min = None
    z_max = None

    force_n_pois = 1

    def requires(self):
        return FitDiagnostics.req(self, skip_save=("WithUncertainties",))

    def output(self):
        parts = ["nlls", "{}To{}".format(self.x_min, self.x_max), self.get_output_postfix()]
        if self.y_log:
            parts.append("log")

        return self.local_target(self.create_plot_name(parts))

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # get input targets
        inputs = self.input()
        fit_result = inputs["collection"][0]["result"]
        fit_diagnostics = inputs["collection"][0]["diagnostics"]

        # open the result file to load the workspace and other objects
        with fit_result.load("READ", formatter="root") as result_file:
            # get workspace
            w = result_file.Get("w")

            # load the dataset
            dataset = w.data("data_obs") if self.unblinded else result_file.Get("toys/toy_asimov")

            # call the plot function
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_nuisance_likelihood_scans",
                path=output.path,
                poi=self.pois[0],
                workspace=w,
                dataset=dataset,
                fit_diagnostics_path=fit_diagnostics.path,
                fit_name="fit_s",
                only_parameters=self.only_parameters,
                skip_parameters=self.skip_pameters,
                parameters_per_page=self.parameters_per_page,
                x_min=self.x_min,
                x_max=self.x_max,
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )