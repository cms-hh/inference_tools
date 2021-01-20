# coding: utf-8

"""
Tasks for creating post fit plots.
"""

import law
import luigi

from dhi.tasks.base import view_output_plots
from dhi.tasks.combine import CombineCommandTask, POITask, POIPlotTask, CreateWorkspace


class PostFitShapes(POITask, CombineCommandTask):

    pois = law.CSVParameter(
        default=("r",),
        unique=True,
        sort=True,
        choices=POITask.r_pois,
        description="names of POIs; choices: {}; default: (r,)".format(",".join(POITask.r_pois)),
    )

    force_n_pois = 1

    run_command_in_tmp = True

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("fitdiagnostics__{}.root".format(self.get_output_postfix()))

    def build_command(self):
        return (
            "combine -M FitDiagnostics {workspace}"
            " -m {self.mass}"
            " -v 1"
            " -t -1"
            " --expectSignal 1"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --saveShapes"
            " --saveWithUncertainties"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv fitDiagnostics.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )


class PlotPostfitSOverB(POIPlotTask):

    pois = PostFitShapes.pois
    bins = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(8,),
        significant=False,
        description="comma-separated list of bin edges to use; when a single number is passed, a "
        "automatic binning is applied with that number of bins; default: (8,)",
    )
    ratio_min = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the lower y-axis limit of the ratio plot; no default",
    )
    ratio_max = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the upper y-axis limit of the ratio plot; no default",
    )
    x_min = None
    x_max = None

    force_n_pois = 1

    def requires(self):
        return PostFitShapes.req(self)

    def output(self):
        name = self.create_plot_name(["postfitsoverb", self.get_output_postfix()])
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # get the path to the fit diagnostics file
        fit_diagnostics_path = self.input().path

        # call the plot function
        self.call_plot_func("dhi.plots.postfit_shapes.plot_s_over_b",
            path=output.path,
            poi=self.pois[0],
            fit_diagnostics_path=fit_diagnostics_path,
            bins=self.bins if len(self.bins) > 1 else int(self.bins[0]),
            y1_min=self.get_axis_limit("y_min"),
            y1_max=self.get_axis_limit("y_max"),
            y2_min=self.get_axis_limit("ratio_min"),
            y2_max=self.get_axis_limit("ratio_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )