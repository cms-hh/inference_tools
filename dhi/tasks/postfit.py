# coding: utf-8

"""
Tasks for working with postfit results.
"""

import copy
import enum

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import CombineCommandTask, POITask, POIPlotTask, CreateWorkspace
from dhi.tasks.snapshot import Snapshot, SnapshotUser
from dhi.tasks.pulls_impacts import PlotPullsAndImpacts


class SAVEFLAGS(str, enum.Enum):

    Shapes = "Shapes"
    WithUncertainties = "WithUncertainties"
    Normalizations = "Normalizations"
    Workspace = "Workspace"
    Toys = "Toys"
    NLL = "NLL"
    OverallShapes = "OverallShapes"

    @classmethod
    def tolist(cls):
        return list(map(lambda x: x.value, cls))


class FitDiagnostics(POITask, CombineCommandTask, SnapshotUser, law.LocalWorkflow, HTCondorWorkflow):

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
        choices=SAVEFLAGS.tolist(),
        sort=True,
        description="comma-separated flags to skip passing to combine as '--save<flag>'; "
        "choices: {}; no default".format(",".join(SAVEFLAGS)),
    )

    force_n_pois = 1
    allow_parameter_values_in_pois = True

    run_command_in_tmp = True

    def create_branch_map(self):
        # single branch with empty data
        return [None]

    def workflow_requires(self):
        reqs = super(FitDiagnostics, self).workflow_requires()
        reqs["workspace"] = CreateWorkspace.req(self)
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self)
        return reqs

    def requires(self):
        reqs = {"workspace": CreateWorkspace.req(self)}
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, branch=0)
        return reqs

    def get_output_postfix(self, join=True):
        parts = super(FitDiagnostics, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts

    def output(self):
        parts = []
        if not self.skip_b_only:
            parts.append("withBOnly")
        if self.skip_save:
            parts.append(map("not{}".format, sorted(self.skip_save)))

        name = lambda prefix: self.join_postfix([prefix, self.get_output_postfix(), parts])
        return {
            "result": self.local_target(name("result") + ".root"),
            "diagnostics": self.local_target(name("fitdiagnostics") + ".root"),
        }

    def build_command(self):
        # get the workspace to use and define snapshot args
        if self.use_snapshot:
            workspace = self.input()["snapshot"].path
            snapshot_args = " --snapshotName MultiDimFit"
        else:
            workspace = self.input()["workspace"].path
            snapshot_args = ""

        # arguments for un/blinding
        blinded_args = ""
        if self.blinded:
            blinded_args = "--toys {self.toys}".format(self=self)

        # prepare optional flags
        flags = []
        if self.skip_b_only:
            flags.append("--skipBOnlyFit")
        for save_flag in SAVEFLAGS:
            if save_flag not in self.skip_save:
                flags.append("--save{}".format(save_flag))

        outputs = self.output()
        return (
            "combine -M FitDiagnostics {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {flags}"
            " {snapshot_args}"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.FitDiagnostics.mH{self.mass_int}{postfix}.root {output_result}"
            " && "
            "mv fitDiagnosticsTest.root {output_diagnostics}"
        ).format(
            self=self,
            workspace=workspace,
            postfix="" if SAVEFLAGS.Toys in self.skip_save else ".123456",
            output_result=outputs["result"].path,
            output_diagnostics=outputs["diagnostics"].path,
            blinded_args=blinded_args,
            snapshot_args=snapshot_args,
            flags=" ".join(flags),
        )


class PostfitPlotBase(POIPlotTask, SnapshotUser):

    def get_output_postfix(self, join=True):
        parts = super(PostfitPlotBase, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts


class PlotPostfitSOverB(PostfitPlotBase):

    pois = FitDiagnostics.pois
    bins = law.CSVParameter(
        cls=luigi.FloatParameter,
        default=(8,),
        significant=False,
        description="comma-separated list of bin edges to use; when a single number is passed, a "
        "automatic binning is applied with that number of bins; default: (8,)",
    )
    show_best_fit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, show the label of the best fit value; default: False",
    )
    signal_superimposed = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, draw the signal at the top pad superimposed; default: False",
    )
    signal_scale = luigi.FloatParameter(
        default=1.0,
        significant=False,
        description="scale the postfit signal by this value; default: 1.0",
    )
    signal_scale_ratio = luigi.FloatParameter(
        default=1.0,
        significant=False,
        description="scale the postfit signal in the ratio plot by this value; only considered "
        "when drawing the signal superimposed; default: 1.0",
    )
    hide_signal = luigi.BoolParameter(
        default=False,
        significant=False,
        description="hide the signal contribution completely; default: False",
    )
    hide_uncertainty = luigi.BoolParameter(
        default=False,
        significant=False,
        description="do not show postfit uncertainties (and also do not require FitDiagnostics to "
        "produce them); default: False",
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
    prefit = luigi.BoolParameter(
        default=False,
        description="plot prefit distributions and uncertainties instead of postfit ones; only "
        "available when not --unblinded; default: False",
    )
    categories = law.CSVParameter(
        default=(),
        description="comma-separated list of category names or name patterns to select; consider "
        "adjusting --campaign accordingly; all categories are used when empty; default: empty",
    )
    x_min = None
    x_max = None
    z_max = None
    z_max = None

    force_n_pois = 1

    def __init__(self, *args, **kwargs):
        super(PlotPostfitSOverB, self).__init__(*args, **kwargs)

        # show a warning when unblinded, not in paper mode and not hiding the best fit value
        if self.unblinded and not self.paper and self.show_best_fit:
            self.logger.warning("running unblinded but not hiding the best fit value")

    def requires(self):
        # normally, we would require FitDiagnostics not matter what, but since it can take ages to
        # complete, skip producing uncertainties when requested and the full fit does not exist yet
        full_fd = FitDiagnostics.req(self)
        if self.hide_uncertainty and not full_fd.complete():
            return FitDiagnostics.req(self, skip_save=("WithUncertainties",))
        return full_fd

    def output(self):
        parts = []
        if self.categories:
            parts.append(["cats"] + list(self.categories))

        name = "prefitsoverb" if self.prefit else "postfitsoverb"
        names = self.create_plot_names([name, self.get_output_postfix()] + parts)
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # get the path to the fit diagnostics file
        fit_diagnostics_path = self.input()["collection"][0]["diagnostics"].path

        # call the plot function
        self.call_plot_func(
            "dhi.plots.postfit_shapes.plot_s_over_b",
            paths=[outp.path for outp in outputs],
            poi=self.pois[0],
            fit_diagnostics_path=fit_diagnostics_path,
            bins=self.bins if len(self.bins) > 1 else int(self.bins[0]),
            signal_superimposed=self.signal_superimposed,
            signal_scale=self.signal_scale,
            signal_scale_ratio=self.signal_scale_ratio,
            show_signal=not self.hide_signal,
            show_uncertainty=not self.hide_uncertainty,
            show_best_fit=self.show_best_fit,
            categories=self.categories,
            y1_min=self.get_axis_limit("y_min"),
            y1_max=self.get_axis_limit("y_max"),
            y2_min=self.get_axis_limit("ratio_min"),
            y2_max=self.get_axis_limit("ratio_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            prefit=self.prefit,
            unblinded=self.unblinded,
            paper=self.paper,
        )


class PlotNuisanceLikelihoodScans(PostfitPlotBase):

    x_min = copy.copy(POIPlotTask.x_min)
    x_max = copy.copy(POIPlotTask.x_max)
    x_min._default = -2.0
    x_max._default = 2.0
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
    mc_stats = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, include MC stats nuisances as well; default: False",
    )
    parameters_per_page = luigi.IntParameter(
        default=1,
        description="number of parameters per page; creates a single page when < 1; default: 1",
    )
    sort_max = luigi.BoolParameter(
        default=False,
        description="when True, sort parameters by their hightest likelihood change in the scan "
        "range; mostly useful when the number of parameters per page is > 1; default: False",
    )
    show_diff = luigi.BoolParameter(
        default=False,
        description="when True, the x-axis shows differences of nuisance parameters with respect "
        "to the best fit value instead of absolute values; default: False",
    )
    labels = PlotPullsAndImpacts.labels

    mc_stats_patterns = ["*prop_bin*"]

    file_types = ("pdf",)
    z_min = None
    z_max = None

    force_n_pois = 1

    def requires(self):
        # normally, we would require FitDiagnostics without saved uncertainties no matter what,
        # but since it could be already complete, use it when it does exist
        full_fd = FitDiagnostics.req(self)
        if full_fd.complete():
            return full_fd
        return FitDiagnostics.req(self, skip_save=("WithUncertainties",), _prefer_cli=["skip_save"])

    def output(self):
        parts = ["nlls", "{}To{}".format(self.x_min, self.x_max), self.get_output_postfix()]
        if self.show_diff:
            parts.append("diffs")
        if self.y_log:
            parts.append("log")
        if self.sort_max:
            parts.append("sorted")

        names = self.create_plot_names(parts)
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # get input targets
        inputs = self.input()
        fit_result = inputs["collection"][0]["result"]
        fit_diagnostics = inputs["collection"][0]["diagnostics"]

        # skip parameter patterns
        skip_parameters = list(self.skip_parameters)
        if not self.mc_stats:
            skip_parameters.extend(self.mc_stats_patterns)

        # open the result file to load the workspace and other objects
        with fit_result.load("READ", formatter="root") as result_file:
            # get workspace
            w = result_file.Get("w")

            # load the dataset
            dataset = w.data("data_obs") if self.unblinded else result_file.Get("toys/toy_asimov")

            # call the plot function
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_nuisance_likelihood_scans",
                paths=[outp.path for outp in outputs],
                poi=self.pois[0],
                workspace=w,
                dataset=dataset,
                fit_diagnostics_path=fit_diagnostics.path,
                fit_name="fit_s",
                only_parameters=self.only_parameters,
                skip_parameters=skip_parameters,
                parameters_per_page=self.parameters_per_page,
                sort_max=self.sort_max,
                show_diff=self.show_diff,
                labels=None if self.labels == law.NO_STR else self.labels,
                x_min=self.x_min,
                x_max=self.x_max,
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                paper=self.paper,
            )
