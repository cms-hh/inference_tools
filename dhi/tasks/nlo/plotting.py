# coding: utf-8

"""
NLO Plotting tasks.
"""

import law
import luigi

from dhi.tasks.base import AnalysisTask
from dhi.tasks.nlo.base import POIScanTask1D
from dhi.tasks.nlo.inference import MergeUpperLimits, MergeLikelihoodScan1D
from dhi.config import br_hww_hbb, k_factor
from dhi.util import get_ggf_xsec, get_vbf_xsec


@law.decorator.factory(accept_generator=True)
def view_output_plots(fn, opts, task, *args, **kwargs):
    def before_call():
        return None

    def call(state):
        return fn(task, *args, **kwargs)

    def after_call(state):
        view_cmd = getattr(task, "view_cmd", None)
        if not view_cmd or view_cmd == law.NO_STR:
            return

        # prepare the view command
        if "{}" not in view_cmd:
            view_cmd += " {}"

        # collect all paths to view
        view_paths = []
        for output in law.util.flatten(task.output()):
            if not getattr(output, "path", None):
                continue
            if output.path.endswith((".pdf", ".png")):
                view_paths.append(output.path)

        # loop through paths and view them
        for path in view_paths:
            task.publish_message("showing {}".format(path))
            law.util.interruptable_popen(view_cmd.format(path), shell=True, executable="/bin/bash")

    return before_call, call, after_call


class PlotTask(AnalysisTask):

    plot_flavor = luigi.ChoiceParameter(
        default="root",
        choices=["root", "mpl"],
        significant=False,
        description="the plot flavor; choices: root,mpl; default: root",
    )
    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots in the "
        "terminal, default: empty",
    )
    campaign = luigi.ChoiceParameter(
        default="2017",
        choices=["2016", "2017", "2018", "FullRun2"],
        significant=False,
        description="the year/campaign (mainly for plotting); default: 2017",
    )


class PlotUpperLimits(PlotTask, POIScanTask1D):
    def requires(self):
        return MergeUpperLimits.req(self)

    def output(self):
        return self.local_target_dc(
            "limits__{}_n{}_{}_{}.pdf".format(self.poi, self.points, *self.poi_range)
        )

    @view_output_plots
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        data = self.input().load(formatter="numpy")["data"]

        # rescale from limit on r to limit on xsec, depending on the poi
        limit_keys = [key for key in data.dtype.names if key.startswith("limit")]
        scale = br_hww_hbb * k_factor * 1000.0  # TODO: generalize this for different analyses

        if self.poi == "kl":
            formula = self.load_hh_model()[0].ggf_formula
            theory_values = []
            is_xsec = True
            for point in data:
                xsec = get_ggf_xsec(formula, kl=point["kl"])
                theory_values.append(xsec)
                for key in limit_keys:
                    point[key] *= xsec * scale

        elif self.poi == "C2V":
            formula = self.load_hh_model()[0].vbf_formula
            theory_values = []
            is_xsec = True
            for point in data:
                xsec = get_vbf_xsec(formula, c2v=point["C2V"])
                theory_values.append(xsec)
                for key in limit_keys:
                    point[key] *= xsec * scale

        else:
            # no scaling
            theory_values = None
            is_xsec = False

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.limits_mpl import plot_limit_scan

        plot_limit_scan(
            path=output.path,
            poi=self.poi,
            data=data,
            theory_values=theory_values,
            is_xsec=is_xsec,
            campaign=self.campaign,
        )


class PlotLikelihoodScan1D(PlotTask, POIScanTask1D):
    def requires(self):
        return MergeLikelihoodScan1D.req(self)

    def output(self):
        return self.local_target_dc(
            "nll__{}_n{}_{}_{}.pdf".format(self.poi, self.points, *self.poi_range)
        )

    @view_output_plots
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        data = self.input().load(formatter="numpy")["data"]

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.scan1d_mpl import plot_likelihood1d

        plot_likelihood1d(
            path=output.path,
            poi=self.poi,
            data=data,
            campaign=self.campaign,
        )
