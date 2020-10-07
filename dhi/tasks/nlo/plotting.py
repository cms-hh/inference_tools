# coding: utf-8

"""
NLO Plotting tasks.
"""

import law
import luigi

from dhi.tasks.base import AnalysisTask
from dhi.tasks.nlo.base import POIScanTask1D, POIScanTask2D
from dhi.tasks.nlo.inference import MergeUpperLimits, MergeLikelihoodScan1D, MergeLikelihoodScan2D
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
            if output.path.endswith((".pdf", ".png")) and output.path not in view_paths:
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
        "terminal; default: empty",
    )
    campaign = luigi.ChoiceParameter(
        default="2017",
        choices=["2016", "2017", "2018", "FullRun2"],
        significant=False,
        description="the year/campaign (mainly for plotting); default: 2017",
    )


class PlotUpperLimits(PlotTask, POIScanTask1D):

    scale_xsec = luigi.BoolParameter(default=False, description="draw the limits on cross sections "
        "instead of signal strength parameters; default: False")
    scale_log = luigi.BoolParameter(default=False, description="apply log scaling to the y-axis; "
        "default: False")

    @classmethod
    def modify_param_values(cls, params):
        params = super(PlotUpperLimits, cls).modify_param_values(params)

        # scaling to xsec is only supported for kl and C2V
        if params.get("scale_xsec") and params.get("poi") not in ("kl", "C2V"):
            params["scale_xsec"] = False

        return params

    def requires(self):
        return MergeUpperLimits.req(self)

    def output(self):
        # postfix from xsec and log
        parts = []
        if self.scale_xsec:
            parts.append("xsec")
        if self.scale_log:
            parts.append("log")
        postfix = ("__" + "_".join(parts)) if parts else ""

        return self.local_target_dc("limits__{}{}.pdf".format(self.get_output_postfix(), postfix))

    @view_output_plots
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        data = self.input().load(formatter="numpy")["data"]
        limit_keys = [key for key in data.dtype.names if key.startswith("limit")]

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        theory_values = None
        is_xsec = self.scale_xsec
        if is_xsec:
            scale = br_hww_hbb * k_factor * 1000.0  # TODO: generalize this for different analyses
            if self.poi == "kl":
                formula = self.load_hh_model()[0].ggf_formula
                theory_values = []
                is_xsec = True
                for point in data:
                    xsec = get_ggf_xsec(formula, kl=point["kl"])
                    theory_values.append(xsec * scale)
                    for key in limit_keys:
                        point[key] *= xsec * scale
            elif self.poi == "C2V":
                formula = self.load_hh_model()[0].vbf_formula
                theory_values = []
                is_xsec = True
                for point in data:
                    xsec = get_vbf_xsec(formula, c2v=point["C2V"])
                    theory_values.append(xsec * scale)
                    for key in limit_keys:
                        point[key] *= xsec * scale
            else:
                # no scaling
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
            log=self.scale_log,
            campaign=self.campaign,
        )


class PlotLikelihoodScan1D(PlotTask, POIScanTask1D):
    def requires(self):
        return MergeLikelihoodScan1D.req(self)

    def output(self):
        return self.local_target_dc("nll1d__{}.pdf".format(self.get_output_postfix()))

    @view_output_plots
    def run(self):
        import numpy as np
        import numpy.lib.recfunctions as rec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        inp = self.input().load(formatter="numpy")
        data = inp["data"]
        poi_min = float(inp["poi_min"]) if not np.isnan(inp["poi_min"]) else None

        # insert a dnll2 column
        data = rec.append_fields(data, ["dnll2"], [data["delta_nll"] * 2.])

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.likelihoods_mpl import plot_likelihood_scan_1d

        plot_likelihood_scan_1d(
            path=output.path,
            poi=self.poi,
            data=data,
            poi_min=poi_min,
            campaign=self.campaign,
        )


class PlotLikelihoodScan2D(PlotTask, POIScanTask2D):
    def requires(self):
        return MergeLikelihoodScan2D.req(self)

    def output(self):
        return self.local_target_dc("nll2d__{}.pdf".format(self.get_output_postfix()))

    @view_output_plots
    def run(self):
        import numpy as np
        import numpy.lib.recfunctions as rec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        inp = self.input().load(formatter="numpy")
        data = inp["data"]
        poi1_min = float(inp["poi1_min"]) if not np.isnan(inp["poi1_min"]) else None
        poi2_min = float(inp["poi2_min"]) if not np.isnan(inp["poi2_min"]) else None

        # insert a dnll2 column
        data = rec.append_fields(data, ["dnll2"], [data["delta_nll"] * 2.])

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.likelihoods_mpl import plot_likelihood_scan_2d

        plot_likelihood_scan_2d(
            path=output.path,
            poi1=self.poi1,
            poi2=self.poi2,
            data=data,
            poi1_min=poi1_min,
            poi2_min=poi2_min,
            campaign=self.campaign,
        )
