# coding: utf-8

"""
NLO Plotting tasks.
"""

import law
import luigi

from dhi.tasks.base import AnalysisTask
from dhi.tasks.nlo.base import POITask1D, POIScanTask1D, POIScanTask2D
from dhi.tasks.nlo.inference import (
    MergeUpperLimits, MergeLikelihoodScan1D, MergeLikelihoodScan2D, MergePullsAndImpacts,
)
from dhi.config import br_hh, k_factor, nuisance_labels
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
    file_type = luigi.ChoiceParameter(
        default="pdf",
        choices=["pdf", "png"],
        description="the type of the output plot file; choices: pdf,png; default: pdf",
    )
    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots right in the "
        "terminal; default: empty",
    )
    campaign = luigi.ChoiceParameter(
        default="2017",
        choices=["2016", "2017", "2018", "FullRun2"],
        significant=False,
        description="the year/campaign (mainly for plotting); default: 2017",
    )


class PlotUpperLimits(PlotTask, POIScanTask1D):

    xsec = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "", "pb", "fb"],
        description="convert limits to cross sections in this unit; only supported for poi's kl "
        "and C2V; choices: '',pb,fb; default: empty",
    )
    br = luigi.Parameter(
        default="1",
        description="branching ratio or name of a branching ratio stored at dhi.config.br_hh to "
        "scale the cross section when xsec is used; default: 1.0",
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    @classmethod
    def modify_param_values(cls, params):
        params = super(PlotUpperLimits, cls).modify_param_values(params)

        # scaling to xsec is only supported for kl and C2V
        if params.get("xsec") not in ("", law.NO_STR) and params.get("poi") not in ("kl", "C2V"):
            cls.logger.warning("xsec conversion is only supported for POIs 'kl' and 'C2V'")
            params["xsec"] = law.NO_STR

        return params

    def requires(self):
        return MergeUpperLimits.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if str(self.br) != "1":
                parts.append(str(self.br))
        if self.y_log:
            parts.append("log")
        postfix = ("__" + "_".join(parts)) if parts else ""

        return self.local_target_dc("limits__{}{}.{}".format(
            self.get_output_postfix(), postfix, self.file_type))

    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit data
        data = self.input().load(formatter="numpy")["data"]
        limit_keys = [key for key in data.dtype.names if key.startswith("limit")]

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        theory_values = None
        xsec_unit = None
        if self.xsec in ["pb", "fb"]:
            # determine the scaling factor
            scale = {"pb": 1., "fb": 1000.}[self.xsec]
            scale *= k_factor  # NLO -> NNLO
            scale *= br_hh[self.br] if self.br in br_hh else float(self.br)

            # perform the scaling
            if self.poi == "kl":
                xsec_unit = self.xsec
                formula = self.load_hh_model()[0].ggf_formula
                theory_values = []
                for point in data:
                    xsec = get_ggf_xsec(formula, kl=point["kl"])
                    theory_values.append(xsec * scale)
                    for key in limit_keys:
                        point[key] *= xsec * scale
            elif self.poi == "C2V":
                xsec_unit = self.xsec
                formula = self.load_hh_model()[0].vbf_formula
                theory_values = []
                for point in data:
                    xsec = get_vbf_xsec(formula, c2v=point["C2V"])
                    theory_values.append(xsec * scale)
                    for key in limit_keys:
                        point[key] *= xsec * scale

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.limits_root import plot_limit_scan
        else:
            from dhi.plots.limits_mpl import plot_limit_scan

        plot_limit_scan(
            path=output.path,
            poi=self.poi,
            data=data,
            theory_values=theory_values,
            xsec_unit=xsec_unit,
            y_log=self.y_log,
            campaign=self.campaign,
        )


class PlotLikelihoodScan1D(PlotTask, POIScanTask1D):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    def requires(self):
        return MergeLikelihoodScan1D.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.y_log:
            parts.append("log")
        postfix = ("__" + "_".join(parts)) if parts else ""

        return self.local_target_dc("nll1d__{}{}.{}".format(
            self.get_output_postfix(), postfix, self.file_type))

    @view_output_plots
    @law.decorator.safe_output
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
        data = rec.append_fields(data, ["dnll2"], [data["delta_nll"] * 2.0])

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.likelihoods_root import plot_likelihood_scan_1d
        else:
            from dhi.plots.likelihoods_mpl import plot_likelihood_scan_1d

        plot_likelihood_scan_1d(
            path=output.path,
            poi=self.poi,
            data=data,
            poi_min=poi_min,
            campaign=self.campaign,
            y_log=self.y_log,
        )


class PlotLikelihoodScan2D(PlotTask, POIScanTask2D):

    z_log = luigi.BoolParameter(
        default=True,
        description="apply log scaling to the z-axis; default: True",
    )

    def requires(self):
        return MergeLikelihoodScan2D.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.z_log:
            parts.append("log")
        postfix = ("__" + "_".join(parts)) if parts else ""

        return self.local_target_dc("nll2d__{}{}.{}".format(
            self.get_output_postfix(), postfix, self.file_type))

    @view_output_plots
    @law.decorator.safe_output
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
        data = rec.append_fields(data, ["dnll2"], [data["delta_nll"] * 2.0])

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.likelihoods_root import plot_likelihood_scan_2d
        else:
            from dhi.plots.likelihoods_mpl import plot_likelihood_scan_2d

        plot_likelihood_scan_2d(
            path=output.path,
            poi1=self.poi1,
            poi2=self.poi2,
            data=data,
            poi1_min=poi1_min,
            poi2_min=poi2_min,
            campaign=self.campaign,
            z_log=self.z_log,
        )


class PlotPullsAndImpacts(PlotTask, POITask1D):

    mc_stats = MergePullsAndImpacts.mc_stats
    parameters_per_page = luigi.IntParameter(
        default=-1,
        description="number of parameters per page; creates a single page when < 1; only applied "
        "for file type 'pdf'; default: -1",
    )
    skip_parameters = law.CSVParameter(
        default=(),
        description="list of parameters or files containing parameters line-by-line that should be "
        "skipped; supports patterns; default: empty",
    )
    order_parameters = law.CSVParameter(
        default=(),
        description="list of parameters or files containing parameters line-by-line for ordering; "
        "supports patterns; default: empty",
    )
    order_by_impact = luigi.BoolParameter(
        default=False,
        description="when True, --parameter-order is neglected and parameters are ordered by "
        "absolute maximum impact; default: False",
    )

    def __init__(self, *args, **kwargs):
        super(PlotPullsAndImpacts, self).__init__(*args, **kwargs)

        # complain when parameters_per_page is set for non pdf file types
        if self.parameters_per_page > 0 and self.file_type != "pdf":
            self.logger.warning("parameters_per_page is not supported for file_type {}".format(
                self.file_type))
            self.parameters_per_page = -1

    def requires(self):
        return MergePullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = [self.poi]
        if self.mc_stats:
            parts.append("mcstats")
        postfix = "__".join(parts)

        return self.local_target_dc("pulls_impacts__{}.{}".format(postfix, self.file_type))

    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load input data
        data = self.input().load(formatter="json")

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.pulls_impacts_root import plot_pulls_impacts

        plot_pulls_impacts(
            path=output.path,
            data=data,
            parameters_per_page=self.parameters_per_page,
            skip_parameters=self.skip_parameters,
            order_parameters=self.order_parameters,
            order_by_impact=self.order_by_impact,
            labels=nuisance_labels,
            campaign=self.campaign,
        )
