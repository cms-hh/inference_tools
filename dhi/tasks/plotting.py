# coding: utf-8

"""
NLO Plotting tasks.
"""

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask
from dhi.tasks.combine import (
    MultiDatacardTask, MultiHHModelTask, POITask1D, POIScanTask1D, POITask1DWithR,
    POIScanTask1DWithR, POIScanTask2D,
)
from dhi.tasks.inference import (
    UpperLimits, MergeUpperLimits, MergeLikelihoodScan1D, MergeLikelihoodScan2D,
    MergePullsAndImpacts, SignificanceScan, MergeSignificanceScan, PostFitShapes,
)
from dhi.config import br_hh, br_hh_names, nuisance_labels


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
    campaign = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="the campaign name used for plotting; no default",
    )
    x_min = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the lower x-axis limit; no default",
    )
    x_max = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the upper x-axis limit; no default",
    )
    y_min = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the lower y-axis limit; no default",
    )
    y_max = luigi.FloatParameter(
        default=-1000.,
        significant=False,
        description="the upper y-axis limit; no default",
    )

    def create_plot_name(self, *parts):
        # join lists/tuples in parts by "_"
        _parts = []
        for part in parts:
            if isinstance(part, (list, tuple)):
                part = "_".join(str(p) for p in part)
            if part not in ("", None):
                _parts.append(str(part))

        # join parts by "__" and append the file type
        name = "__".join(_parts) + "." + self.file_type

        return name

    def get_axis_limit(self, value):
        if isinstance(value, six.string_types):
            value = getattr(self, value)
        return None if value == -1000. else value


class PlotUpperLimits(PlotTask, POIScanTask1DWithR):

    xsec = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "", "pb", "fb"],
        description="convert limits to cross sections in this unit; only supported for poi's kl "
        "and C2V; choices: '',pb,fb; default: empty",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: '',{}; default: empty".format(",".join(br_hh.keys())),
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    def __init__(self, *args, **kwargs):
        super(PlotUpperLimits, self).__init__(*args, **kwargs)

        # scaling to xsec is only supported for kl and C2V
        if self.xsec not in ("", law.NO_STR) and self.poi not in ("kl", "C2V"):
            raise Exception("xsec conversion is only supported for POIs 'kl' and 'C2V'")

    def requires(self):
        return MergeUpperLimits.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br not in (law.NO_STR, ""):
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name("limits", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load expected limit values
        expected_values = self.input().load(formatter="numpy")["data"]

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        theory_values = None
        xsec_unit = None
        hh_process = "HH"
        if self.poi in ["kl", "C2V"]:
            if self.xsec in ["pb", "fb"]:
                expected_values = self.convert_to_xsecs(expected_values, self.poi, self.xsec,
                    self.br)
                theory_values = self.get_theory_xsecs(expected_values[self.poi], self.poi,
                    self.xsec, self.br)
                xsec_unit = self.xsec
                if self.br in br_hh:
                    hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]
            else:
                # normalized values at one with errors
                theory_values = self.get_theory_xsecs(expected_values[self.poi], self.poi,
                    normalize=True)

        # some printing
        for v in range(-2, 4 + 1):
            if v in expected_values[self.poi]:
                record = expected_values[expected_values[self.poi] == v][0]
                self.publish_message("{} = {} -> {} {}".format(self.poi, v, record["limit"],
                    xsec_unit or "({})".format(self.r_poi)))

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.limits_root import plot_limit_scan
        else:
            from dhi.plots.limits_mpl import plot_limit_scan

        plot_limit_scan(
            path=output.path,
            poi=self.poi,
            expected_values=expected_values,
            theory_values=theory_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            hh_process=hh_process,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotMultipleUpperLimits(MultiDatacardTask, PlotUpperLimits):

    @classmethod
    def modify_param_values(cls, params):
        params = MultiDatacardTask.modify_param_values(params)
        params = PlotUpperLimits.modify_param_values(params)
        return params

    def requires(self):
        return [
            MergeUpperLimits.req(self, datacards=datacards)
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br not in (law.NO_STR, ""):
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name("multilimits", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values
        expected_values = []
        names = []
        theory_values = None
        xsec_unit = None
        hh_process = "HH"
        for i, inp in enumerate(self.input()):
            exp_values = inp.load(formatter="numpy")["data"]

            # rescale from limit on r to limit on xsec when requested, depending on the poi
            if self.poi in ["kl", "C2V"]:
                if self.xsec in ["pb", "fb"]:
                    exp_values = self.convert_to_xsecs(exp_values, self.poi, self.xsec, self.br)
                    xsec_unit = self.xsec
                    if self.br in br_hh:
                        hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]
                    if i == 0:
                        theory_values = self.get_theory_xsecs(exp_values[self.poi], self.poi,
                            self.xsec, self.br)
                elif i == 0:
                    # normalized values at one with errors
                    theory_values = self.get_theory_xsecs(exp_values[self.poi], self.poi,
                        normalize=True)

            expected_values.append(exp_values)
            names.append("datacards {}".format(i + 1))

        # set names if requested
        if self.datacard_names:
            names = list(self.datacard_names)

        # reoder if requested
        if self.datacard_order:
            expected_values = [expected_values[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # get the proper plot function and call it
        from dhi.plots.limits_root import plot_limit_scans

        plot_limit_scans(
            path=output.path,
            poi=self.poi,
            names=names,
            expected_values=expected_values,
            theory_values=theory_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            hh_process=hh_process,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotMultipleUpperLimitsByModel(PlotUpperLimits, MultiHHModelTask):

    def requires(self):
        return [
            MergeUpperLimits.req(self, hh_model=hh_model, hh_nlo=hh_nlo)
            for hh_model, hh_nlo in self.get_model_nlo_pairs()
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br not in (law.NO_STR, ""):
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name("multilimitsbymodel", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values
        expected_values = []
        names = []
        theory_values = None
        xsec_unit = None
        hh_process = "HH"
        for i, ((hh_model, hh_nlo), inp) in enumerate(zip(self.get_model_nlo_pairs(), self.input())):
            exp_values = inp.load(formatter="numpy")["data"]

            # rescale from limit on r to limit on xsec when requested, depending on the poi
            if self.poi in ["kl", "C2V"]:
                if self.xsec in ["pb", "fb"]:
                    exp_values = self._convert_to_xsecs(hh_model, hh_nlo, exp_values, self.poi,
                        self.xsec, self.br)
                    xsec_unit = self.xsec
                    if self.br in br_hh:
                        hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]
                    if i == 0:
                        theory_values = self._get_theory_xsecs(hh_model, hh_nlo,
                            exp_values[self.poi], self.poi, self.xsec, self.br)
                elif i == 0:
                    # normalized values at one with errors
                    theory_values = self._get_theory_xsecs(hh_model, hh_nlo, exp_values[self.poi],
                        self.poi, normalize=True)

            # prepare the name
            name = hh_model.split(":", 1)[-1].replace("_", " ")
            if name.startswith("model "):
                name = name.split("model ", 1)[-1]

            expected_values.append(exp_values)
            names.append(name)

        # get the proper plot function and call it
        from dhi.plots.limits_root import plot_limit_scans

        plot_limit_scans(
            path=output.path,
            poi=self.poi,
            names=names,
            expected_values=expected_values,
            theory_values=theory_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            hh_process=hh_process,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotUpperLimitsAtPOI(PlotTask, MultiDatacardTask, POITask1DWithR):

    xsec = PlotUpperLimits.xsec
    br = PlotUpperLimits.br
    x_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the x-axis; default: False",
    )
    h_lines = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(),
        significant=False,
        description="comma-separated vertical positions of horizontal lines; default: empty",
    )
    y_min = None
    y_max = None

    def requires(self):
        return [
            UpperLimits.req(self, poi_range=(self.poi_value, self.poi_value), poi_points=1,
                branch=0, datacards=datacards)
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br not in (law.NO_STR, ""):
                parts.append(self.br)
        if self.x_log:
            parts.append("log")

        name = self.create_plot_name("limitsatpoi", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values
        names = [self.poi, "limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]
        expected_values = np.array([
            MergeUpperLimits.load_limits(inp, poi_value=self.poi_value)
            for inp in self.input()
        ], dtype=[(name, np.float32) for name in names])

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        theory_values = [None] * expected_values.size
        xsec_unit = None
        hh_process = "HH"
        if self.poi in ["kl", "C2V"]:
            if self.xsec in ["pb", "fb"]:
                expected_values = self.convert_to_xsecs(expected_values, self.poi, self.xsec,
                    self.br)
                theory_values = self.get_theory_xsecs(expected_values[self.poi], self.poi,
                    self.xsec, self.br)
                xsec_unit = self.xsec
                if self.br in br_hh:
                    hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]
            else:
                # normalized values at one with errors
                theory_values = self.get_theory_xsecs(expected_values[self.poi], self.poi,
                    normalize=True)

        # fill data entries as expected by the plot function
        data = []
        for i, (exp_record, thy_record) in enumerate(zip(expected_values, theory_values)):
            data.append({
                "name": "datacards {}".format(i + 1),
                "expected": exp_record.tolist()[1:],
                "theory": thy_record and thy_record.tolist()[1:],
            })

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # get the proper plot function and call it
        from dhi.plots.limits_root import plot_limit_points

        plot_limit_points(
            path=output.path,
            data=data,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            x_log=self.x_log,
            xsec_unit=xsec_unit,
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            hh_process=hh_process,
            h_lines=self.h_lines,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotLikelihoodScan1D(PlotTask, POIScanTask1D):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    y_max = None

    def requires(self):
        return MergeLikelihoodScan1D.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name("nll1d", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np
        import numpy.lib.recfunctions as rec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load scan data
        data = self.input().load(formatter="numpy")
        expected_values = data["data"]
        poi_min = float(data["poi_min"]) if not np.isnan(data["poi_min"]) else None

        # insert a dnll2 column
        expected_values = np.array(rec.append_fields(expected_values, ["dnll2"],
            [expected_values["delta_nll"] * 2.0]))

        # TODO: when the poi is r*, we could perhaps also get and plot the theory uncertainty

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.likelihoods_root import plot_likelihood_scan_1d
        else:
            from dhi.plots.likelihoods_mpl import plot_likelihood_scan_1d

        plot_likelihood_scan_1d(
            path=output.path,
            poi=self.poi,
            expected_values=expected_values,
            poi_min=poi_min,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_log=self.y_log,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
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

        name = self.create_plot_name("nll2d", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np
        import numpy.lib.recfunctions as rec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load scan data
        data = self.input().load(formatter="numpy")
        expected_values = data["data"]
        poi1_min = float(data["poi1_min"]) if not np.isnan(data["poi1_min"]) else None
        poi2_min = float(data["poi2_min"]) if not np.isnan(data["poi2_min"]) else None

        # insert a dnll2 column
        expected_values = np.array(rec.append_fields(expected_values, ["dnll2"],
            [expected_values["delta_nll"] * 2.0]))

        # get the proper plot function and call it
        if self.plot_flavor == "root":
            from dhi.plots.likelihoods_root import plot_likelihood_scan_2d
        else:
            from dhi.plots.likelihoods_mpl import plot_likelihood_scan_2d

        plot_likelihood_scan_2d(
            path=output.path,
            poi1=self.poi1,
            poi2=self.poi2,
            expected_values=expected_values,
            poi1_min=poi1_min,
            poi2_min=poi2_min,
            x1_min=self.get_axis_limit("x_min"),
            x1_max=self.get_axis_limit("x_max"),
            x2_min=self.get_axis_limit("y_min"),
            x2_max=self.get_axis_limit("y_max"),
            z_log=self.z_log,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
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
    x_min = None
    x_max = None
    y_min = None
    y_max = None

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
    @law.decorator.log
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
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotBestFitAndExclusion(PlotTask, MultiDatacardTask, POIScanTask1DWithR):

    y_min = None
    y_max = None

    def requires(self):
        return [
            {
                "limits": MergeUpperLimits.req(self, datacards=datacards),
                "likelihoods": MergeLikelihoodScan1D.req(self, datacards=datacards),
            }
            for datacards in self.multi_datacards
        ]

    def output(self):
        name = self.create_plot_name("bestfitexclusion", self.get_poi_postfix())
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np
        import numpy.lib.recfunctions as rec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load input data
        data = []
        for i, inp in enumerate(self.input()):
            # load limits
            limit_data = inp["limits"].load(formatter="numpy")
            expected_limits = limit_data["data"]

            # load likelihoods
            ll_data = inp["likelihoods"].load(formatter="numpy")
            poi_min = float(ll_data["poi_min"]) if not np.isnan(ll_data["poi_min"]) else None
            expected_nll = ll_data["data"]
            # insert a dnll2 column
            expected_nll = np.array(rec.append_fields(expected_nll, ["dnll2"],
                [expected_nll["delta_nll"] * 2.0]))

            # store data
            data.append({
                "name": "datacards {}".format(i + 1),
                "expected_limits": expected_limits,
                "expected_nll": expected_nll,
                "poi_min": poi_min,
            })

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # get the proper plot function and call it
        from dhi.plots.misc_root import plot_bestfit_and_exclusion

        plot_bestfit_and_exclusion(
            path=output.path,
            data=data,
            poi=self.poi,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotSignificanceScan(PlotTask, POIScanTask1DWithR):

    def requires(self):
        return MergeSignificanceScan.req(self)

    def output(self):
        # additional postfix
        parts = []

        name = self.create_plot_name("significances", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load expected significances
        expected_values = self.input().load(formatter="numpy")["data"]

        # some printing
        for v in range(-2, 4 + 1):
            if v in expected_values[self.poi]:
                record = expected_values[expected_values[self.poi] == v][0]
                self.publish_message("{} = {} -> {:.4f} sigma".format(
                    self.poi, v, record["significance"]))

        # get the proper plot function and call it
        from dhi.plots.significances_root import plot_significance_scan

        plot_significance_scan(
            path=output.path,
            poi=self.poi,
            expected_values=expected_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotMultipleSignificanceScans(MultiDatacardTask, PlotSignificanceScan):

    @classmethod
    def modify_param_values(cls, params):
        params = MultiDatacardTask.modify_param_values(params)
        params = PlotSignificanceScan.modify_param_values(params)
        return params

    def requires(self):
        return [
            MergeSignificanceScan.req(self, datacards=datacards)
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []

        name = self.create_plot_name("multisignificances", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load significances
        expected_values = []
        names = []
        for i, inp in enumerate(self.input()):
            expected_values.append(inp.load(formatter="numpy")["data"])
            names.append("datacards {}".format(i + 1))

        # set names if requested
        if self.datacard_names:
            names = list(self.datacard_names)

        # reoder if requested
        if self.datacard_order:
            expected_values = [expected_values[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # get the proper plot function and call it
        from dhi.plots.significances_root import plot_significance_scans

        plot_significance_scans(
            path=output.path,
            poi=self.poi,
            expected_values=expected_values,
            names=names,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotSOverB(PlotTask, POITask1D):

    poi = PostFitShapes.poi
    bin_edges = law.CSVParameter(
        default=tuple(),
        significant=False,
        description="comma-separated list of bin edges to use; when empty, a automatic binning is "
        "applied; default: empty",
    )

    def requires(self):
        return PostFitShapes.req(self)

    def output(self):
        return self.local_target_dc("sb__{}.{}".format(self.get_poi_postfix(), self.file_type))

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # get the path to the fit diagnostics file
        fit_diagnostics_path = self.input().path

        # get the proper plot function and call it
        # (only the mpl version exists right now)
        from dhi.plots.misc_root import plot_s_over_b

        plot_s_over_b(
            path=output.path,
            fit_diagnostics_path=fit_diagnostics_path,
            poi=self.poi,
            bin_edges=self.bin_edges,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
