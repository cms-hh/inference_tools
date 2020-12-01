# coding: utf-8

"""
NLO Plotting tasks.
"""

import law
import luigi
import six

from dhi.tasks.base import AnalysisTask
from dhi.tasks.combine import (
    MultiDatacardTask, POITask1D, POIScanTask1D, POITask1DWithR, POIScanTask1DWithR,
    POIScanTask2D,
)
from dhi.tasks.inference import (
    UpperLimits, MergeUpperLimits, MergeLikelihoodScan1D, MergeLikelihoodScan2D,
    MergePullsAndImpacts, SignificanceScan, MergeSignificanceScan,
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

    def convert_r_to_xsec(self, expected_values, poi=None, xsec=None, br=None):
        import numpy as np

        # set defaults
        if poi is None:
            poi = self.poi
        if xsec is None and self.xsec != law.NO_STR:
            xsec = self.xsec
        if br is None and self.br != law.NO_STR:
            br = self.br

        # create the conversion function
        get_xsec = self.create_xsec_func(poi, xsec, br=br)

        # convert values and remember theory values
        expected_values = np.array(expected_values)
        theory_values = []
        limit_keys = ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]
        for point in expected_values:
            xsec = get_xsec(point[poi])
            theory_values.append(xsec)
            for key in limit_keys:
                point[key] *= xsec

        return expected_values, np.array(theory_values)

    @view_output_plots
    @law.decorator.safe_output
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
        if self.xsec in ["pb", "fb"] and self.poi in ["kl", "C2V"]:
            expected_values, theory_values = self.convert_r_to_xsec(expected_values)
            xsec_unit = self.xsec
            if self.br in br_hh:
                hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]
        else:
            theory_values = np.ones(expected_values.size, dtype=np.float32)

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
            _expected_values = inp.load(formatter="numpy")["data"]

            # rescale from limit on r to limit on xsec when requested, depending on the poi
            if self.xsec in ["pb", "fb"] and self.poi in ["kl", "C2V"]:
                _expected_values, _theory_values = self.convert_r_to_xsec(_expected_values)
                if i == 0:
                    theory_values = _theory_values
                    xsec_unit = self.xsec
                    if self.br in br_hh:
                        hh_process = r"HH $\rightarrow$ " + br_hh_names[self.br]

            expected_values.append(_expected_values)
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
            expected_values=expected_values,
            names=names,
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
    x_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the x-axis; default: False",
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
        if self.x_log:
            parts.append("log")

        name = self.create_plot_name("limitsatpoi", self.get_poi_postfix(), parts)
        return self.local_target_dc(name)

    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # prepare theory value and xsec converter
        theory_value = 1.0
        xsec_unit = None
        get_xsec = None
        if self.xsec in ["fb", "pb"] and self.poi in ["kl", "C2V"]:
            xsec_unit = self.xsec
            get_xsec = self.create_xsec_func(self.poi, xsec_unit)
            theory_value = get_xsec(theory_value)

        # load limit values
        data = []
        for i, inp in enumerate(self.input()):
            # get limits and convert to xsec when requested
            limits = MergeUpperLimits.load_limits(inp)
            limits = tuple(l * theory_value for l in limits)

            # add data entry
            data.append({
                "name": "datacards {}".format(i + 1),
                "expected": limits,
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
            theory_value=theory_value,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            x_log=self.x_log,
            xsec_unit=xsec_unit,
            pp_process={"r": "pp", "r_gghh": "gg", "r_qqhh": "qq"}[self.r_poi],
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
