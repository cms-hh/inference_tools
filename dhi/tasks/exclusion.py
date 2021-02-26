# coding: utf-8

"""
Tasks for obtaining exclusion plots.
"""

import os

import law
import luigi
import six

from dhi.tasks.base import view_output_plots
from dhi.tasks.combine import MultiDatacardTask, POIScanTask, POIPlotTask
from dhi.tasks.limits import MergeUpperLimits
from dhi.tasks.likelihoods import MergeLikelihoodScan
from dhi.config import br_hh


class PlotExclusionAndBestFit(POIScanTask, MultiDatacardTask, POIPlotTask):

    h_lines = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(),
        significant=False,
        description="comma-separated vertical positions of horizontal lines; default: empty",
    )

    y_min = None
    y_max = None
    z_min = None
    z_max = None

    force_n_pois = 1
    force_n_scan_parameters = 1
    force_scan_parameters_unequal_pois = True

    def requires(self):
        return [
            {
                "limits": MergeUpperLimits.req(self, datacards=datacards),
                "likelihoods": MergeLikelihoodScan.req(
                    self,
                    pois=tuple(self.scan_parameter_names),
                    datacards=datacards,
                    _prefer_cli=["scan_parameters"],
                ),
            }
            for datacards in self.multi_datacards
        ]

    def output(self):
        name = self.create_plot_name(["exclusionbestfit", self.get_output_postfix()])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load input data
        data = []
        for i, inp in enumerate(self.input()):
            # load limits
            limit_data = inp["limits"].load(formatter="numpy")
            limits = limit_data["data"]

            # load likelihoods
            ll_data = inp["likelihoods"].load(formatter="numpy")
            nll_values = ll_data["data"]
            # scan parameter mininum
            scan_min = ll_data["poi_mins"][0]
            scan_min = None if np.isnan(scan_min) else float(scan_min)

            # store data
            entry = dict(
                name="datacards {}".format(i + 1),
                expected_limits=limits,
                nll_values=nll_values,
                scan_min=scan_min,
            )
            if self.unblinded:
                entry["observed_limits"] = {
                    self.scan_parameter_names[0]: limits[self.scan_parameter_names[0]],
                    "limit": limits["observed"],
                }
            data.append(entry)

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.exclusion.plot_exclusion_and_bestfit_1d",
            path=output.path,
            data=data,
            poi=self.pois[0],
            scan_parameter=self.scan_parameter_names[0],
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            model_parameters=self.get_shown_parameters(),
            h_lines=self.h_lines,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotExclusionAndBestFit2D(POIScanTask, POIPlotTask):

    xsec_contours = law.CSVParameter(
        default=("auto",),
        significant=False,
        unique=True,
        description="draw cross section contours at these values; only supported for r POIs; the "
        "unit is defined by --xsec; when 'auto', default contours are drawn without labels; can "
        "also be a file with contours defined line-by-line in the format "
        "'xsec_in_fb[,label_x,label_y,label_rotation[,...]]'; label positions are interpreted in "
        "scan parameter units; default: auto",
    )
    xsec = luigi.ChoiceParameter(
        default="pb",
        choices=["pb", "fb"],
        significant=False,
        description="compute cross section values in this unit for the contours configured with "
        "--xsec-contours;"
        "choices: pb,fb; default: pb",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when --xsec is set; choices: '',{}; default: empty".format(",".join(br_hh.keys())),
    )
    z_min = None
    z_max = None

    force_n_pois = 1
    force_n_scan_parameters = 2
    force_scan_parameters_unequal_pois = True
    sort_scan_parameters = False

    def __init__(self, *args, **kwargs):
        super(PlotExclusionAndBestFit2D, self).__init__(*args, **kwargs)

        if self.pois[0] not in self.r_pois and self.xsec_contours:
            self.logger.warning(
                "cross section contours not supported for POI {}".format(self.pois[0])
            )
            self.xsec_contours = tuple()

    def requires(self):
        return {
            "limits": MergeUpperLimits.req(self),
            "likelihoods": MergeLikelihoodScan.req(
                self, pois=tuple(self.scan_parameter_names), _prefer_cli=["scan_parameters"]
            ),
        }

    def output(self):
        name = self.create_plot_name(["exclusionbestfit2d", self.get_output_postfix()])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit scan data
        inputs = self.input()
        limits = inputs["limits"].load(formatter="numpy")["data"]

        # prepare observed limits
        obs_limits = None
        if self.unblinded:
            obs_limits = {
                self.scan_parameter_names[0]: limits[self.scan_parameter_names[0]],
                self.scan_parameter_names[1]: limits[self.scan_parameter_names[1]],
                "limit": limits["observed"],
            }

        # also compute limit values in a specified unit when requested
        xsec_values = None
        xsec_levels = None
        xsec_label_positions = None
        if self.pois[0] in self.r_pois and self.xsec_contours:
            # obtain xsec values for the visible range
            def visible_border(i, j, axis_param):
                axis_limt = self.get_axis_limit(axis_param)
                if axis_limt is None:
                    return self.scan_parameters[i][j]
                else:
                    comp = min if axis_param.endswith("_min") else max
                    return comp(axis_limt, self.scan_parameters[i][j])
            linspace_parameters = [
                (None, visible_border(0, 1, "x_min"), visible_border(0, 2, "x_max"), None),
                (None, visible_border(1, 1, "y_min"), visible_border(1, 2, "y_max"), None),
            ]
            xsec_values = self.get_theory_xsecs(
                self.pois[0],
                self.scan_parameter_names,
                self._get_scan_linspace(linspace_parameters, step_size=0.1),
                self.xsec,
                self.br,
                xsec_kwargs=self.parameter_values_dict,
            )

            # configure contours
            if self.xsec_contours[0] and self.xsec_contours[0] == "auto":
                # automatic contours without labels
                xsec_levels = "auto"
            else:
                xsec_levels = [float(l) for l in self.xsec_contours]

        # load likelihood scan data
        ll_data = inputs["likelihoods"].load(formatter="numpy")
        nll_values = ll_data["data"]
        # scan parameter minima
        scan_mins = [ll_data["poi_mins"][i] for i in range(self.n_scan_parameters)]
        scan_mins = [(None if np.isnan(v) else float(v)) for v in scan_mins]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.exclusion.plot_exclusion_and_bestfit_2d",
            path=output.path,
            poi=self.pois[0],
            scan_parameter1=self.scan_parameter_names[0],
            scan_parameter2=self.scan_parameter_names[1],
            expected_limits=limits,
            observed_limits=obs_limits,
            xsec_values=xsec_values,
            xsec_levels=xsec_levels,
            xsec_unit=self.xsec,
            nll_values=nll_values,
            scan_minima=scan_mins,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
