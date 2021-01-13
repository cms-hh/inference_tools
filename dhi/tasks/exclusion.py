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
                "likelihoods": MergeLikelihoodScan.req(self, pois=tuple(self.scan_parameter_names),
                    datacards=datacards, _prefer_cli=["scan_parameters"]),
            }
            for datacards in self.multi_datacards
        ]

    def output(self):
        name = self.create_plot_name(["exclusionbestfit", self.get_output_postfix()])
        return self.local_target(name)

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
            exp_limits = limit_data["data"]

            # load likelihoods
            ll_data = inp["likelihoods"].load(formatter="numpy")
            exp_nll = ll_data["data"]
            # insert a dnll2 column
            exp_nll = np.array(rec.append_fields(exp_nll, ["dnll2"],
                [exp_nll["delta_nll"] * 2.0]))
            # scan parameter mininum
            exp_scan_min = ll_data["poi_mins"][0]
            exp_scan_min = None if np.isnan(exp_scan_min) else float(exp_scan_min)

            # store data
            data.append({
                "name": "datacards {}".format(i + 1),
                "expected_limits": exp_limits,
                "expected_nll": exp_nll,
                "expected_scan_min": exp_scan_min,
            })

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func("dhi.plots.exclusion.plot_exclusion_and_bestfit_1d",
            path=output.path,
            data=data,
            poi=self.pois[0],
            scan_parameter=self.scan_parameter_names[0],
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotExclusionAndBestFit2D(POIScanTask, POIPlotTask):

    xsec_contours = law.CSVParameter(
        default=("auto",),
        significant=False,
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

    def __init__(self, *args, **kwargs):
        super(PlotExclusionAndBestFit2D, self).__init__(*args, **kwargs)

        if self.pois[0] not in self.r_pois and self.xsec_contours:
            self.logger.warning("cross section contours not supported for POI {}".format(
                self.pois[0]))
            self.xsec_contours = tuple()

    def requires(self):
        return {
            "limits": MergeUpperLimits.req(self),
            "likelihoods": MergeLikelihoodScan.req(self, pois=tuple(self.scan_parameter_names),
                _prefer_cli=["scan_parameters"]),
        }

    def output(self):
        name = self.create_plot_name(["exclusionbestfit2d", self.get_output_postfix()])
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit scan data
        inputs = self.input()
        exp_limits = inputs["limits"].load(formatter="numpy")["data"]

        # also compute limit values in a specified unit when requested
        xsec_values = None
        xsec_levels = None
        xsec_label_positions = None
        if self.pois[0] in self.r_pois and self.xsec_contours:
            xsec_values = self.get_theory_xsecs(self.pois[0], self.scan_parameter_names,
                self.get_scan_linspace(0.1), self.xsec, self.br,
                xsec_kwargs=self.parameter_values_dict)

            # configure contours
            if len(self.xsec_contours) == 1 and isinstance(self.xsec_contours[0], six.string_types):
                if self.xsec_contours[0] == "auto":
                    # automatic contours without labels
                    xsec_levels = "auto"
                else:
                    # interpret as path
                    path = os.path.expandvars(os.path.expanduser(self.xsec_contours[0]))
                    if not os.path.exists(path):
                        raise Exception("invalid cross section contour value '{}'".format(
                            self.xsec_contours[0]))

                    xsec_levels = []
                    xsec_label_positions = []
                    with open(path, "r") as f:
                        for line in f.readlines():
                            parts = [p.strip() for p in line.strip().split(",")]
                            if not parts or parts[0].startswith(("#", "//")):
                                continue

                            # get the level
                            xsec_levels.append(float(parts.pop(0)))

                            # get label positions, remaining length must be 0, 3, 6, ...
                            if len(parts) % 3 != 0:
                                raise Exception("invalid xsec contour definition '{}'".format(
                                    line))
                            positions = [
                                tuple(float(p) for p in parts[3 * i:3 * (i + 1)])
                                for i in range(len(parts) / 3)
                            ]
                            xsec_label_positions.append(positions)
            else:
                xsec_levels = self.xsec_contours or None

        # load likelihood scan data
        llh_data = inputs["likelihoods"].load(formatter="numpy")
        exp_llh_values = llh_data["data"]
        # scan parameter minima
        exp_scan_mins = [llh_data["poi_mins"][i] for i in range(self.n_scan_parameters)]
        exp_scan_mins = [(None if np.isnan(v) else float(v)) for v in exp_scan_mins]

        # call the plot function
        self.call_plot_func("dhi.plots.exclusion.plot_exclusion_and_bestfit_2d",
            path=output.path,
            poi=self.pois[0],
            scan_parameter1=self.scan_parameter_names[0],
            scan_parameter2=self.scan_parameter_names[1],
            expected_limits=exp_limits,
            observed_limits=None,
            xsec_values=xsec_values,
            xsec_levels=xsec_levels,
            xsec_unit=self.xsec,
            xsec_label_positions=xsec_label_positions,
            expected_likelihoods=exp_llh_values,
            expected_scan_minima=exp_scan_mins,
            observed_likelihoods=None,
            observed_scan_minima=None,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
