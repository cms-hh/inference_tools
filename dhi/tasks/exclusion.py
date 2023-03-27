# coding: utf-8

"""
Tasks for obtaining exclusion plots.
"""

import copy

import law
import luigi

from dhi.tasks.base import BoxPlotTask, view_output_plots
from dhi.tasks.combine import MultiDatacardTask, POIScanTask, POIMultiTask, POIPlotTask
from dhi.tasks.snapshot import SnapshotUser
from dhi.tasks.limits import MergeUpperLimits, PlotUpperLimits
from dhi.tasks.likelihoods import MergeLikelihoodScan, PlotLikelihoodScan
from dhi.config import br_hh


class PlotExclusionAndBestFit(
    POIScanTask,
    POIMultiTask,
    MultiDatacardTask,
    POIPlotTask,
    SnapshotUser,
    BoxPlotTask,
):

    show_best_fit = PlotLikelihoodScan.show_best_fit
    show_best_fit_error = PlotLikelihoodScan.show_best_fit_error
    recompute_best_fit = PlotLikelihoodScan.recompute_best_fit
    h_lines = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(),
        significant=False,
        description="comma-separated vertical positions of horizontal lines; no default",
    )

    y_min = None
    y_max = None
    z_min = None
    z_max = None
    save_hep_data = False

    force_n_pois = 1
    force_n_scan_parameters = 1
    force_scan_parameters_unequal_pois = True
    allow_multiple_scan_ranges = True
    compare_multi_sequence = "multi_datacards"

    default_plot_function = "dhi.plots.exclusion.plot_exclusion_and_bestfit_1d"

    def requires(self):
        def merge_tasks(cls, **kwargs):
            return [
                cls.req(self, scan_parameters=scan_parameters, **kwargs)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]

        reqs = []
        for datacards, kwargs in zip(self.multi_datacards, self.get_multi_task_kwargs()):
            req = {"limits": merge_tasks(MergeUpperLimits, datacards=datacards, **kwargs)}
            if self.show_best_fit:
                req["likelihoods"] = merge_tasks(
                    MergeLikelihoodScan,
                    datacards=datacards,
                    pois=tuple(self.scan_parameter_names),
                    **kwargs  # noqa
                )
            reqs.append(req)

        return reqs

    def get_output_postfix(self, join=True):
        parts = super(PlotExclusionAndBestFit, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts

    def output(self):
        outputs = {}

        names = self.create_plot_names(["exclusionbestfit", self.get_output_postfix()])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()])
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load input data
        data = []
        inputs = self.input()
        for i, (inp, unblinded) in enumerate(zip(inputs, self.unblinded * len(inputs))):
            # load limits
            limits = PlotUpperLimits._load_scan_data(inp["limits"], self.scan_parameter_names)

            # load likelihoods
            nll_values, scan_min = None, None
            if "likelihoods" in inp:
                nll_values, _scan_min = PlotLikelihoodScan._load_scan_data(
                    inp["likelihoods"],
                    self.scan_parameter_names,
                    self.get_scan_parameter_combinations(),
                )
                _scan_min = _scan_min[self.scan_parameter_names[0]]
                if not self.recompute_best_fit and not np.isnan(_scan_min):
                    scan_min = float(_scan_min)

            # store data
            entry = dict(
                name="datacards {}".format(i + 1),
                expected_limits=limits,
                nll_values=nll_values,
                scan_min=scan_min,
            )
            if unblinded:
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
            paths=[outp.path for outp in outputs["plots"]],
            data=data,
            poi=self.pois[0],
            scan_parameter=self.scan_parameter_names[0],
            show_best_fit_error=self.show_best_fit_error,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            pad_width=None if self.pad_width == law.NO_INT else self.pad_width,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            right_margin=None if self.right_margin == law.NO_INT else self.right_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            model_parameters=self.get_shown_parameters(),
            h_lines=self.h_lines,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )


class PlotExclusionAndBestFit2D(POIScanTask, POIPlotTask, SnapshotUser):

    show_best_fit = PlotLikelihoodScan.show_best_fit
    show_best_fit_error = copy.copy(PlotLikelihoodScan.show_best_fit_error)
    show_best_fit_error._default = False
    recompute_best_fit = PlotLikelihoodScan.recompute_best_fit
    interpolation_method = PlotLikelihoodScan.interpolation_method
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
        "section when --xsec is set; choices: '',{}; no default".format(",".join(br_hh.keys())),
    )

    z_min = None
    z_max = None
    save_hep_data = False

    force_n_pois = 1
    force_n_scan_parameters = 2
    force_scan_parameters_unequal_pois = True
    sort_scan_parameters = False
    allow_multiple_scan_ranges = True

    default_plot_function = "dhi.plots.exclusion.plot_exclusion_and_bestfit_2d"

    def __init__(self, *args, **kwargs):
        super(PlotExclusionAndBestFit2D, self).__init__(*args, **kwargs)

        if self.pois[0] not in self.r_pois and self.xsec_contours:
            self.logger.warning(
                "cross section contours not supported for POI {}".format(self.pois[0]),
            )
            self.xsec_contours = tuple()

    def requires(self):
        def merge_tasks(cls, **kwargs):
            return [
                cls.req(self, scan_parameters=scan_parameters, **kwargs)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]

        reqs = {"limits": merge_tasks(MergeUpperLimits)}
        if self.show_best_fit:
            reqs["likelihoods"] = merge_tasks(
                MergeLikelihoodScan,
                pois=tuple(self.scan_parameter_names),
            )

        return reqs

    def get_output_postfix(self, join=True):
        parts = super(PlotExclusionAndBestFit2D, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts

    def output(self):
        outputs = {}

        names = self.create_plot_names(["exclusionbestfit2d", self.get_output_postfix()])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()])
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load limit scan data
        inputs = self.input()
        limits = PlotUpperLimits._load_scan_data(inputs["limits"], self.scan_parameter_names)

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
        if self.pois[0] in self.r_pois and self.xsec_contours:
            # obtain xsec values for the visible range
            def visible_border(i, j, axis_param):
                comp = min if axis_param.endswith("_min") else max
                axis_limit = self.get_axis_limit(axis_param)
                ranges = self.scan_parameters_dict[self.scan_parameter_names[i]]
                border = comp([r[j] for r in ranges])
                return border if axis_limit is None else comp(axis_limit, border)

            linspace_parameters = [
                (visible_border(0, 0, "x_min"), visible_border(0, 1, "x_max")),
                (visible_border(1, 0, "y_min"), visible_border(1, 1, "y_max")),
            ]
            xsec_values = self.get_theory_xsecs(
                self.pois[0],
                self.scan_parameter_names,
                self._get_scan_linspace(linspace_parameters, step_sizes=0.1),
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
        nll_values, scan_min1, scan_min2 = None, None, None
        if "likelihoods" in inputs:
            nll_values, scan_mins = PlotLikelihoodScan._load_scan_data(
                inputs["likelihoods"],
                self.scan_parameter_names,
                self.get_scan_parameter_combinations(),
            )
            nan_to_none = lambda v: None if np.isnan(float(v)) else float(v)
            scan_min1 = nan_to_none(scan_mins[self.scan_parameter_names[0]])
            scan_min2 = nan_to_none(scan_mins[self.scan_parameter_names[1]])

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
            poi=self.pois[0],
            scan_parameter1=self.scan_parameter_names[0],
            scan_parameter2=self.scan_parameter_names[1],
            expected_limits=limits,
            observed_limits=obs_limits,
            xsec_values=xsec_values,
            xsec_levels=xsec_levels,
            xsec_unit=self.xsec,
            nll_values=nll_values,
            interpolation_method=self.interpolation_method,
            show_best_fit_error=self.show_best_fit_error,
            scan_min1=None if self.recompute_best_fit else scan_min1,
            scan_min2=None if self.recompute_best_fit else scan_min2,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )
