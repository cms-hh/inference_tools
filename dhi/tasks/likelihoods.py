# coding: utf-8

"""
Tasks related to likelihood scans.
"""

import copy

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    CombineCommandTask,
    MultiDatacardTask,
    MultiHHModelTask,
    POIScanTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.util import unique_recarray, extend_recarray, convert_dnll2


class LikelihoodBase(POIScanTask):

    pois = copy.copy(POIScanTask.pois)
    pois._default = ("kl",)

    force_scan_parameters_equal_pois = True


class LikelihoodScan(LikelihoodBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        linspace = self.get_scan_linspace()

        # when blinded and the expected best fit values of r and k POIs (== 1) are not contained
        # in the grid points, log an error as this leads to a scenario where the global minimum nll
        # is not computed and the deltas to all other nll values become arbitrary and incomparable
        if self.blinded:
            for i, poi in enumerate(self.pois):
                values = [tpl[i] for tpl in linspace]
                if poi in self.all_pois and 1 not in values:
                    scan = "start: {}, stop: {}, points: {}".format(*self.scan_parameters[i][1:])
                    self.logger.error("the expected best fit value of 1 is not contained in the "
                        "values to scan for POI {} ({}), leading to dnll values being computed "
                        "relative to an arbitrary minimum".format(poi, scan))

        return linspace

    def workflow_requires(self):
        reqs = super(LikelihoodScan, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("likelihood__" + self.get_output_postfix() + ".root")

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)
        else:
            return "--seed {self.branch} --toys {self.toys}".format(self=self)

    def build_command(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --algo grid"
            " --redefineSignalPOIs {self.joined_pois}"
            " --gridPoints {self.joined_scan_points}"
            " --firstPoint {self.branch}"
            " --lastPoint {self.branch}"
            " --alignEdges 1"
            " --setParameterRanges {self.joined_scan_ranges}:{self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --robustFit 1"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )


class MergeLikelihoodScan(LikelihoodBase):

    def requires(self):
        return LikelihoodScan.req(self)

    def output(self):
        return self.local_target("limits__" + self.get_output_postfix() + ".npz")

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        data = []
        dtype = [(p, np.float32) for p in self.scan_parameter_names] + [
            ("dnll", np.float32),
            ("dnll2", np.float32),
        ]
        poi_mins = self.n_pois * [np.nan]
        branch_map = self.requires().branch_map
        for branch, inp in self.input()["collection"].targets.items():
            if not inp.exists():
                self.logger.warning("input of branch {} at {} does not exist".format(
                    branch, inp.path))
                continue

            scan_values = branch_map[branch]
            f = inp.load(formatter="uproot")["limit"]
            dnll = f["deltaNLL"].array()
            failed = len(dnll) <= 1
            if failed:
                data.append(scan_values + (len(dtype) - self.n_pois) * (np.nan,))
                continue
            dnll = float(dnll[1])

            # save the best fit values
            if np.nan in poi_mins:
                poi_mins = [f[p].array()[0] for p in self.pois]

            # compute the dnll2 value
            dnll2 = dnll * 2.

            # store the value of that point
            data.append(scan_values + (dnll, dnll2))

        data = np.array(data, dtype=dtype)
        self.output().dump(data=data, poi_mins=np.array(poi_mins), formatter="numpy")


class PlotLikelihoodScan(LikelihoodBase, POIPlotTask):

    convert = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "significance", "pvalue"],
        description="convert dnll2 values to either a 'significance' or 'pvalue'; no default",
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; 1D only; default: False",
    )
    show_best_fit = luigi.BoolParameter(
        default=True,
        description="when False, do not draw the best fit value; default: True",
    )
    show_best_fit_error = luigi.BoolParameter(
        default=True,
        description="when False, the uncertainty bars of the POI's best fit values are not shown; "
        "default: True",
    )
    recompute_best_fit = luigi.BoolParameter(
        default=False,
        description="when True, do not use the best fit value as reported from combine but "
        "recompute it using scipy.minimize; default: False",
    )
    show_points = luigi.BoolParameter(
        default=False,
        significant=False,
        description="show points of central likelihood values; 1D only; default: False",
    )
    show_box = luigi.BoolParameter(
        default=False,
        significant=False,
        description="draw a box around the 1 sigma error contour and estimate a standard error "
        "from its dimensions; 2D only; default: False",
    )

    force_n_pois = (1, 2)
    force_n_scan_parameters = (1, 2)
    sort_pois = False
    sort_scan_parameters = False
    allow_multiple_scan_ranges = True

    def requires(self):
        return [
            MergeLikelihoodScan.req(self, scan_parameters=scan_parameters)
            for scan_parameters in self.get_scan_parameter_combinations()
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        prefix = "nll" if self.convert == law.NO_STR else self.convert
        names = self.create_plot_names(
            ["{}{}d".format(prefix, self.n_pois), self.get_output_postfix(), parts]
        )
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load scan data
        values, poi_mins = self.load_scan_data(self.input(), merge_scans=self.n_pois == 1)

        # use significance plots if requested
        if self.convert in ["significance", "pvalue"]:
            if self.n_pois == 1:
                sig = convert_dnll2(values["dnll2"], n=1)[1]
                values = extend_recarray(values, ("significance", float, sig))
                self.call_plot_func(
                    "dhi.plots.significances.plot_significance_scan_1d",
                    paths=[outp.path for outp in outputs],
                    scan_parameter=self.pois[0],
                    expected_values=None if self.unblinded else values,
                    observed_values=values if self.unblinded else None,
                    show_p_values=self.convert == "pvalue",
                    x_min=self.get_axis_limit("x_min"),
                    x_max=self.get_axis_limit("x_max"),
                    y_min=self.get_axis_limit("y_min"),
                    y_max=self.get_axis_limit("y_max"),
                    y_log=self.y_log,
                    model_parameters=self.get_shown_parameters(),
                    campaign=self.campaign if self.campaign != law.NO_STR else None,
                    show_points=self.show_points,
                    paper=self.paper,
                )
            else:  # 2
                values = [
                    extend_recarray(vals, ("significance", float,
                        convert_dnll2(vals["dnll2"], n=2)[1]))
                    for vals in values
                ]
                self.call_plot_func(
                    "dhi.plots.significances.plot_significance_scan_2d",
                    paths=[outp.path for outp in outputs],
                    scan_parameter1=self.pois[0],
                    scan_parameter2=self.pois[1],
                    values=values,
                    show_p_values=self.convert == "pvalue",
                    x_min=self.get_axis_limit("x_min"),
                    x_max=self.get_axis_limit("x_max"),
                    y_min=self.get_axis_limit("y_min"),
                    y_max=self.get_axis_limit("y_max"),
                    z_min=self.get_axis_limit("z_min"),
                    z_max=self.get_axis_limit("z_max"),
                    model_parameters=self.get_shown_parameters(),
                    campaign=self.campaign if self.campaign != law.NO_STR else None,
                    paper=self.paper,
                )
            return

        # call the plot function
        if self.n_pois == 1:
            # theory value when this is an r poi
            theory_value = None
            if self.pois[0] in self.r_pois:
                get_xsec = self.create_xsec_func(self.pois[0], "fb", safe_signature=True)
                if get_xsec.has_unc:
                    xsec = get_xsec(**self.parameter_values_dict)
                    xsec_up = get_xsec(unc="up", **self.parameter_values_dict)
                    xsec_down = get_xsec(unc="down", **self.parameter_values_dict)
                    theory_value = (xsec, xsec_up - xsec, xsec - xsec_down)
                else:
                    theory_value = (get_xsec(**self.parameter_values_dict),)
                # normalize
                theory_value = tuple(v / theory_value[0] for v in theory_value)

            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scan_1d",
                paths=[outp.path for outp in outputs],
                poi=self.pois[0],
                values=values,
                theory_value=theory_value,
                poi_min=None if self.recompute_best_fit else poi_mins[0],
                show_best_fit=self.show_best_fit,
                show_best_fit_error=self.show_best_fit_error,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                show_points=self.show_points,
                paper=self.paper,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scan_2d",
                paths=[outp.path for outp in outputs],
                poi1=self.pois[0],
                poi2=self.pois[1],
                values=values,
                poi1_min=None if self.recompute_best_fit else poi_mins[0],
                poi2_min=None if self.recompute_best_fit else poi_mins[1],
                show_best_fit=self.show_best_fit,
                show_best_fit_error=self.show_best_fit_error,
                show_box=self.show_box,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                z_min=self.get_axis_limit("z_min"),
                z_max=self.get_axis_limit("z_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                paper=self.paper,
            )

    def load_scan_data(self, inputs, merge_scans=True):
        return self._load_scan_data(inputs, self.scan_parameter_names,
            self.get_scan_parameter_combinations(), merge_scans=merge_scans)

    @classmethod
    def _load_scan_data(cls, inputs, scan_parameter_names, scan_parameter_combinations,
            merge_scans=True):
        import numpy as np

        # load values of each input
        values = []
        all_poi_mins = []
        for inp in inputs:
            data = inp.load(formatter="numpy")
            values.append(data["data"])
            all_poi_mins.append([
                (None if np.isnan(data["poi_mins"][i]) else float(data["poi_mins"][i]))
                for i in range(len(scan_parameter_names))
            ])

        # concatenate values and safely remove duplicates when configured
        if merge_scans:
            test_fn = lambda kept, removed: kept < 1e-7 or abs((kept - removed) / kept) < 0.001
            values = unique_recarray(values, cols=scan_parameter_names,
                test_metric=("dnll2", test_fn))

        # pick the most appropriate poi mins
        poi_mins = cls._select_poi_mins(all_poi_mins, scan_parameter_combinations)

        return values, poi_mins

    @classmethod
    def _select_poi_mins(cls, poi_mins, scan_parameter_combinations):
        # pick the poi mins for the scan range that has the lowest step size around the mins
        # the combined step size of multiple dims is simply defined by their sum
        min_step_size = 1e5
        best_poi_mins = poi_mins[0]
        for _poi_mins, scan_parameters in zip(poi_mins, scan_parameter_combinations):
            if None in _poi_mins:
                continue
            # each min is required to be in the corresponding scan range
            if not all((a <= v <= b) for v, (_, a, b, _) in zip(_poi_mins, scan_parameters)):
                continue
            # compute the merged step size
            step_size = sum((b - a) / (n - 1.) for (_, a, b, n) in scan_parameters)
            # store
            if step_size < min_step_size:
                min_step_size = step_size
                best_poi_mins = _poi_mins
        return best_poi_mins


class PlotMultipleLikelihoodScans(PlotLikelihoodScan, MultiDatacardTask):

    convert = law.NO_STR
    show_best_fit_error = None
    z_min = None
    z_max = None
    z_log = None

    @classmethod
    def modify_param_values(cls, params):
        params = PlotLikelihoodScan.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def requires(self):
        return [
            [
                MergeLikelihoodScan.req(self, datacards=datacards, scan_parameters=scan_parameters)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        names = self.create_plot_names(
            ["multinll{}d".format(self.n_pois), self.get_output_postfix(), parts]
        )
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load scan data
        data = []
        for i, inps in enumerate(self.input()):
            values, poi_mins = self.load_scan_data(inps)

            if self.recompute_best_fit:
                poi_mins = [None] * len(poi_mins)

            # store a data entry
            data.append(dict([
                ("values", values),
                ("poi_min", poi_mins[0]) if self.n_pois == 1 else ("poi_mins", poi_mins),
                ("name", "Cards {}".format(i + 1)),
            ]))

        # set names if requested
        if self.datacard_names:
            for d, name in zip(data, self.datacard_names):
                d["name"] = name

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # call the plot function
        if self.n_pois == 1:
            # theory value when this is an r poi
            theory_value = None
            if self.pois[0] in self.r_pois:
                get_xsec = self.create_xsec_func(self.pois[0], "fb", safe_signature=True)
                if get_xsec.has_unc:
                    xsec = get_xsec(**self.parameter_values_dict)
                    xsec_up = get_xsec(unc="up", **self.parameter_values_dict)
                    xsec_down = get_xsec(unc="down", **self.parameter_values_dict)
                    theory_value = (xsec, xsec_up - xsec, xsec - xsec_down)
                else:
                    theory_value = (get_xsec(**self.parameter_values_dict),)
                # normalize
                theory_value = tuple(v / theory_value[0] for v in theory_value)

            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_1d",
                paths=[outp.path for outp in outputs],
                poi=self.pois[0],
                data=data,
                theory_value=theory_value,
                show_best_fit=self.show_best_fit,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                show_points=self.show_points,
                paper=self.paper,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_2d",
                paths=[outp.path for outp in outputs],
                poi1=self.pois[0],
                poi2=self.pois[1],
                data=data,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                paper=self.paper,
            )


class PlotMultipleLikelihoodScansByModel(PlotLikelihoodScan, MultiHHModelTask):

    convert = law.NO_STR
    show_best_fit_error = None
    z_min = None
    z_max = None
    z_log = None

    def requires(self):
        return [
            [
                MergeLikelihoodScan.req(self, hh_model=hh_model, scan_parameters=scan_parameters)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]
            for hh_model in self.hh_models
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        names = self.create_plot_names(
            ["multinllbymodel{}d".format(self.n_pois), self.get_output_postfix(), parts]
        )
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load scan data
        data = []
        for hh_model, inps in zip(self.hh_models, self.input()):
            values, poi_mins = self.load_scan_data(inps)

            if self.recompute_best_fit:
                poi_mins = [None] * len(poi_mins)

            # prepare the name
            name = hh_model.rsplit(".", 1)[-1].replace("_", " ")
            if name.startswith("model "):
                name = name.split("model ", 1)[-1]

            # store a data entry
            data.append(dict([
                ("values", values),
                ("poi_min", poi_mins[0]) if self.n_pois == 1 else ("poi_mins", poi_mins),
                ("name", name),
            ]))

        # set names if requested
        if self.hh_model_names:
            for d, name in zip(data, self.hh_model_names):
                d["name"] = name

        # reoder if requested
        if self.hh_model_order:
            data = [data[i] for i in self.hh_model_order]

        # call the plot function
        if self.n_pois == 1:
            # theory value when this is an r poi
            theory_value = None
            if self.pois[0] in self.r_pois:
                hh_model = self.hh_models[0]
                get_xsec = self._create_xsec_func(hh_model, self.pois[0], "fb", safe_signature=True)
                if get_xsec.has_unc:
                    xsec = get_xsec(**self.parameter_values_dict)
                    xsec_up = get_xsec(unc="up", **self.parameter_values_dict)
                    xsec_down = get_xsec(unc="down", **self.parameter_values_dict)
                    theory_value = (xsec, xsec_up - xsec, xsec - xsec_down)
                else:
                    theory_value = (get_xsec(**self.parameter_values_dict),)
                # normalize
                theory_value = tuple(v / theory_value[0] for v in theory_value)

            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_1d",
                paths=[outp.path for outp in outputs],
                poi=self.pois[0],
                data=data,
                theory_value=theory_value,
                show_best_fit=self.show_best_fit,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                show_points=self.show_points,
                paper=self.paper,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_2d",
                paths=[outp.path for outp in outputs],
                poi1=self.pois[0],
                poi2=self.pois[1],
                data=data,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                paper=self.paper,
            )
