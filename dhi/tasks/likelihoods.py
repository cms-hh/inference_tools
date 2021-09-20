# coding: utf-8

"""
Tasks related to likelihood scans.
"""

import copy
from operator import mul

import law
import luigi
import six

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    CombineCommandTask,
    MultiDatacardTask,
    MultiHHModelTask,
    POIScanTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.tasks.snapshot import Snapshot, SnapshotUser
from dhi.config import poi_data
from dhi.util import unique_recarray, extend_recarray, convert_dnll2


class LikelihoodBase(POIScanTask, SnapshotUser):

    pois = copy.copy(POIScanTask.pois)
    pois._default = ("kl",)

    force_scan_parameters_equal_pois = True
    allow_parameter_values_in_pois = True

    def get_output_postfix(self, join=True):
        parts = super(LikelihoodBase, self).get_output_postfix(join=False)

        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts


class LikelihoodScan(LikelihoodBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(LikelihoodScan, self).workflow_requires()
        reqs["workspace"] = CreateWorkspace.req(self)
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self)
        return reqs

    def requires(self):
        reqs = {"workspace": CreateWorkspace.req(self)}
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, branch=0)
        return reqs

    def output(self):
        name = self.join_postfix(["likelihood", self.get_output_postfix()]) + ".root"
        return self.local_target(name)

    def build_command(self):
        # get the workspace to use and define snapshot args
        if self.use_snapshot:
            workspace = self.input()["snapshot"].path
            snapshot_args = " --snapshotName MultiDimFit"
        else:
            workspace = self.input()["workspace"].path
            snapshot_args = ""

        # args for blinding / unblinding
        if self.unblinded:
            blinded_args = "--seed {self.branch}".format(self=self)
        else:
            blinded_args = "--seed {self.branch} --toys {self.toys}".format(self=self)

        # ensure that ranges of scanned parameters contain their SM values (plus one step)
        # in order for the likelihood scan to find the expected minimum and compute all deltaNLL
        # values with respect that minimum; otherwise, results of different scans cannot be stitched
        # together as they were potentially compared against different minima; this could be
        # simplified by https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/686 which
        # would only require to set "--setParameterRangesForGrid {self.joined_scan_ranges}"
        ext_ranges = []
        for p, ranges in self.scan_parameters_dict.items():
            # gather data
            start, stop, points = ranges[0]
            sm_value = poi_data.get(p, {}).get("sm_value", 1.)
            step_size = (float(stop - start) / (points - 1)) if points > 1 else 1
            assert step_size > 0
            # decrease the starting point until the sm value is fully contained
            while start >= sm_value:
                start -= step_size
                points += 1
            # increase the stopping point until the sm value is fully contained
            while stop <= sm_value:
                stop += step_size
                points += 1
            # store the extended range
            ext_ranges.append((start, stop, points))
        # compute the new n-D point space
        ext_space = self._get_scan_linspace(ext_ranges)
        # get the point index of the current branch
        ext_point = ext_space.index(self.branch_data)
        # recreate joined expressions for the combine command
        ext_joined_scan_points = ",".join(map(str, (p for _, _, p in ext_ranges)))
        ext_joined_scan_ranges = ":".join(
            "{}={},{}".format(name, start, stop)
            for name, (start, stop, _) in zip(self.scan_parameter_names, ext_ranges)
        )

        # build the command
        cmd = (
            "combine -M MultiDimFit {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " --algo grid"
            " --redefineSignalPOIs {self.joined_pois}"
            " --gridPoints {ext_joined_scan_points}"
            " --firstPoint {ext_point}"
            " --lastPoint {ext_point}"
            " --alignEdges 1"
            " --setParameterRanges {ext_joined_scan_ranges}:{self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {snapshot_args}"
            " --saveNLL"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=workspace,
            output=self.output().path,
            blinded_args=blinded_args,
            snapshot_args=snapshot_args,
            ext_joined_scan_points=ext_joined_scan_points,
            ext_joined_scan_ranges=ext_joined_scan_ranges,
            ext_point=ext_point,
        )

        return cmd


class MergeLikelihoodScan(LikelihoodBase):

    def requires(self):
        return LikelihoodScan.req(self)

    def output(self):
        name = self.join_postfix(["likelihoods", self.get_output_postfix()]) + ".npz"
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        data = []
        dtype = [(p, np.float32) for p in self.scan_parameter_names] + [
            # raw nll0, nll and deltaNLL values from combine
            ("nll0", np.float32),
            ("nll", np.float32),
            ("dnll", np.float32),
            # dnll times two
            ("dnll2", np.float32),
            # absolute nll value of the fit
            ("fit_nll", np.float32),
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

            # get the raw nll and nll0 values
            nll0 = float(f["nll0"].array()[1])
            nll = float(f["nll"].array()[1])

            # absolute nll value of the particular fit
            fit_nll = nll + dnll

            # store the value of that point
            data.append(scan_values + (nll0, nll, dnll, dnll2, fit_nll))

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
        significant=False,
        description="when False, do not draw the best fit value; default: True",
    )
    show_best_fit_error = luigi.BoolParameter(
        default=True,
        significant=False,
        description="when False, the uncertainty bars of the POI's best fit values are not shown; "
        "default: True",
    )
    recompute_best_fit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, do not use the best fit value as reported from combine but "
        "recompute it using scipy.interpolate and scipy.minimize; default: False",
    )
    shift_negative_values = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, dnll2 values are vertically shifted to move the minimum back to 0; "
        "default: False",
    )
    interpolate_nans = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, interpolate NaN values with information from neighboring fits "
        "instead of drawing white pixels; 2D only; default: False",
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
            # get the theory value when this is a poi
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
            elif self.pois[0] in poi_data:
                theory_value = (poi_data[self.pois[0]].sm_value,)

            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scan_1d",
                paths=[outp.path for outp in outputs],
                poi=self.pois[0],
                values=values,
                theory_value=theory_value,
                poi_min=None if self.recompute_best_fit else poi_mins[0],
                show_best_fit=self.show_best_fit,
                show_best_fit_error=self.show_best_fit_error,
                shift_negative_values=self.shift_negative_values,
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
                shift_negative_values=self.shift_negative_values,
                interpolate_nans=self.interpolate_nans,
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

    def load_scan_data(self, inputs, recompute_dnll2=True, merge_scans=True):
        return self._load_scan_data(inputs, self.scan_parameter_names,
            self.get_scan_parameter_combinations(), recompute_dnll2=recompute_dnll2,
            merge_scans=merge_scans)

    @classmethod
    def _load_scan_data(cls, inputs, scan_parameter_names, scan_parameter_combinations,
            recompute_dnll2=True, merge_scans=True):
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

        # nll0 values must be identical per input (otherwise there might be an issue with the model)
        for v in values:
            nll_unique = np.unique(v["nll0"])
            nll_unique = nll_unique[~np.isnan(nll_unique)]
            if len(nll_unique) != 1:
                raise Exception("found {} different nll0 values in scan data which indicates in "
                    "issue with the model: {}".format(len(nll_unique), nll_unique))

        # recompute dnll2 from the minimum nll and fit_nll
        if recompute_dnll2:
            # use the overall minimal nll as a common reference value when merging
            min_nll = min(np.nanmin(v["nll"]) for v in values)
            for v in values:
                _min_nll = min_nll if merge_scans else np.nanmin(v["nll"])
                v["dnll2"] = 2 * (v["fit_nll"] - _min_nll)

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
        # pick the poi min corrsponding to the largest spanned region, i.e., range / area / volume
        regions = [
            (i, six.moves.reduce(mul, [stop - start for _, start, stop, _ in scan_range]))
            for i, scan_range in enumerate(scan_parameter_combinations)
        ]
        best_i = sorted(regions, key=lambda pair: -pair[1])[0][0]
        best_poi_mins = poi_mins[best_i]

        # old algorithm:
        # # pick the poi mins for the scan range that has the lowest step size around the mins
        # # the combined step size of multiple dims is simply defined by their sum
        # min_step_size = 1e5
        # best_poi_mins = poi_mins[0]
        # for _poi_mins, scan_parameters in zip(poi_mins, scan_parameter_combinations):
        #     if None in _poi_mins:
        #         continue
        #     # each min is required to be in the corresponding scan range
        #     if not all((a <= v <= b) for v, (_, a, b, _) in zip(_poi_mins, scan_parameters)):
        #         continue
        #     # compute the merged step size
        #     step_size = sum((b - a) / (n - 1.) for (_, a, b, n) in scan_parameters)
        #     # store
        #     if step_size < min_step_size:
        #         min_step_size = step_size
        #         best_poi_mins = _poi_mins

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
                shift_negative_values=self.shift_negative_values,
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
                shift_negative_values=self.shift_negative_values,
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
                shift_negative_values=self.shift_negative_values,
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
                shift_negative_values=self.shift_negative_values,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
                paper=self.paper,
            )
