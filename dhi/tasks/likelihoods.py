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


class LikelihoodScan(POIScanTask, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    pois = copy.copy(POIScanTask.pois)
    pois._default = ("kl",)

    run_command_in_tmp = True
    force_scan_parameters_equal_pois = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(LikelihoodScan, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("likelihood__" + self.get_output_postfix() + ".root")

    def build_command(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -v 1"
            " -m {self.mass}"
            " -t -1"
            " --algo grid"
            " --expectSignal 1"
            " --redefineSignalPOIs {self.joined_pois}"
            " --gridPoints {self.joined_scan_points}"
            " --setParameterRanges {self.joined_scan_ranges}"
            " --firstPoint {self.branch}"
            " --lastPoint {self.branch}"
            " --alignEdges 1"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --robustFit 1"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )


class MergeLikelihoodScan(POIScanTask):

    force_scan_parameters_equal_pois = True

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
            ("delta_nll", np.float32),
            ("dnll2", np.float32),
        ]
        poi_mins = self.n_pois * [np.nan]
        branch_map = self.requires().branch_map
        for branch, inp in self.input()["collection"].targets.items():
            scan_values = branch_map[branch]

            f = inp.load(formatter="uproot")["limit"]
            failed = len(f["deltaNLL"].array()) <= 1
            if failed:
                data.append(scan_values + (np.nan, np.nan))
                continue

            # save the best fit values
            if np.nan in poi_mins:
                poi_mins = [f[p].array()[0] for p in self.pois]

            # store the value of that point
            dnll = f["deltaNLL"].array()[1]
            data.append(scan_values + (dnll, dnll * 2.0))

        data = np.array(data, dtype=dtype)
        self.output().dump(data=data, poi_mins=np.array(poi_mins), formatter="numpy")


class PlotLikelihoodScan(POIScanTask, POIPlotTask):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis when the plot is 1D; default: False",
    )

    force_n_pois = (1, 2)
    force_n_scan_parameters = (1, 2)
    force_scan_parameters_equal_pois = True

    def requires(self):
        return MergeLikelihoodScan.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        name = self.create_plot_name(
            ["nll{}d".format(self.n_pois), self.get_output_postfix(), parts]
        )
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load scan data
        data = self.input().load(formatter="numpy")
        exp_values = data["data"]
        # poi minima
        poi_mins = [
            (None if np.isnan(data["poi_mins"][i]) else float(data["poi_mins"][i]))
            for i in range(self.n_pois)
        ]

        # call the plot function
        if self.n_pois == 1:
            # theory value when this is an r poi
            theory_value = None
            if self.pois[0] in self.r_pois:
                get_xsec = self.create_xsec_func(self.pois[0], "fb", safe_signature=True)
                has_unc = self.pois[0] in ("r", "r_gghh") and self.load_hh_model()[1].doNNLOscaling
                if has_unc:
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
                path=output.path,
                poi=self.pois[0],
                expected_values=exp_values,
                theory_value=theory_value,
                poi_min=poi_mins[0],
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scan_2d",
                path=output.path,
                poi1=self.pois[0],
                poi2=self.pois[1],
                expected_values=exp_values,
                poi1_min=poi_mins[0],
                poi2_min=poi_mins[1],
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                z_min=self.get_axis_limit("z_min"),
                z_max=self.get_axis_limit("z_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )


class PlotMultipleLikelihoodScans(PlotLikelihoodScan, MultiDatacardTask):

    z_min = None
    z_max = None
    z_log = None

    @classmethod
    def modify_param_values(cls, params):
        params = PlotLikelihoodScan.modify_param_values(params)
        params = MultiDatacardTask.modify_param_values(params)
        return params

    def requires(self):
        return [
            MergeLikelihoodScan.req(self, datacards=datacards) for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        name = self.create_plot_name(
            ["multinll{}d".format(self.n_pois), self.get_output_postfix(), parts]
        )
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load scan data
        data = []
        for i, inp in enumerate(self.input()):
            _data = inp.load(formatter="numpy")
            # expected data
            exp_values = _data["data"]

            # poi minima
            poi_mins = [
                (None if np.isnan(_data["poi_mins"][j]) else float(_data["poi_mins"][j]))
                for j in range(self.n_pois)
            ]

            # store a data entry
            data.append(dict([
                ("expected_values", exp_values),
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
                has_unc = self.pois[0] in ("r", "r_gghh") and self.load_hh_model()[1].doNNLOscaling
                if has_unc:
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
                path=output.path,
                poi=self.pois[0],
                data=data,
                theory_value=theory_value,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_2d",
                path=output.path,
                poi1=self.pois[0],
                poi2=self.pois[1],
                data=data,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )


class PlotMultipleLikelihoodScansByModel(PlotLikelihoodScan, MultiHHModelTask):

    z_min = None
    z_max = None
    z_log = None

    def requires(self):
        return [MergeLikelihoodScan.req(self, hh_model=hh_model) for hh_model in self.hh_models]

    def output(self):
        # additional postfix
        parts = []
        if self.n_pois == 1 and self.y_log:
            parts.append("log")

        name = self.create_plot_name(
            ["multinllbymodel{}d".format(self.n_pois), self.get_output_postfix(), parts]
        )
        return self.local_target(name)

    @view_output_plots
    @law.decorator.safe_output
    @law.decorator.log
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load scan data
        data = []
        for hh_model, inp in zip(self.hh_models, self.input()):
            _data = inp.load(formatter="numpy")
            # expected data
            exp_values = _data["data"]

            # poi minima
            poi_mins = [
                (None if np.isnan(_data["poi_mins"][j]) else float(_data["poi_mins"][j]))
                for j in range(self.n_pois)
            ]

            # prepare the name
            name = hh_model.rsplit(".", 1)[-1].replace("_", " ")
            if name.startswith("model "):
                name = name.split("model ", 1)[-1]

            # store a data entry
            data.append(dict([
                ("expected_values", exp_values),
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
                has_unc = self.pois[0] in ("r", "r_gghh") and self._load_hh_model(hh_model)[1].doNNLOscaling
                if has_unc:
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
                path=output.path,
                poi=self.pois[0],
                data=data,
                theory_value=theory_value,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_log=self.y_log,
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )
        else:  # 2
            self.call_plot_func(
                "dhi.plots.likelihoods.plot_likelihood_scans_2d",
                path=output.path,
                poi1=self.pois[0],
                poi2=self.pois[1],
                data=data,
                x_min=self.get_axis_limit("x_min"),
                x_max=self.get_axis_limit("x_max"),
                y_min=self.get_axis_limit("y_min"),
                y_max=self.get_axis_limit("y_max"),
                model_parameters=self.get_shown_parameters(),
                campaign=self.campaign if self.campaign != law.NO_STR else None,
            )
