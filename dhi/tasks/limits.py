# coding: utf-8

"""
Tasks related to upper limits.
"""

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    MultiHHModelTask,
    CombineCommandTask,
    POIScanTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.util import unique_recarray
from dhi.config import br_hh


class UpperLimitsBase(POIScanTask):

    force_scan_parameters_unequal_pois = True


class UpperLimits(UpperLimitsBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(UpperLimits, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("limit__" + self.get_output_postfix() + ".root")

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)
        else:
            return "--seed {self.branch} --toys {self.toys} --run expected --noFitAsimov".format(
                self=self)

    def build_command(self):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_scan_values},{self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.AsymptoticLimits.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )

    @classmethod
    def load_limits(cls, target, unblinded=False):
        import numpy as np

        # load raw values
        limits = target.load(formatter="uproot")["limit"].array("limit")

        # convert to (nominal, err1_up, err1_down, err2_up, err2_down)
        if len(limits) == 0:
            # no values, fit failed completely
            values = (np.nan, np.nan, np.nan, np.nan, np.nan)
        elif len(limits) == 1:
            # only nominal value
            values = (limits[0], np.nan, np.nan, np.nan, np.nan)
        elif len(limits) == 3:
            # 1 sigma variations exist, but not 2 sigma
            values = (limits[1], limits[2], limits[0], np.nan, np.nan)
        else:
            # both 1 and 2 sigma variations exist
            values = (limits[2], limits[3], limits[1], limits[4], limits[0])

        # when unblinded, append the observed value
        if unblinded:
            # get the observed value
            obs = limits[5] if len(limits) == 6 else np.nan
            values += (obs,)

        return values


class MergeUpperLimits(UpperLimitsBase):

    def requires(self):
        return UpperLimits.req(self)

    def output(self):
        return self.local_target("limits__" + self.get_output_postfix() + ".npz")

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        records = []
        dtype = [(p, np.float32) for p in self.scan_parameter_names] + [
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        if self.unblinded:
            dtype.append(("observed", np.float32))

        scan_task = self.requires()
        for branch, inp in self.input()["collection"].targets.items():
            if not inp.exists():
                self.logger.warning("input of branch {} at {} does not exist".format(
                    branch, inp.path))
                continue

            scan_values = scan_task.branch_map[branch]
            limits = UpperLimits.load_limits(inp, unblinded=self.unblinded)
            records.append(scan_values + limits)

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class PlotUpperLimits(UpperLimitsBase, POIPlotTask):

    xsec = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "pb", "fb"],
        description="convert limits to cross sections in this unit; only supported for r POIs; "
        "choices: pb,fb; no default",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: {}; no default".format(",".join(br_hh.keys())),
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    show_points = luigi.BoolParameter(
        default=False,
        significant=False,
        description="show points of central limit values; default: False",
    )

    z_min = None
    z_max = None

    restrict_n_pois = 1
    restrict_n_scan_parameters = 1
    allow_multiple_scan_ranges = True

    def __init__(self, *args, **kwargs):
        super(PlotUpperLimits, self).__init__(*args, **kwargs)

        self.poi = self.pois[0]
        self.scan_parameter = self.scan_parameter_names[0]

        # scaling to xsec is only supported for r pois
        if self.xsec != law.NO_STR and self.poi not in self.r_pois:
            raise Exception("{!r}: xsec conversion is only supported for r POIs".format(self))

        # show a hint when xsec and br related nuisances can be frozen
        if self.xsec != law.NO_STR:
            if self.br != law.NO_STR:
                hint = "when calculating limits on 'XS x BR', nuisances related to both signal " \
                    "cross sections and branch ratios should be frozen (nuisance group " \
                    "'signal_norm_xsbr' in the combination)"
            else:
                hint = "when calculating limits on 'XS', nuisances related to signal cross " \
                    "sections should be frozen (nuisance group 'signal_norm_xs' in the combination)"
            self.logger.info("HINT: " + hint)

    def requires(self):
        return [
            MergeUpperLimits.req(self, scan_parameters=scan_parameters)
            for scan_parameters in self.get_scan_parameter_combinations()
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["limits", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        limit_values = self.load_scan_data(self.input())

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        thy_values = None
        xsec_unit = None
        if self.poi in self.r_pois:
            thy_linspace = np.linspace(limit_values[self.scan_parameter].min(),
                limit_values[self.scan_parameter].max(), num=100)
            if self.xsec in ["pb", "fb"]:
                limit_values = self.convert_to_xsecs(
                    self.poi,
                    limit_values,
                    self.xsec,
                    self.br,
                    param_keys=[self.scan_parameter],
                    xsec_kwargs=self.parameter_values_dict,
                )
                thy_values = self.get_theory_xsecs(
                    self.poi,
                    [self.scan_parameter],
                    thy_linspace,
                    self.xsec,
                    self.br,
                    xsec_kwargs=self.parameter_values_dict,
                )
                xsec_unit = self.xsec
            else:
                # normalized values
                thy_values = self.get_theory_xsecs(
                    self.poi,
                    [self.scan_parameter],
                    thy_linspace,
                    normalize=True,
                    xsec_kwargs=self.parameter_values_dict,
                )

        # some printing
        for v in range(-2, 4 + 1):
            if v in limit_values[self.scan_parameter]:
                record = limit_values[limit_values[self.scan_parameter] == v][0]
                self.publish_message("{} = {} -> {} {}".format(self.scan_parameter, v,
                    record["limit"], xsec_unit or "({})".format(self.poi)))

        # prepare observed values
        obs_values = None
        if self.unblinded:
            obs_values = {
                self.scan_parameter: limit_values[self.scan_parameter],
                "limit": limit_values["observed"],
            }

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scan",
            paths=[out.path for out in outputs],
            poi=self.poi,
            scan_parameter=self.scan_parameter,
            expected_values=limit_values,
            observed_values=obs_values,
            theory_values=thy_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            show_points=self.show_points,
        )

    def load_scan_data(self, inputs):
        return self._load_scan_data(inputs, self.scan_parameter_names)

    @classmethod
    def _load_scan_data(cls, inputs, scan_parameter_names):
        # load values of each input
        values = []
        for inp in inputs:
            data = inp.load(formatter="numpy")
            values.append(data["data"])

        # concatenate values and safely remove duplicates
        test_fn = lambda kept, removed: kept < 1e-7 or abs((kept - removed) / kept) < 0.001
        values = unique_recarray(values, cols=scan_parameter_names, test_metric=("limit", test_fn))

        return values


class PlotMultipleUpperLimits(PlotUpperLimits, MultiDatacardTask):

    unblinded = None

    @classmethod
    def modify_param_values(cls, params):
        params = PlotUpperLimits.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def requires(self):
        return [
            [
                MergeUpperLimits.req(self, datacards=datacards, scan_parameters=scan_parameters)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["multilimits", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        limit_values = []
        names = []
        thy_values = None
        xsec_unit = None
        for i, inps in enumerate(self.input()):
            _limit_values = self.load_scan_data(inps)

            # rescale from limit on r to limit on xsec when requested, depending on the poi
            if self.poi in self.r_pois:
                thy_linspace = np.linspace(_limit_values[self.scan_parameter].min(),
                    _limit_values[self.scan_parameter].max(), num=100)
                if self.xsec in ["pb", "fb"]:
                    _limit_values = self.convert_to_xsecs(
                        self.poi,
                        _limit_values,
                        self.xsec,
                        self.br,
                        param_keys=[self.scan_parameter],
                        xsec_kwargs=self.parameter_values_dict,
                    )
                    xsec_unit = self.xsec
                    if i == 0:
                        thy_values = self.get_theory_xsecs(
                            self.poi,
                            [self.scan_parameter],
                            thy_linspace,
                            self.xsec,
                            self.br,
                            xsec_kwargs=self.parameter_values_dict,
                        )
                elif i == 0:
                    # normalized values
                    thy_values = self.get_theory_xsecs(
                        self.poi,
                        [self.scan_parameter],
                        thy_linspace,
                        normalize=True,
                        xsec_kwargs=self.parameter_values_dict,
                    )

            limit_values.append(_limit_values)
            names.append("datacards {}".format(i + 1))

        # set names if requested
        if self.datacard_names:
            names = list(self.datacard_names)

        # reoder if requested
        if self.datacard_order:
            limit_values = [limit_values[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scans",
            paths=[out.path for out in outputs],
            poi=self.poi,
            scan_parameter=self.scan_parameter,
            names=names,
            expected_values=limit_values,
            theory_values=thy_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            show_points=self.show_points,
        )


class PlotMultipleUpperLimitsByModel(PlotUpperLimits, MultiHHModelTask):

    unblinded = None

    allow_empty_hh_model = True

    def requires(self):
        return [
            [
                MergeUpperLimits.req(self, hh_model=hh_model, scan_parameters=scan_parameters)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]
            for hh_model in self.hh_models
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        names = self.create_plot_names(["multilimitsbymodel", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        limit_values = []
        names = []
        thy_values = None
        xsec_unit = None
        for i, (hh_model, inps) in enumerate(zip(self.hh_models, self.input())):
            _limit_values = self.load_scan_data(inps)

            # rescale from limit on r to limit on xsec when requested, depending on the poi
            if self.poi in self.r_pois:
                thy_linspace = np.linspace(_limit_values[self.scan_parameter].min(),
                    _limit_values[self.scan_parameter].max(), num=100)
                if self.xsec in ["pb", "fb"]:
                    _limit_values = self._convert_to_xsecs(
                        hh_model,
                        self.poi,
                        _limit_values,
                        self.xsec,
                        self.br,
                        param_keys=[self.scan_parameter],
                        xsec_kwargs=self.parameter_values_dict,
                    )
                    xsec_unit = self.xsec
                    if i == 0:
                        thy_values = self._get_theory_xsecs(
                            hh_model,
                            self.poi,
                            [self.scan_parameter],
                            thy_linspace,
                            self.xsec,
                            self.br,
                            xsec_kwargs=self.parameter_values_dict,
                        )
                elif i == 0:
                    # normalized values at one with errors
                    thy_values = self._get_theory_xsecs(
                        hh_model,
                        self.poi,
                        [self.scan_parameter],
                        thy_linspace,
                        normalize=True,
                        xsec_kwargs=self.parameter_values_dict,
                    )

            # prepare the name
            name = hh_model.rsplit(".", 1)[-1].replace("_", " ")
            if name.startswith("model "):
                name = name.split("model ", 1)[-1]

            limit_values.append(_limit_values)
            names.append(name)

        # set names if requested
        if self.hh_model_names:
            names = list(self.hh_model_names)

        # reoder if requested
        if self.hh_model_order:
            limit_values = [limit_values[i] for i in self.hh_model_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scans",
            paths=[out.path for out in outputs],
            poi=self.poi,
            scan_parameter=self.scan_parameter,
            names=names,
            expected_values=limit_values,
            theory_values=thy_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            show_points=self.show_points,
        )


class PlotUpperLimitsAtPoint(POIPlotTask, MultiDatacardTask):

    xsec = PlotUpperLimits.xsec
    br = PlotUpperLimits.br
    x_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the x-axis; default: False",
    )
    sort_by = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=(law.NO_STR, "expected", "observed"),
        significant=False,
        description="either 'expected' or 'observed' for sorting entries from top to bottom in "
        "descending order; has precedence over --datacard-order when set; no default",
    )
    h_lines = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(),
        significant=False,
        description="comma-separated vertical positions of horizontal lines; no default",
    )
    left_margin = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="left margin of the pad in pixels; uses the default of the plot when empty; no "
        "default",
    )
    entry_height = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="vertical height of each entry in pixels; uses the default of the plot when "
        "empty; no default",
    )

    y_min = None
    y_max = None
    z_min = None
    z_max = None

    force_n_pois = 1

    def __init__(self, *args, **kwargs):
        super(PlotUpperLimitsAtPoint, self).__init__(*args, **kwargs)

        self.poi = self.pois[0]

        # this task depends on the UpperLimits task which does a scan over several parameters, but
        # we rather require a single point, so define a pseudo scan parameter for easier handling
        pois_with_values = [p for p in self.parameter_values_dict if p in self.all_pois]
        other_pois = [p for p in (self.k_pois + self.r_pois) if p != self.pois[0]]
        self.pseudo_scan_parameter = (pois_with_values + other_pois)[0]

        # show a hint when xsec and br related nuisances can be frozen
        if self.xsec != law.NO_STR:
            if self.br != law.NO_STR:
                hint = "when calculating limits on 'XS x BR', nuisances related to both signal " \
                    "cross sections and branch ratios should be frozen (nuisance group " \
                    "'signal_norm_xsbr' in the combination)"
            else:
                hint = "when calculating limits on 'XS', nuisances related to signal cross " \
                    "sections should be frozen (nuisance group 'signal_norm_xs' in the combination)"
            self.logger.info("HINT: " + hint)

    def requires(self):
        scan_parameter_value = self.parameter_values_dict.get(self.pseudo_scan_parameter, 1.0)
        scan_parameter = (self.pseudo_scan_parameter, scan_parameter_value, scan_parameter_value, 1)
        parameter_values = tuple(
            pv for pv in self.parameter_values
            if not pv.startswith(self.pseudo_scan_parameter + "=")
        )
        return [
            UpperLimits.req(
                self,
                scan_parameters=(scan_parameter,),
                parameter_values=parameter_values,
                datacards=datacards,
            )
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.x_log:
            parts.append("log")

        names = self.create_plot_names(["limitsatpoint", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit values
        names = ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2"]
        if self.unblinded:
            names .append("observed")
        limit_values = np.array(
            [
                UpperLimits.load_limits(coll["collection"][0], unblinded=self.unblinded)
                for coll in self.input()
            ],
            dtype=[(name, np.float32) for name in names],
        )

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        thy_value = None
        xsec_unit = None
        if self.poi in self.r_pois:
            if self.xsec in ["pb", "fb"]:
                limit_values = self.convert_to_xsecs(
                    self.poi, limit_values, self.xsec, self.br, xsec_kwargs=self.parameter_values_dict
                )
                thy_value = self.get_theory_xsecs(
                    self.poi,
                    [self.pseudo_scan_parameter],
                    [self.parameter_values_dict.get(self.pseudo_scan_parameter, 1.0)],
                    self.xsec,
                    self.br,
                    xsec_kwargs=self.parameter_values_dict,
                )
                xsec_unit = self.xsec
            else:
                # normalized values at one with errors
                thy_value = self.get_theory_xsecs(
                    self.poi,
                    [self.pseudo_scan_parameter],
                    [self.parameter_values_dict.get(self.pseudo_scan_parameter, 1.0)],
                    normalize=True,
                    xsec_kwargs=self.parameter_values_dict,
                )

        # fill data entries as expected by the plot function
        data = []
        for i, record in enumerate(limit_values):
            entry = {
                "name": "datacards {}".format(i + 1),
                "expected": record.tolist()[:5],
                "theory": thy_value and thy_value[0].tolist()[1:],
            }
            if self.unblinded:
                entry["observed"] = float(record[5])
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
            "dhi.plots.limits.plot_limit_points",
            paths=[out.path for out in outputs],
            poi=self.poi,
            data=data,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            sort_by=None if self.sort_by == law.NO_STR else self.sort_by,
            x_log=self.x_log,
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            model_parameters=self.get_shown_parameters(),
            h_lines=self.h_lines,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotUpperLimits2D(UpperLimitsBase, POIPlotTask):

    z_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the z-axis; default: False",
    )
    h_lines = law.CSVParameter(
        default=tuple(),
        significant=False,
        description="comma-separated values for drawing horizontal lines; no default",
    )
    v_lines = law.CSVParameter(
        default=tuple(),
        significant=False,
        description="comma-separated values for drawing vertical lines; no default",
    )

    force_n_pois = 1
    force_n_scan_parameters = 2
    sort_scan_parameters = False
    allow_multiple_scan_ranges = True

    def requires(self):
        return [
            MergeUpperLimits.req(self, scan_parameters=scan_parameters)
            for scan_parameters in self.get_scan_parameter_combinations()
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.z_log:
            parts.append("log")

        names = self.create_plot_names(["limits2d", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load limit scan data
        limits = []
        for inp in self.input():
            data = inp.load(formatter="numpy")
            limits.append(data["data"])

        # get observed limits when unblinded
        obs_limits = None
        if self.unblinded:
            obs_limits = [
                {
                    self.scan_parameter_names[0]: _limits[self.scan_parameter_names[0]],
                    self.scan_parameter_names[1]: _limits[self.scan_parameter_names[1]],
                    "limit": _limits["observed"],
                }
                for _limits in limits
            ]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scan_2d",
            paths=[out.path for out in outputs],
            poi=self.pois[0],
            scan_parameter1=self.scan_parameter_names[0],
            scan_parameter2=self.scan_parameter_names[1],
            expected_limits=limits,
            observed_limits=obs_limits,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            z_min=self.get_axis_limit("z_min"),
            z_max=self.get_axis_limit("z_max"),
            z_log=self.z_log,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            h_lines=self.h_lines,
            v_lines=self.v_lines,
        )
