# coding: utf-8

"""
Tasks related to upper limits.
"""

import os

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, BoxPlotTask, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    MultiHHModelTask,
    CombineCommandTask,
    POIScanTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.tasks.snapshot import Snapshot, SnapshotUser
from dhi.util import unique_recarray, real_path
from dhi.config import br_hh


class UpperLimitsBase(POIScanTask, SnapshotUser):

    force_scan_parameters_unequal_pois = True


class UpperLimits(UpperLimitsBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(UpperLimits, self).workflow_requires()
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
        parts = []
        if self.use_snapshot:
            parts.append("fromsnapshot")

        name = self.join_postfix(["limit", self.get_output_postfix(), parts]) + ".root"
        return self.local_target(name)

    def build_command(self):
        # get the workspace to use and define snapshot args
        if self.use_snapshot:
            workspace = self.input()["snapshot"].path
            snapshot_args = " --snapshotName MultiDimFit"
        else:
            workspace = self.input()["workspace"].path
            snapshot_args = ""

        # arguments for un/blinding
        if self.unblinded:
            blinded_args = "--seed {self.branch}".format(self=self)
        else:
            blinded_args = (
                " --seed {self.branch}"
                " --toys {self.toys}"
                " --run expected"
                " --noFitAsimov"
            ).format(self=self)

        # build the command
        cmd = (
            "combine -M AsymptoticLimits {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_scan_values},{self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {snapshot_args}"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.AsymptoticLimits.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=workspace,
            output=self.output().path,
            blinded_args=blinded_args,
            snapshot_args=snapshot_args,
        )

        return cmd

    @classmethod
    def load_limits(cls, target, unblinded=False):
        import numpy as np

        # load raw values
        data = target.load(formatter="uproot")["limit"].arrays(["limit", "quantileExpected"])
        limits = data["limit"]
        quantiles = data["quantileExpected"]

        # prepare limit values in the format (nominal, err1_up, err1_down, err2_up, err2_down)
        indices = {0.5: 0, 0.84: 1, 0.16: 2, 0.975: 3, 0.025: 4}
        values = [np.nan] * len(indices)
        for l, q in zip(limits, quantiles)[:len(indices)]:
            q = round(float(q), 3)
            if q in indices:
                values[indices[q]] = l

        # when unblinded, append the observed value
        if unblinded:
            values.append(limits[5] if len(limits) == 6 else np.nan)

        return tuple(values)


class MergeUpperLimits(UpperLimitsBase):

    def requires(self):
        return UpperLimits.req(self)

    def output(self):
        parts = []
        if self.use_snapshot:
            parts.append("fromsnapshot")

        name = self.join_postfix(["limits", self.get_output_postfix(), parts]) + ".npz"
        return self.local_target(name)

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
        if self.use_snapshot:
            parts.append("fromsnapshot")
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
                    skip_unc=True,
                    xsec_kwargs=self.parameter_values_dict,
                )

        # print some limits
        msg = self.poi
        if xsec_unit:
            br = "" if self.br in (None, law.NO_STR) else " x BR({})".format(self.br)
            msg = "cross section{} in {}, POI {}".format(br, xsec_unit, self.poi)
        self.publish_message("selected limits on {}".format(msg))
        for v in range(-3, 5 + 1):
            if v in limit_values[self.scan_parameter]:
                record = limit_values[limit_values[self.scan_parameter] == v][0]
                msg = "{} = {} -> {:.5f}".format(self.scan_parameter, v, record["limit"])
                if self.unblinded:
                    msg += " (obs. {:.5f})".format(record["observed"])
                self.publish_message(msg)

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
            paths=[outp.path for outp in outputs],
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
            paper=self.paper,
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
        if self.use_snapshot:
            parts.append("fromsnapshot")
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
                        skip_unc=True,
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
            paths=[outp.path for outp in outputs],
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
            paper=self.paper,
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
        if self.use_snapshot:
            parts.append("fromsnapshot")
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
                        skip_unc=True,
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
            names = [names[i] for i in self.hh_model_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scans",
            paths=[outp.path for outp in outputs],
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
            paper=self.paper,
        )


class PlotUpperLimitsAtPoint(POIPlotTask, SnapshotUser, MultiDatacardTask, BoxPlotTask):

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
        "descending order; has precedence over --datacard-order when set; default: empty",
    )
    h_lines = law.CSVParameter(
        cls=luigi.IntParameter,
        default=tuple(),
        significant=False,
        description="comma-separated vertical positions of horizontal lines; default: empty",
    )
    extra_labels = law.CSVParameter(
        default=tuple(),
        description="comma-separated labels to be shown per entry; default: empty"
    )
    external_limits = law.CSVParameter(
        default=tuple(),
        description="one or multiple json files that contain externally computed limit values to "
        "be shown below the ones computed with actual datacards; default: empty",
    )

    y_min = None
    y_max = None
    z_min = None
    z_max = None

    force_n_pois = 1

    def __init__(self, *args, **kwargs):
        # cached external limit values
        self._external_limit_values = None

        super(PlotUpperLimitsAtPoint, self).__init__(*args, **kwargs)

        # shorthand to the poi
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

        # check the length of extra labels
        n = self.n_datacard_entries
        if self.extra_labels and len(self.extra_labels) != n:
            raise Exception("found {} entries in extra_labels whereas {} is expected".format(
                len(self.extra_labels), n))

    @property
    def n_datacard_entries(self):
        n = len(self.multi_datacards)

        # add external limits when set
        external_limits = self.read_external_limits()
        if external_limits:
            n += len(external_limits)

        return n

    def read_external_limits(self):
        if self._external_limit_values is None and self.external_limits:
            external_limits = []

            for path in self.external_limits:
                # check the file
                path = real_path(path)
                if not os.path.exists(path):
                    raise Exception("external limit file '{}' does not exist".format(path))

                # read it and store values
                content = law.LocalFileTarget(path).load(formatter="json")
                limits = content["limits"]

                # optionally filter with "use" list
                if "use" in content:
                    limits = [l for l in limits if l["name"] in content["use"]]

                external_limits.extend(limits)

            self._external_limit_values = external_limits

        return self._external_limit_values

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
        if self.use_snapshot:
            parts.append("fromsnapshot")
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br != law.NO_STR:
                parts.append(self.br)
        if self.x_log:
            parts.append("log")
        if self.external_limits:
            parts.append("ext" + law.util.create_hash(self.external_limits))

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
            names.append("observed")
        limit_values = np.array(
            [
                UpperLimits.load_limits(coll["collection"][0], unblinded=self.unblinded)
                for coll in self.input()
            ],
            dtype=[(name, np.float32) for name in names],
        )

        # append external values when given
        external_limits = self.read_external_limits()
        if external_limits:
            ext = np.array(
                [tuple(l[name] for name in names) for l in external_limits],
                dtype=limit_values.dtype,
            )
            limit_values = np.concatenate([limit_values, ext], axis=0)

        # rescale from limit on r to limit on xsec when requested, depending on the poi
        thy_value = None
        xsec_unit = None
        if self.poi in self.r_pois:
            if self.xsec in ["pb", "fb"]:
                limit_values = self.convert_to_xsecs(
                    self.poi,
                    limit_values,
                    self.xsec,
                    self.br,
                    xsec_kwargs=self.parameter_values_dict,
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
                # normalized values
                thy_value = self.get_theory_xsecs(
                    self.poi,
                    [self.pseudo_scan_parameter],
                    [self.parameter_values_dict.get(self.pseudo_scan_parameter, 1.0)],
                    normalize=True,
                    skip_unc=True,
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

        # set extra labels is set
        if self.extra_labels:
            for d, label in zip(data, self.extra_labels):
                d["label"] = label

        # reoder if requested
        if self.datacard_order:
            data = [data[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_points",
            paths=[outp.path for outp in outputs],
            poi=self.poi,
            data=data,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            sort_by=None if self.sort_by == law.NO_STR else self.sort_by,
            x_log=self.x_log,
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            pad_width=None if self.pad_width == law.NO_INT else self.pad_width,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            right_margin=None if self.right_margin == law.NO_INT else self.right_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            model_parameters=self.get_shown_parameters(),
            h_lines=self.h_lines,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            paper=self.paper,
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
        if self.use_snapshot:
            parts.append("fromsnapshot")
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
            paths=[outp.path for outp in outputs],
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
            paper=self.paper,
        )
