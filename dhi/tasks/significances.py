# coding: utf-8

"""
Tasks related to significance calculation.
"""

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    CombineCommandTask,
    POIScanTask,
    POIMultiTask,
    POIPlotTask,
    CreateWorkspace,
)
from dhi.tasks.snapshot import Snapshot, SnapshotUser
from dhi.util import unique_recarray


class SignificanceBase(POIScanTask, SnapshotUser):

    force_scan_parameters_unequal_pois = True
    allow_parameter_values_in_pois = True

    frequentist_toys = luigi.BoolParameter(
        default=False,
        description="when set, create frequentist postfit toys, producing a-posteriori expected "
        "significances, which depend on observed data; has no effect when --unblinded is used as "
        "well; default: False",
    )

    def __init__(self, *args, **kwargs):
        super(SignificanceBase, self).__init__(*args, **kwargs)

        if self.unblinded and self.frequentist_toys:
            self.frequentist_toys = False
            self.logger.warning("both --unblinded and --frequentist_toys were set, will only "
                "consider the unblinded flag")

    def get_output_postfix(self, join=True):
        parts = super(SignificanceBase, self).get_output_postfix(join=False)

        if not self.unblinded and self.frequentist_toys:
            parts.insert(0, ["postfit"])
        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts


class SignificanceScan(SignificanceBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(SignificanceScan, self).workflow_requires()
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self)
        else:
            reqs["workspace"] = CreateWorkspace.req(self)
        return reqs

    def requires(self):
        reqs = {}
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, branch=0)
        else:
            reqs["workspace"] = CreateWorkspace.req(self, branch=0)
        return reqs

    def output(self):
        name = self.join_postfix(["significance", self.get_output_postfix()]) + ".root"
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
        elif self.frequentist_toys:
            blinded_args = "--seed {self.branch} --toys {self.toys} --toysFreq".format(self=self)
        else:
            blinded_args = "--seed {self.branch} --toys {self.toys}".format(self=self)

        # build the command
        cmd = (
            "combine -M Significance {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " {snapshot_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_scan_values},{self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.Significance.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=workspace,
            output=self.output().path,
            blinded_args=blinded_args,
            snapshot_args=snapshot_args,
        )

        return cmd


class MergeSignificanceScan(SignificanceBase):

    def requires(self):
        return SignificanceScan.req(self)

    def output(self):
        name = self.join_postfix(["significance", self.get_output_postfix()]) + ".npz"
        return self.local_target(name)

    @law.decorator.log
    def run(self):
        import numpy as np
        import scipy.stats

        records = []
        dtype = [(p, np.float32) for p in self.scan_parameter_names] + [
            ("significance", np.float32),
            ("p_value", np.float32),
        ]
        scan_task = self.requires()
        for branch, inp in self.input()["collection"].targets.items():
            if not inp.exists():
                self.logger.warning(
                    "input of branch {} at {} does not exist".format(branch, inp.path),
                )
                continue

            scan_values = scan_task.branch_map[branch]
            sig = inp.load(formatter="uproot")["limit"].array("limit")
            if sig:
                sig = sig[0]
                pval = scipy.stats.norm.sf(sig)
            else:
                self.logger.warning(
                    "significance calculation failed for scan values {}".format(scan_values),
                )
                sig, pval = np.nan, np.nan
            records.append(scan_values + (sig, pval))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class PlotSignificanceScan(SignificanceBase, POIPlotTask):

    convert = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "pvalue"],
        description="convert dnll2 values to 'pvalue'; no default",
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )
    show_points = luigi.BoolParameter(
        default=False,
        significant=False,
        description="show points of central significance values; default: False",
    )

    z_min = None
    z_max = None
    save_hep_data = None

    force_n_scan_parameters = 1
    sort_pois = False
    allow_multiple_scan_ranges = True

    default_plot_function = "dhi.plots.significances.plot_significance_scan_1d"

    def requires(self):
        def merge_tasks(**kwargs):
            return [
                MergeSignificanceScan.req(self, scan_parameters=scan_parameters, **kwargs)
                for scan_parameters in self.get_scan_parameter_combinations()
            ]

        reqs = {}
        if self.unblinded:
            reqs["expected"] = merge_tasks(unblinded=False)
            reqs["observed"] = merge_tasks(unblinded=True, frequentist_toys=False)
        else:
            reqs["expected"] = merge_tasks(unblinded=False)
        return reqs

    def output(self):
        # additional postfix
        parts = []
        if self.y_log:
            parts.append("log")

        outputs = {}

        prefix = "significance" if self.convert == law.NO_STR else self.convert
        names = self.create_plot_names([prefix, self.get_output_postfix(), parts])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load significances
        inputs = self.input()
        scan_parameter = self.scan_parameter_names[0]
        exp_values = self.load_scan_data(inputs["expected"])
        obs_values = self.load_scan_data(inputs["observed"]) if self.unblinded else None

        # some printing
        for v in range(-2, 4 + 1):
            if v in exp_values[scan_parameter]:
                record = exp_values[exp_values[scan_parameter] == v][0]
                self.publish_message(
                    "{} = {} -> {:.4f} sigma".format(scan_parameter, v, record["significance"]),
                )

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
            scan_parameter=scan_parameter,
            expected_values=exp_values,
            observed_values=obs_values,
            show_p_values=self.convert == "pvalue",
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            show_points=self.show_points,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
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
        values = unique_recarray(
            values,
            cols=scan_parameter_names,
            test_metric=("significance", test_fn),
        )

        return values


class PlotMultipleSignificanceScans(PlotSignificanceScan, POIMultiTask, MultiDatacardTask):

    unblinded = None

    compare_multi_sequence = "multi_datacards"

    default_plot_function = "dhi.plots.significances.plot_significance_scans_1d"

    @classmethod
    def modify_param_values(cls, params):
        params = PlotSignificanceScan.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def requires(self):
        return [
            [
                MergeSignificanceScan.req(
                    self,
                    datacards=datacards,
                    scan_parameters=scan_parameters,
                    **kwargs  # noqa
                )
                for scan_parameters in self.get_scan_parameter_combinations()
            ]
            for datacards, kwargs in zip(self.multi_datacards, self.get_multi_task_kwargs())
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.y_log:
            parts.append("log")

        outputs = {}

        prefix = "significance" if self.convert == law.NO_STR else self.convert
        names = self.create_plot_names(["multi{}s".format(prefix), self.get_output_postfix(), parts])
        outputs["plots"] = [self.local_target(name) for name in names]

        # plot data
        if self.save_plot_data:
            name = self.join_postfix(["plotdata", self.get_output_postfix()] + parts)
            outputs["plot_data"] = self.local_target("{}.pkl".format(name))

        return outputs

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs["plots"][0].parent.touch()

        # load significances
        values = []
        names = []
        for i, inps in enumerate(self.input()):
            values.append(self.load_scan_data(inps))
            names.append("datacards {}".format(i + 1))

        # set names if requested
        if self.datacard_names:
            names = list(self.datacard_names)

        # reoder if requested
        if self.datacard_order:
            values = [values[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            paths=[outp.path for outp in outputs["plots"]],
            scan_parameter=self.scan_parameter_names[0],
            values=values,
            names=names,
            show_p_values=self.convert == "pvalue",
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            show_points=self.show_points,
            cms_postfix=self.cms_postfix,
            style=self.style,
            dump_target=outputs.get("plot_data"),
        )
