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
    POIPlotTask,
    CreateWorkspace,
)
from dhi.util import unique_recarray


class SignificanceBase(POIScanTask):

    force_scan_parameters_unequal_pois = True
    allow_parameter_values_in_pois = True

    postfit_toys = luigi.BoolParameter(
        default=False,
        description="when set, create frequentist postfit toys, producing a-posteriori expected "
        "significances, which depend on observed data; has no effect when --unblinded is used as "
        "well; default: False",
    )

    def __init__(self, *args, **kwargs):
        super(SignificanceBase, self).__init__(*args, **kwargs)

        if self.unblinded and self.postfit_toys:
            self.postfit_toys = False
            self.logger.warning("both unblinded and postfit_toys were set, will only consider "
                "the unblinded flag")

    def get_output_postfix(self, join=True):
        parts = super(SignificanceBase, self).get_output_postfix(join=False)

        if not self.unblinded and self.postfit_toys:
            parts.insert(0, ["postfit"])

        return self.join_postfix(parts) if join else parts


class SignificanceScan(SignificanceBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return self.get_scan_linspace()

    def workflow_requires(self):
        reqs = super(SignificanceScan, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("significance__" + self.get_output_postfix() + ".root")

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)
        elif self.postfit_toys:
            return "--seed {self.branch} --toys {self.toys} --toysFreq".format(self=self)
        else:
            return "--seed {self.branch} --toys {self.toys}".format(self=self)

    def build_command(self):
        return (
            "combine -M Significance {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameters {self.joined_scan_values},{self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.Significance.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )


class MergeSignificanceScan(SignificanceBase):
    def requires(self):
        return SignificanceScan.req(self)

    def output(self):
        return self.local_target("significance__{}.npz".format(self.get_output_postfix()))

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
            scan_values = scan_task.branch_map[branch]
            sig = inp.load(formatter="uproot")["limit"].array("limit")[0]
            pval = scipy.stats.norm.sf(sig)
            records.append(scan_values + (sig, pval))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class PlotSignificanceScan(SignificanceBase, POIPlotTask):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    z_min = None
    z_max = None

    force_n_scan_parameters = 1
    sort_pois = False
    allow_multiple_scan_ranges = True

    def requires(self):
        def merge_tasks(**kwargs):
            return [
                MergeSignificanceScan.req(self, scan_parameters=scan_parameters, **kwargs)
                for scan_parameters in self.get_scan_parameters_product()
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

        name = self.create_plot_name(["significances", self.get_output_postfix(), parts])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

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
                    "{} = {} -> {:.4f} sigma".format(scan_parameter, v, record["significance"])
                )

        # call the plot function
        self.call_plot_func(
            "dhi.plots.significances.plot_significance_scan",
            path=output.path,
            poi=self.pois[0],
            scan_parameter=scan_parameter,
            expected_values=exp_values,
            observed_values=obs_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )

    def load_scan_data(self, inputs):
        return self._load_scan_data(inputs, self.scan_parameter_names)

    @classmethod
    def _load_scan_data(cls, inputs, scan_parameter_names):
        import numpy as np

        # load values of each input
        all_values = []
        for inp in inputs:
            data = inp.load(formatter="numpy")
            all_values.append(data["data"])

        # concatenate values and safely remove duplicates
        values = np.concatenate(all_values, axis=0)
        test_fn = lambda kept, removed: kept < 1e-7 or abs((kept - removed) / kept) < 0.001
        values = unique_recarray(values, cols=scan_parameter_names,
            test_metric=("significance", test_fn))

        return values


class PlotMultipleSignificanceScans(PlotSignificanceScan, MultiDatacardTask):

    unblinded = None

    @classmethod
    def modify_param_values(cls, params):
        params = PlotSignificanceScan.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def requires(self):
        return [
            [
                MergeSignificanceScan.req(self, datacards=datacards, scan_parameters=scan_parameters)
                for scan_parameters in self.get_scan_parameters_product()
            ]
            for datacards in self.multi_datacards
        ]

    def output(self):
        # additional postfix
        parts = []
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name(["multisignificances", self.get_output_postfix(), parts])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load significances
        exp_values = []
        names = []
        for i, inps in enumerate(self.input()):
            exp_values.append(self.load_scan_data(inps))
            names.append("datacards {}".format(i + 1))

        # set names if requested
        if self.datacard_names:
            names = list(self.datacard_names)

        # reoder if requested
        if self.datacard_order:
            exp_values = [exp_values[i] for i in self.datacard_order]
            names = [names[i] for i in self.datacard_order]

        # call the plot function
        self.call_plot_func(
            "dhi.plots.significances.plot_significance_scans",
            path=output.path,
            poi=self.pois[0],
            scan_parameter=self.scan_parameter_names[0],
            expected_values=exp_values,
            names=names,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            model_parameters=self.get_shown_parameters(),
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
