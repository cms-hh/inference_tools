# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

import os
import re
from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, PlotTask, view_output_plots
from dhi.tasks.combine import (
    DatacardTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits


class EFTBase(DatacardTask):

    datacard_pattern = luigi.Parameter(
        default="datacard_(.+).txt",
        description="regular expression for extracting benchmark names from basenames of datacards "
        "with a single group; when set on the command line, single quotes should be used; "
        "default: datacard_(.+).txt",
    )
    frozen_parameters = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of parameters to be frozen in addition to non-POI and "
        "scan parameters",
    )
    frozen_groups = law.CSVParameter(
        default=(),
        unique=True,
        sort=True,
        description="comma-separated names of groups of parameters to be frozen",
    )
    unblinded = luigi.BoolParameter(
        default=False,
        description="unblinded computation and plotting of results; default: False",
    )

    hh_model = law.NO_STR
    allow_empty_hh_model = True

    @classmethod
    def modify_param_values(cls, params):
        params = DatacardTask.modify_param_values.__func__.__get__(cls)(params)

        # sort frozen parameters
        if "frozen_parameters" in params:
            params["frozen_parameters"] = tuple(sorted(params["frozen_parameters"]))

        # sort frozen groups
        if "frozen_groups" in params:
            params["frozen_groups"] = tuple(sorted(params["frozen_groups"]))

        return params

    def __init__(self, *args, **kwargs):
        super(EFTBase, self).__init__(*args, **kwargs)

        # create a map of datacard names (e.g. benchmark number or EFT parameters) to datacard paths
        self.eft_datacards = OrderedDict()
        for datacard in sorted(self.datacards):
            m = re.match(self.datacard_pattern, os.path.basename(datacard))
            if not m:
                raise ValueError("datacard {} does not match pattern '{}'".format(
                    datacard, self.datacard_pattern))
            name = m.group(1)
            if name in self.eft_datacards:
                raise ValueError("EFT datacard name '{}' is not unique".format(name))
            self.eft_datacards[name] = datacard

    def get_output_postfix(self, join=True):
        parts = []

        # add the unblinded flag
        if self.unblinded:
            parts.append(["unblinded"])

        # add frozen paramaters
        if self.frozen_parameters:
            parts.append(["fzp"] + list(self.frozen_parameters))

        # add frozen groups
        if self.frozen_groups:
            parts.append(["fzg"] + list(self.frozen_groups))

        return self.join_postfix(parts) if join else parts

    @property
    def joined_frozen_parameters(self):
        return ",".join(self.frozen_parameters) or '""'

    @property
    def joined_frozen_groups(self):
        return ",".join(self.frozen_groups) or '""'


class EFTBenchmarkBase(EFTBase):

    def __init__(self, *args, **kwargs):
        super(EFTBenchmarkBase, self).__init__(*args, **kwargs)

        # sort EFT datacards according to benchmark names
        names = self.sort_benchmark_names(self.eft_datacards.keys())
        self.benchmarks = OrderedDict((name, self.eft_datacards[name]) for name in names)

    @classmethod
    def sort_benchmark_names(cls, names):
        """
        Example order: 1, 2, 3, 3a, 3b, 4, 5, a_string, other_string, z_string
        """
        names = law.util.make_list(names)

        # split into names being a number or starting with one, and pure strings
        # store numeric names as tuples as sorted() will do exactly what we want
        num_names, str_names = [], []
        for name in names:
            m = re.match(r"^(\d+)(.*)$", name)
            if m:
                num_names.append((int(m.group(1)), m.group(2)))
            else:
                str_names.append(name)

        # sort and add
        num_names.sort()
        str_names.sort()
        return ["{}{}".format(*pair) for pair in num_names] + str_names


class EFTScanBase(EFTBase):

    def __init__(self, *args, **kwargs):
        super(EFTScanBase, self).__init__(*args, **kwargs)

        # get the name of the parameter to scan
        scan_parameters = set(map(self.extract_scan_parameter, self.eft_datacards.keys()))
        if len(scan_parameters) != 1:
            raise Exception("datacards belong to more than one EFT scan parameter: {}".format(
                ",".join(scan_parameters)))
        self.scan_parameter = list(scan_parameters)[0]

        # sort EFT datacards according to scan parameter values
        values = self.sort_scan_names(self.scan_parameter, self.eft_datacards.keys())
        self.scan_datacards = OrderedDict((v, self.eft_datacards[name]) for name, v in values)

    @classmethod
    def extract_scan_parameter(cls, name):
        """
        c2_1p5 -> c2
        """
        if "_" not in name:
            raise ValueError("invalid datacard name '{}'".format(name))
        return name.split("_", 1)[0]

    @classmethod
    def sort_scan_names(cls, scan_parameter, names):
        """
        Names have the format "<scan_parameters>_<number>"" where number might have "-" replaced by
        "m" and "." replaced by "d", so revert this and simply sort by number.
        """
        names = law.util.make_list(names)

        # extract the scan values
        values = []
        for name in names:
            if not name.startswith(scan_parameter + "_"):
                raise ValueError("invalid datacard name '{}'".format(name))
            v = name[len(scan_parameter) + 1:].replace("d", ".").replace("m", "-")
            try:
                v = float(v)
            except ValueError:
                raise ValueError("invalid scan value '{}' in datacard name '{}'".format(v, name))
            values.append((name, v))

        # sort by value
        values.sort(key=lambda tpl: tpl[1])

        return values

    def get_output_postfix(self, join=True):
        parts = [self.scan_parameter]

        parts += super(EFTScanBase, self).get_output_postfix(join=False)

        return self.join_postfix(parts) if join else parts


class EFTLimitBase(CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def workflow_requires(self):
        reqs = super(EFTLimitBase, self).workflow_requires()
        if not self.pilot:
            # require the requirements of all branch tasks when not in pilot mode
            reqs["workspace"] = {b: t.requires() for b, t in self.get_branch_tasks().items()}
        return reqs

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


class EFTBenchmarkLimits(EFTBenchmarkBase, EFTLimitBase):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.benchmarks.keys())

    def requires(self):
        return CreateWorkspace.req(self, datacards=(self.benchmarks[self.branch_data],),
            hh_model=law.NO_STR)

    def output(self):
        parts = self.get_output_postfix(join=False)
        parts.append("bm{}".format(self.branch_data))

        return self.local_target("limit__{}.root".format(self.join_postfix(parts)))


class EFTUpperLimits(EFTScanBase, EFTLimitBase):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.scan_datacards.keys())

    def requires(self):
        return CreateWorkspace.req(self, datacards=(self.scan_datacards[self.branch_data],),
            hh_model=law.NO_STR)

    def output(self):
        parts = self.get_output_postfix(join=False)
        parts[0] = "{}_{}".format(self.scan_parameter, self.branch_data)

        return self.local_target("limit__{}.root".format(self.join_postfix(parts)))


class MergeEFTBenchmarkLimits(EFTBenchmarkBase):

    def requires(self):
        return EFTBenchmarkLimits.req(self)

    def output(self):
        parts = self.get_output_postfix(join=False)
        postfix = ("__" + self.join_postfix(parts)) if parts else ""

        return self.local_target("limits{}.npz".format(postfix))

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        records = []
        dtype = [
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        if self.unblinded:
            dtype.append(("observed", np.float32))

        for branch, inp in self.input()["collection"].targets.items():
            limits = UpperLimits.load_limits(inp, unblinded=self.unblinded)
            records.append(limits)

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class MergeEFTUpperLimits(EFTScanBase):

    def requires(self):
        return EFTUpperLimits.req(self)

    def output(self):
        return self.local_target("limits__{}.npz".format(self.get_output_postfix()))

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        records = []
        dtype = [
            (self.scan_parameter, np.float32),
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
            limits = UpperLimits.load_limits(inp, unblinded=self.unblinded)
            records.append((scan_task.branch_map[branch],) + limits)

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")


class PlotEFTBenchmarkLimits(EFTBenchmarkBase, PlotTask):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    x_min = None
    x_max = None
    z_min = None
    z_max = None

    def requires(self):
        return MergeEFTBenchmarkLimits.req(self)

    def output(self):
        parts = self.get_output_postfix(join=False)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name(["limits"] + parts)
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values
        limit_values = self.input().load(formatter="numpy")["data"]

        # fill data entries as expected by the plot function
        data = []
        for name, record in zip(self.benchmarks, limit_values):
            entry = {
                "name": name,
                "expected": record.tolist()[:5],
            }
            if self.unblinded:
                entry["observed"] = float(record[5])
            data.append(entry)

            # some printing
            self.publish_message("BM {} -> {}".format(name, record["limit"]))

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_benchmark_limits",
            path=output.path,
            data=data,
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit="fb",
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotEFTUpperLimits(EFTScanBase, PlotTask):

    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    z_min = None
    z_max = None

    def requires(self):
        return MergeEFTUpperLimits.req(self)

    def output(self):
        parts = self.get_output_postfix(join=False)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name(["limits"] + parts)
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        from dhi.eft_tools import get_ggf_xsec

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values
        limit_values = self.input().load(formatter="numpy")["data"]

        # get theory values
        thy_values = {
            self.scan_parameter: limit_values[self.scan_parameter],
            "xsec": [
                get_ggf_xsec(**{self.scan_parameter: v}) * 0.001
                for v in limit_values[self.scan_parameter]
            ],
        }

        # prepare observed values
        obs_values = None
        if self.unblinded:
            obs_values = {
                self.scan_parameter: limit_values[self.scan_parameter],
                "limit": limit_values["observed"],
            }

        # fixed model parameters
        model_parameters = OrderedDict([(("kl", "kt"), 1.), (("cg", "c2g"), 1.)])

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_limit_scan",
            path=output.path,
            poi="r_gghh",
            scan_parameter=self.scan_parameter,
            expected_values=limit_values,
            observed_values=obs_values,
            theory_values=thy_values,
            x_min=self.get_axis_limit("x_min"),
            x_max=self.get_axis_limit("x_max"),
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit="fb",
            model_parameters=model_parameters,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
