# coding: utf-8

"""
Tasks related to EFT benchmarks and scans.
"""

import os
import re
from collections import OrderedDict, defaultdict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, PlotTask, view_output_plots
from dhi.tasks.combine import (
    MultiDatacardTask,
    CombineCommandTask,
    CreateWorkspace,
)
from dhi.tasks.limits import UpperLimits
from dhi.eft_tools import (
    get_eft_ggf_xsec_nnlo, sort_eft_benchmark_names, sort_eft_scan_names,
    extract_eft_scan_parameter,
)
from dhi.config import br_hh


class EFTBase(MultiDatacardTask):

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

    exclude_params_index = {"datacard_names", "datacard_order"}

    hh_model = law.NO_STR
    datacard_names = None
    datacard_order = None
    allow_empty_hh_model = True

    poi = "r_gghh"

    @classmethod
    def modify_param_values(cls, params):
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)

        # re-group multi datacards by basenames
        if "multi_datacards" in params:
            groups = defaultdict(set)
            for datacards in params["multi_datacards"]:
                for datacard in datacards:
                    groups[os.path.basename(datacard)].add(datacard)
            params["multi_datacards"] = tuple(sorted(
                tuple(sorted(datacards))
                for datacards in groups.values()
            ))

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
        self.eft_datacards = {}
        for datacards in self.multi_datacards:
            # check if all basenames are identical
            basenames = set(map(os.path.basename, datacards))
            if len(basenames) != 1:
                raise Exception("found multiple basenames {} in datacards:\n  {}".format(
                    ",".join(basenames), "\n  ".join(datacards)))

            # extract the name
            basename = list(basenames)[0]
            m = re.match(self.datacard_pattern, basename)
            if not m:
                raise ValueError("datacard basename {} does not match pattern '{}'".format(
                    basename, self.datacard_pattern))
            self.eft_datacards[m.group(1)] = datacards

    def store_parts(self):
        parts = super(EFTBase, self).store_parts()
        parts["poi"] = "poi_{}".format(self.poi)
        return parts

    def get_output_postfix(self, join=True):
        parts = []

        # add the unblinded flag
        if self.unblinded:
            parts.append(["unblinded"])

        # add the poi
        parts.append(["poi", self.poi])

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
        names = sort_eft_benchmark_names(self.eft_datacards.keys())
        self.benchmark_datacards = OrderedDict((name, self.eft_datacards[name]) for name in names)


class EFTScanBase(EFTBase):

    scan_range = law.CSVParameter(
        default=(-100., 100.),
        cls=luigi.FloatParameter,
        min_len=2,
        max_len=2,
        sort=True,
        description="the range of the scan parameter extracted from the datacards in the format "
        "'min,max'; when empty, the full range is used; no default",
    )

    def __init__(self, *args, **kwargs):
        super(EFTScanBase, self).__init__(*args, **kwargs)

        # get the name of the parameter to scan
        scan_parameters = set(map(extract_eft_scan_parameter, self.eft_datacards.keys()))
        if len(scan_parameters) != 1:
            raise Exception("datacards belong to more than one EFT scan parameter: {}".format(
                ",".join(scan_parameters)))
        self.scan_parameter = list(scan_parameters)[0]

        # sort EFT datacards according to scan parameter values
        values = sort_eft_scan_names(self.scan_parameter, self.eft_datacards.keys())

        # apply the requested scan range
        scan_min = max(self.scan_range[0], min(v for _, v in values))
        scan_max = min(self.scan_range[1], max(v for _, v in values))
        self.scan_range = (scan_min, scan_max)

        # store a mapping of scan value to datacards
        self.scan_datacards = OrderedDict(
            (v, self.eft_datacards[name])
            for name, v in values
            if scan_min <= v <= scan_max
        )

    def get_output_postfix(self, join=True):
        parts = super(EFTScanBase, self).get_output_postfix(join=False)

        # insert the scan parameter value when this is a workflow branch, and the range otherwise
        if isinstance(self, law.BaseWorkflow) and self.is_branch():
            scan_part = [self.scan_parameter, self.branch_data]
        else:
            scan_part = ["scan", self.scan_parameter] + list(self.scan_range)
        parts.insert(2 if self.unblinded else 1, scan_part)

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

    def htcondor_output_postfix(self):
        return "_{}__{}".format(self.get_branches_repr(), self.get_output_postfix())


class EFTBenchmarkLimits(EFTBenchmarkBase, EFTLimitBase):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.benchmark_datacards.keys())

    def requires(self):
        return CreateWorkspace.req(self, datacards=self.benchmark_datacards[self.branch_data],
            hh_model=law.NO_STR)

    def output(self):
        parts = self.get_output_postfix(join=False)
        parts.append("bm{}".format(self.branch_data))

        return self.local_target("eftlimit__{}.root".format(self.join_postfix(parts)))


class EFTUpperLimits(EFTScanBase, EFTLimitBase):

    run_command_in_tmp = True

    def create_branch_map(self):
        return list(self.scan_datacards.keys())

    def requires(self):
        return CreateWorkspace.req(self, datacards=self.scan_datacards[self.branch_data],
            hh_model=law.NO_STR)

    def output(self):
        return self.local_target("eftlimit__{}.root".format(self.get_output_postfix()))


class MergeEFTBenchmarkLimits(EFTBenchmarkBase):

    def requires(self):
        return EFTBenchmarkLimits.req(self)

    def output(self):
        parts = self.get_output_postfix(join=False)
        postfix = ("__" + self.join_postfix(parts)) if parts else ""

        return self.local_target("eftlimits{}.npz".format(postfix))

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
        return self.local_target("eftlimits__{}.npz".format(self.get_output_postfix()))

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

    xsec = luigi.ChoiceParameter(
        default="fb",
        choices=["pb", "fb"],
        description="convert limits to cross sections in this unit; choices: pb,fb; default: fb",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: '',{}; no default".format(",".join(br_hh.keys())),
    )
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
        # additional postfix
        parts = []
        parts.append(self.xsec)
        if self.br and self.br != law.NO_STR:
            parts.append(self.br)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name(["eftbenchmarks", self.get_output_postfix(), parts])
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

        # prepare conversion scale
        scale = br_hh.get(self.br, 1.) * {"fb": 1., "pb": 0.001}[self.xsec]

        # fill data entries as expected by the plot function
        data = []
        for name, record in zip(self.benchmark_datacards.keys(), limit_values):
            entry = {
                "name": name,
                "expected": [v * scale for v in record.tolist()[:5]],
            }
            if self.unblinded:
                entry["observed"] = float(record[5]) * scale
            data.append(entry)

            # some printing
            self.publish_message("BM {} -> {}".format(name, record["limit"]))

        # call the plot function
        self.call_plot_func(
            "dhi.plots.limits.plot_benchmark_limits",
            path=output.path,
            data=data,
            poi="r_gghh",
            y_min=self.get_axis_limit("y_min"),
            y_max=self.get_axis_limit("y_max"),
            y_log=self.y_log,
            xsec_unit=self.xsec,
            hh_process=self.br if self.br in br_hh else None,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )


class PlotEFTUpperLimits(EFTScanBase, PlotTask):

    xsec = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, "", "pb", "fb"],
        description="convert limits to cross sections in this unit; choices: '',pb,fb; no default",
    )
    br = luigi.ChoiceParameter(
        default=law.NO_STR,
        choices=[law.NO_STR, ""] + list(br_hh.keys()),
        description="name of a branching ratio defined in dhi.config.br_hh to scale the cross "
        "section when xsec is used; choices: '',{}; no default".format(",".join(br_hh.keys())),
    )
    y_log = luigi.BoolParameter(
        default=False,
        description="apply log scaling to the y-axis; default: False",
    )

    z_min = None
    z_max = None

    def requires(self):
        return MergeEFTUpperLimits.req(self)

    def output(self):
        # additional postfix
        parts = []
        if self.xsec in ["pb", "fb"]:
            parts.append(self.xsec)
            if self.br not in (law.NO_STR, ""):
                parts.append(self.br)
        if self.y_log:
            parts.append("log")

        name = self.create_plot_name(["eftlimits", self.get_output_postfix(), parts])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        import numpy as np

        # prepare the output
        output = self.output()
        output.parent.touch()

        # load limit values, given in fb
        limit_values = self.input().load(formatter="numpy")["data"]

        # get nnlo cross sections in pb
        xsecs = OrderedDict(
            (v, get_eft_ggf_xsec_nnlo(**{self.scan_parameter: v}))
            for v in limit_values[self.scan_parameter]
        )

        # prepare the br value
        br_value = br_hh.get(self.br, 1.)

        # get scaling factors for xsec and limit values
        # apply scaling by xsec and br
        n = len(xsecs)
        if self.xsec == "fb":
            xsec_unit = "fb"
            limit_scales = n * [br_value]
            xsec_scales = n * [1000. * br_value]
        elif self.xsec == "pb":
            xsec_unit = "pb"
            limit_scales = n * [0.001 * br_value]
            xsec_scales = n * [br_value]
        else:
            xsec_unit = None
            limit_scales = (np.array(xsecs.values()) * 1000.)**-1
            xsec_scales = [xsec**-1 for xsec in xsecs.values()]

        # apply scales
        for name in limit_values.dtype.names:
            if name in ["limit", "limit_p1", "limit_m1", "limit_p2", "limit_m2", "observed"]:
                limit_values[name] *= limit_scales
        xsecs = OrderedDict((v, xsec * s) for (v, xsec), s in zip(xsecs.items(), xsec_scales))

        # some printing
        for v in np.linspace(-0.5, 0.5, 11):
            if v in limit_values[self.scan_parameter]:
                record = limit_values[limit_values[self.scan_parameter] == v][0]
                self.publish_message("{} = {} -> {} {}".format(self.scan_parameter, v,
                    record["limit"], xsec_unit or "({})".format(self.poi)))

        # preate theory values in the correct structure
        thy_values = {
            self.scan_parameter: limit_values[self.scan_parameter],
            "xsec": list(xsecs.values()),
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
            xsec_unit=xsec_unit,
            hh_process=self.br if xsec_unit and self.br in br_hh else None,
            model_parameters=model_parameters,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
