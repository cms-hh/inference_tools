# coding: utf-8

"""
NLO inference tasks.
"""

import os
import itertools
from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow
from dhi.tasks.combine import (
    CombineCommandTask, DatacardTask, POITask1D, POIScanTask1D, POIScanTask1DWithR,
    POIScanTask2D,
)
from dhi.util import linspace
from dhi.config import poi_data
from dhi.datacard_tools import get_workspace_parameters, bundle_datacard
from dhi.scripts.remove_processes import remove_processes as remove_processes_script


class CombineDatacards(DatacardTask, CombineCommandTask):

    def output(self):
        return self.local_target_dc("datacard.txt")

    def build_command(self, datacards=None):
        if not datacards:
            datacards = self.datacards

        inputs = " ".join(datacards)
        output = self.output()

        return "combineCards.py {} {} > {}".format(self.custom_args, inputs, output.path)

    @law.decorator.safe_output
    def run(self):
        # before running the actual card combination command, copy shape files and handle collisions
        # first, create a tmp dir to work in
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # remove any bin name from the datacard paths
        datacards = [self.split_datacard_path(card)[0] for card in self.datacards]
        bin_names = [self.split_datacard_path(card)[1] for card in self.datacards]

        # run the bundling for all cards which handles collision-free copying on its own
        datacards = [os.path.basename(bundle_datacard(card, tmp_dir.path)) for card in datacards]

        # add back bin names
        datacards = [
            ("{}={}".format(bin_name, card) if bin_name else card)
            for bin_name, card in zip(bin_names, datacards)
        ]

        # build and run the command
        output = self.output()
        output.parent.touch()
        self.run_command(self.build_command(datacards), cwd=tmp_dir.path)

        # remove ggf and vbf processes that are not covered by the physics model
        mod, model = self.load_hh_model()
        all_hh_processes = {sample.label for sample in mod.ggf_samples.values()}
        all_hh_processes |= {sample.label for sample in mod.vbf_samples.values()}
        model_hh_processes = {sample.label for sample in model.ggf_formula.sample_list}
        model_hh_processes |= {sample.label for sample in model.vbf_formula.sample_list}
        to_remove = all_hh_processes - model_hh_processes
        if to_remove:
            self.logger.info("trying to remove processe(s) {} from the combined datacard as they "
                "are not part of the phyics model {}".format(",".join(to_remove), self.hh_model))
            remove_processes_script(output.path, map("{}*".format, to_remove))

        # copy shape files to output location
        for basename in tmp_dir.listdir(pattern="*.root", type="f"):
            tmp_dir.child(basename).copy_to(output.parent)


class CreateWorkspace(DatacardTask, CombineCommandTask):

    def requires(self):
        return CombineDatacards.req(self)

    def output(self):
        return self.local_target_dc("workspace.root")

    def build_command(self):
        return (
            "text2workspace.py {datacard}"
            " -o {workspace}"
            " -m {self.mass}"
            " -P dhi.models.{self.hh_model}"
            " --PO doNNLOscaling={nnlo}"
            " {self.custom_args}"
        ).format(
            self=self,
            datacard=self.input().path,
            workspace=self.output().path,
            nnlo=not self.hh_nlo,
        )


class UpperLimits(POIScanTask1DWithR, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return linspace(self.poi_range[0], self.poi_range[1], self.poi_points)

    def workflow_requires(self):
        reqs = super(UpperLimits, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target_dc("limit__{}__{}_{}.root".format(
            self.r_poi, self.poi, self.branch_data))

    def build_command(self):
        return (
            "combine -M AsymptoticLimits {workspace}"
            " -v 1"
            " -m {self.mass}"
            " -t -1"
            " --run expected"
            " --noFitAsimov"
            " --redefineSignalPOIs {self.r_poi}"
            " --setParameters {self.set_pois},{self.poi}={point}"
            " --freezeParameters {frozen_pois}"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.AsymptoticLimits.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point=self.branch_data,
            frozen_pois=self.get_frozen_pois([p for p in self.all_pois if p != self.r_poi])
        )


class MergeUpperLimits(POIScanTask1DWithR):

    def requires(self):
        return UpperLimits.req(self)

    def output(self):
        return self.local_target_dc("limits__{}.npz".format(self.get_poi_postfix()))

    @law.decorator.log
    def run(self):
        import numpy as np

        records = []
        dtype = [
            (self.poi, np.float32),
            ("limit", np.float32),
            ("limit_p1", np.float32),
            ("limit_m1", np.float32),
            ("limit_p2", np.float32),
            ("limit_m2", np.float32),
        ]
        scan_task = self.requires()
        for branch, inp in self.input()["collection"].targets.items():
            poi_value = scan_task.branch_map[branch]
            records.append((poi_value,) + self.load_limits(inp))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")

    @classmethod
    def load_limits(cls, inp, poi_value=None):
        import numpy as np

        # load raw values
        limits = inp.load(formatter="uproot")["limit"].array("limit")

        # convert to (nominal, err1_up, err1_down, err2_up, err2_down)
        # TODO: what to do when errors occurred?
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

        # prepend the poi_value when given
        if poi_value is not None:
            values = (poi_value,) + values

        return values


class LikelihoodScan1D(POIScanTask1D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return linspace(self.poi_range[0], self.poi_range[1], self.poi_points)

    def workflow_requires(self):
        reqs = super(LikelihoodScan1D, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target_dc("likelihood__{}_{}.root".format(self.poi, self.branch_data))

    def build_command(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -v 1"
            " -m {self.mass}"
            " -t -1"
            " --algo grid"
            " --points {self.poi_points}"
            " --setParameterRanges {self.poi}={self.poi_range[0]},{self.poi_range[1]}"
            " --firstPoint {self.branch}"
            " --lastPoint {self.branch}"
            " --alignEdges 1"
            " --redefineSignalPOIs {self.poi}"
            " --setParameters {self.set_pois}"
            " --freezeParameters {self.frozen_pois}"
            " --robustFit 1"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point=self.branch_data,
        )


class MergeLikelihoodScan1D(POIScanTask1D):
    def requires(self):
        return LikelihoodScan1D.req(self)

    def output(self):
        return self.local_target_dc("likelihoods__{}.npz".format(self.get_poi_postfix()))

    @law.decorator.log
    def run(self):
        import numpy as np

        data = []
        dtype = [(self.poi, np.float32), ("delta_nll", np.float32)]
        poi_min = np.nan
        branch_map = self.requires().branch_map
        for b, inp in self.input()["collection"].targets.items():
            f = inp.load(formatter="uproot")["limit"]
            failed = len(f["deltaNLL"].array()) <= 1
            if failed:
                data.append((branch_map[b], np.nan))
                continue
            # save the best fit value
            if poi_min is np.nan:
                poi_min = f[self.poi].array()[0]
            # store the value of that point
            data.append((branch_map[b], f["deltaNLL"].array()[1]))

        data = np.array(data, dtype=dtype)
        self.output().dump(poi_min=poi_min, data=data, formatter="numpy")


class LikelihoodScan2D(POIScanTask2D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True
    store_pois_sorted = True

    def create_branch_map(self):
        range1 = linspace(self.poi1_range[0], self.poi1_range[1], self.poi1_points)
        range2 = linspace(self.poi2_range[0], self.poi2_range[1], self.poi2_points)
        return list(itertools.product(range1, range2))

    def workflow_requires(self):
        reqs = super(LikelihoodScan2D, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        # create a verbose file postfix and apply sorting manually here
        postfix1 = "{}_{}".format(self.poi1, self.branch_data[0])
        postfix2 = "{}_{}".format(self.poi2, self.branch_data[1])
        if self.store_pois_sorted and self.poi1 > self.poi2:
            postfix1, postfix2 = postfix2, postfix1

        return self.local_target_dc("likelihood__{}__{}.root".format(postfix1, postfix2))

    def build_command(self):
        return (
            "combine -M MultiDimFit {workspace}"
            " -v 1"
            " -m {self.mass}"
            " -t -1"
            " --algo grid"
            " --gridPoints {self.poi1_points},{self.poi2_points}"
            " --setParameterRanges {self.poi1}={self.poi1_range[0]},{self.poi1_range[1]}:"
            "{self.poi2}={self.poi2_range[0]},{self.poi2_range[1]}"
            " --firstPoint {self.branch}"
            " --lastPoint {self.branch}"
            " --alignEdges 1"
            " --redefineSignalPOIs {self.poi1},{self.poi2}"
            " --setParameters {self.set_pois}"
            " --freezeParameters {self.frozen_pois}"
            " --robustFit 1"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
        )


class MergeLikelihoodScan2D(POIScanTask2D):

    store_pois_sorted = True

    def requires(self):
        return LikelihoodScan2D.req(self)

    def output(self):
        return self.local_target_dc("likelihoods__{}.npz".format(self.get_poi_postfix()))

    @law.decorator.log
    def run(self):
        import numpy as np

        data = []
        dtype = [
            (self.poi1, np.float32),
            (self.poi2, np.float32),
            ("delta_nll", np.float32),
        ]
        poi1_min = np.nan
        poi2_min = np.nan
        branch_map = self.requires().branch_map
        for b, inp in self.input()["collection"].targets.items():
            f = inp.load(formatter="uproot")["limit"]
            failed = len(f["deltaNLL"].array()) <= 1
            if failed:
                data.append(branch_map[b] + (np.nan,))
                continue
            # save the best fit value
            if poi1_min is np.nan or poi2_min is np.nan:
                poi1_min = f[self.poi1].array()[0]
                poi2_min = f[self.poi2].array()[0]
            # store the value of that point
            data.append((branch_map[b] + (f["deltaNLL"].array()[1],)))

        data = np.array(data, dtype=dtype)
        self.output().dump(poi1_min=poi1_min, poi2_min=poi2_min, data=data, formatter="numpy")


class PullsAndImpacts(POITask1D, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    mc_stats = luigi.BoolParameter(
        default=False,
        description="when True, calculate pulls and impacts for MC stats nuisances as well, "
        "default: False",
    )

    mc_stats_patterns = ["prop_bin*"]

    reset_branch_map_before_run = True
    run_command_in_tmp = True

    def create_branch_map(self):
        # the first branch (index 0) is the nominal fit
        branches = ["nominal"]

        # read the nuisance parameters from the workspace file when present
        params = self.workspace_parameters
        if params:
            # remove poi
            params = [p for p in params if p != self.poi]
            # remove mc stats parameters if requested, otherwise move them to the end
            is_mc_stats = lambda p: law.util.multi_match(p, self.mc_stats_patterns, mode=any)
            if self.mc_stats:
                sort_fn = lambda p: params.index(p) * (100000 if is_mc_stats(p) else 1)
                params = sorted(params, key=sort_fn)
            else:
                params = [p for p in params if not is_mc_stats(p)]
            # add to branches
            branches.extend(params)

        return branches

    @law.cached_workflow_property(setter=False)
    def workspace_parameters(self):
        ws_input = CreateWorkspace.req(self).output()
        if not ws_input.exists():
            # not existing yet, return no_value to mark this value is still to be resolved
            return law.util.no_value
        else:
            return get_workspace_parameters(ws_input.path)

    def workflow_requires(self):
        reqs = super(PullsAndImpacts, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        # build the output file postfix
        postfix = "{}__{}".format(self.poi, self.branch_data)

        return self.local_target_dc("fit__{}.root".format(postfix))

    def build_command(self):
        # build the part of the command that is common between the initial fit and nuisance fits
        common_cmd = (
            "combine -M MultiDimFit {workspace}"
            " -v 1"
            " -m {self.mass}"
            " -t -1"
            " --expectSignal 1"
            " --robustFit 1"
            " --redefineSignalPOIs {self.poi}"
            " --setParameterRanges {self.poi}={start},{stop}"
            " --setParameters {self.set_pois}"
            " --freezeParameters {self.frozen_pois}"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " {{branch_opts}}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            start=poi_data[self.poi].range[0],
            stop=poi_data[self.poi].range[1],
        )

        # merge with branch dependent options
        if self.branch == 0:
            # initial fit
            branch_opts = "--algo singles"
        else:
            # nuisance fits
            branch_opts = (
                " --algo impact"
                " -P {}"
                " --floatOtherPOIs 1"
                " --saveInactivePOI 1"
            ).format(self.branch_data)

        return common_cmd.format(branch_opts=branch_opts)


class MergePullsAndImpacts(POITask1D):

    mc_stats = PullsAndImpacts.mc_stats

    def requires(self):
        return PullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = [self.poi]
        if self.mc_stats:
            parts.append("mcstats")
        postfix = "__".join(parts)

        return self.local_target_dc("pulls_impacts__{}.json".format(postfix))

    @law.decorator.log
    def run(self):
        # get input targets, the branch map for resolving names and further parameter information
        req = self.requires()
        inputs = req.output().collection.targets
        branch_map = req.branch_map
        params = req.workspace_parameters

        # helper to extract results from an fit file target
        def extract_values(inp, param):
            # read the plain values
            values = inp.load(formatter="uproot")["limit"].arrays(param)
            # check that each arrays has length 3
            for p, v in values.items():
                if v.size != 3:
                    raise ValueError("fit result for parameter '{}' at {} must contain 3 entries, "
                        "but found {}".format(p, inp.path, v.size))
            # return the values dict when multiple params were given, otherwise a single array
            return values if isinstance(param, (list, tuple)) else values[param]

        # merge into data structure similar to the one produced by CombineHarvester
        # the only difference is that two sided impacts are stored as well
        data = OrderedDict(method="default")

        # load and store nominal value
        nom = extract_values(inputs[0], self.poi)
        data["POIs"] = [{"name": self.poi, "fit": nom[[1, 0, 2]].tolist()}]
        self.publish_message("read nominal values")

        # load and store parameter results
        data["params"] = []
        for b, name in branch_map.items():
            # skip the nominal fit
            if b == 0:
                continue

            vals = extract_values(inputs[b], [self.poi, name])
            d = OrderedDict()
            d["name"] = name
            d["type"] = params[name]["type"]
            d["groups"] = params[name]["groups"]
            d["prefit"] = params[name]["prefit"]
            d["fit"] = vals[name][[1, 0, 2]].tolist()
            d[self.poi] = vals[self.poi][[1, 0, 2]].tolist()
            d["impacts"] = {self.poi: [
                d[self.poi][1] - d[self.poi][0],
                d[self.poi][2] - d[self.poi][1],
            ]}
            d["impact_" + self.poi] = max(map(abs, d["impacts"][self.poi]))

            data["params"].append(d)
            self.publish_message("read values for " + name)

        self.output().dump(data, formatter="json")


class SignificanceScan(POIScanTask1DWithR, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def create_branch_map(self):
        return linspace(self.poi_range[0], self.poi_range[1], self.poi_points)

    def workflow_requires(self):
        reqs = super(SignificanceScan, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target_dc("significance__{}__{}_{}.root".format(
            self.r_poi, self.poi, self.branch_data))

    def build_command(self):
        return (
            "combine -M Significance {workspace}"
            " -m {self.mass}"
            " -v 1"
            " -t -1"
            " --signalForSignificance 0"
            " --redefineSignalPOIs {self.r_poi}"
            " --setParameters {self.set_pois},{self.poi}={point}"
            " --freezeParameters {frozen_pois}"
            " {self.combine_stable_options}"
            " {self.custom_args}"
            " && "
            "mv higgsCombineTest.Significance.mH{self.mass_int}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            point=self.branch_data,
            frozen_pois=self.get_frozen_pois([p for p in self.all_pois if p != self.r_poi])
        )


class MergeSignificanceScan(POIScanTask1DWithR):

    def requires(self):
        return SignificanceScan.req(self)

    def output(self):
        return self.local_target_dc("significance__{}.npz".format(self.get_poi_postfix()))

    @law.decorator.log
    def run(self):
        import numpy as np
        import scipy as sp
        import scipy.stats

        records = []
        dtype = [
            (self.poi, np.float32),
            ("significance", np.float32),
            ("p_value", np.float32),
        ]
        scan_task = self.requires()
        for branch, inp in self.input()["collection"].targets.items():
            poi_value = scan_task.branch_map[branch]
            sig = inp.load(formatter="uproot")["limit"].array("limit")[0]
            pval = scipy.stats.norm.sf(sig)
            records.append((poi_value, sig, pval))

        data = np.array(records, dtype=dtype)
        self.output().dump(data=data, formatter="numpy")
