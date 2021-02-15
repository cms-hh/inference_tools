# coding: utf-8

"""
Tasks related to pulls and impacts.
"""

from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, PlotTask, view_output_plots
from dhi.tasks.combine import CombineCommandTask, POITask, CreateWorkspace
from dhi.config import poi_data, nuisance_labels
from dhi.datacard_tools import get_workspace_parameters


class PullsAndImpacts(POITask, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    skip_parameters = law.CSVParameter(
        default=(),
        description="comma-separated parameter names to be skipped; supports patterns; "
        "default: empty",
    )
    mc_stats = luigi.BoolParameter(
        default=False,
        description="when True, calculate pulls and impacts for MC stats nuisances as well; "
        "default: False",
    )

    mc_stats_patterns = ["prop_bin*"]

    force_n_pois = 1
    run_command_in_tmp = True

    def __init__(self, *args, **kwargs):
        super(PullsAndImpacts, self).__init__(*args, **kwargs)

        self._cache_branches = False

    def create_branch_map(self):
        # the first branch (index 0) is the nominal fit
        branches = ["nominal"]

        # read the nuisance parameters from the workspace file when present
        params = self.workspace_parameters
        if params:
            # remove poi
            params = [p for p in params if p != self.pois[0]]

            # remove mc stats parameters if requested, otherwise move them to the end
            is_mc_stats = lambda p: law.util.multi_match(p, self.mc_stats_patterns, mode=any)
            if self.mc_stats:
                sort_fn = lambda p: params.index(p) * (100000 if is_mc_stats(p) else 1)
                params = sorted(params, key=sort_fn)
            else:
                params = [p for p in params if not is_mc_stats(p)]

            # skip
            if self.skip_parameters:
                params = [p for p in params if not law.util.multi_match(p, self.skip_parameters)]

            # add to branches
            branches.extend(params)

        return branches

    @law.cached_workflow_property(setter=False, empty_value=law.no_value)
    def workspace_parameters(self):
        ws_input = CreateWorkspace.req(self).output()
        if not ws_input.exists():
            # not existing yet, return no_value to express that this value is still to be resolved
            return law.no_value
        else:
            self._cache_branches = True
            return get_workspace_parameters(ws_input.path)

    def workflow_requires(self):
        reqs = super(PullsAndImpacts, self).workflow_requires()
        reqs["workspace"] = self.requires_from_branch()
        return reqs

    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        name = "fit__{}__{}.root".format(self.get_output_postfix(), self.branch_data)
        return self.local_target(name)

    @property
    def blinded_args(self):
        if self.unblinded:
            return "--seed {self.branch}".format(self=self)
        else:
            return "--toys {self.toys} --seed {self.branch}".format(self=self)

    def build_command(self):
        # define branch dependent options
        if self.branch == 0:
            # initial fit
            branch_opts = "--algo singles"
        else:
            # nuisance fits
            branch_opts = "--algo impact -P {} --floatOtherPOIs 1 --saveInactivePOI 1".format(
                self.branch_data)

        return (
            "combine -M MultiDimFit {workspace}"
            " --verbose 1"
            " --mass {self.mass}"
            " {self.blinded_args}"
            " --redefineSignalPOIs {self.pois[0]}"
            " --setParameterRanges {self.pois[0]}={start},{stop}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --robustFit 1"
            " {self.combine_optimization_args}"
            " {self.custom_args}"
            " {branch_opts}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            start=poi_data[self.pois[0]].range[0],
            stop=poi_data[self.pois[0]].range[1],
            branch_opts=branch_opts,
        )

    def htcondor_output_postfix(self):
        postfix = super(PullsAndImpacts, self).htcondor_output_postfix()

        parts = []
        if self.mc_stats:
            postfix += "_mcstats"
        if self.skip_parameters:
            postfix += "_skip" + law.util.create_hash(sorted(self.skip_parameters))

        return postfix


class MergePullsAndImpacts(POITask):

    mc_stats = PullsAndImpacts.mc_stats
    skip_parameters = PullsAndImpacts.skip_parameters

    force_n_pois = 1

    def requires(self):
        return PullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = []
        if self.mc_stats:
            parts.append("mcstats")
        if self.skip_parameters:
            parts.append("skip_" + law.util.create_hash(sorted(self.skip_parameters)))

        name = self.join_postfix(["pulls_impacts", self.get_output_postfix(), parts]) + ".json"
        return self.local_target(name)

    @law.decorator.log
    def run(self):
        # get input targets, the branch map for resolving names and further parameter information
        req = self.requires()
        inputs = req.output().collection.targets
        branch_map = req.branch_map
        params = req.workspace_parameters

        # helper to extract results from an fit file target
        def extract_values(b, param):
            # read the plain values
            values = inputs[b].load(formatter="uproot")["limit"].arrays(param)
            # check that each arrays has length 3
            for p, v in values.items():
                if v.size != 3:
                    raise ValueError(
                        "fit result for parameter '{0}' of branch {1} stored at {2} must contain 3 "
                        "entries, but found {3}; this is likely due to a failed fit, so you can "
                        "try to remove the file and add\n\n"
                        "--{4}-custom-args='--X-rtd MINIMIZER_analytic'\n\n"
                        "to your law command, or you skip this particular branch by adding\n\n"
                        "--{4}-branches 0-{5},{6}-\n\n"
                        "which effectively selects all branches but the failing one".format(
                            p, b, inputs[b].path, v.size, "PullsAndImpacts", b - 1, b + 1
                        )
                    )
            # return the values dict when multiple params were given, otherwise a single array
            return values if isinstance(param, (list, tuple)) else values[param]

        # merge into data structure similar to the one produced by CombineHarvester
        # the only difference is that two sided impacts are stored as well
        data = OrderedDict(method="default")

        # load and store nominal value
        poi = self.pois[0]
        nom = extract_values(0, poi)
        data["POIs"] = [{"name": poi, "fit": nom[[1, 0, 2]].tolist()}]
        self.publish_message("read nominal values")

        # load and store parameter results
        data["params"] = []
        for b, name in branch_map.items():
            # skip the nominal fit
            if b == 0:
                continue

            vals = extract_values(b, [poi, name])
            d = OrderedDict()
            d["name"] = name
            d["type"] = params[name]["type"]
            d["groups"] = params[name]["groups"]
            d["prefit"] = params[name]["prefit"]
            d["fit"] = vals[name][[1, 0, 2]].tolist()
            d[poi] = vals[poi][[1, 0, 2]].tolist()
            d["impacts"] = {
                poi: [
                    d[poi][1] - d[poi][0],
                    d[poi][2] - d[poi][1],
                ]
            }
            d["impact_" + poi] = max(map(abs, d["impacts"][poi]))

            data["params"].append(d)
            self.publish_message("read values for " + name)

        self.output().dump(data, indent=4, formatter="json")


class PlotPullsAndImpacts(PlotTask, POITask):

    mc_stats = MergePullsAndImpacts.mc_stats
    skip_parameters = MergePullsAndImpacts.skip_parameters
    parameters_per_page = luigi.IntParameter(
        default=-1,
        description="number of parameters per page; creates a single page when < 1; only applied "
        "for file type 'pdf'; default: -1",
    )
    order_parameters = law.CSVParameter(
        default=(),
        description="list of parameters or files containing parameters line-by-line for ordering; "
        "supports patterns; default: empty",
    )
    order_by_impact = luigi.BoolParameter(
        default=False,
        description="when True, --parameter-order is neglected and parameters are ordered by "
        "absolute maximum impact; default: False",
    )
    pull_range = luigi.IntParameter(
        default=2,
        significant=False,
        description="the maximum integer value of pulls on the lower x-axis; default: 2",
    )
    impact_range = luigi.FloatParameter(
        default=-1.0,
        significant=False,
        description="the maximum value of impacts on the upper x-axis; for visual clearness, both "
        "x-axes have the same divisions, so make sure that the ratio impact_range/pull_range "
        "is a rational number with few digits; when not positive, an automatic value is chosen; "
        "default: -1.0",
    )
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None

    force_n_pois = 1

    def __init__(self, *args, **kwargs):
        super(PlotPullsAndImpacts, self).__init__(*args, **kwargs)

        # complain when parameters_per_page is set for non pdf file types
        if self.parameters_per_page > 0 and self.file_type != "pdf":
            self.logger.warning(
                "parameters_per_page is not supported for file_type {}".format(self.file_type)
            )
            self.parameters_per_page = -1

    def requires(self):
        return MergePullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = []
        if self.mc_stats:
            parts.append("mcstats")
        if self.skip_parameters:
            parts.append("skip_" + law.util.create_hash(sorted(self.skip_parameters)))

        name = self.create_plot_name(["pulls_impacts", self.get_output_postfix(), parts])
        return self.local_target(name)

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        output = self.output()
        output.parent.touch()

        # load input data
        data = self.input().load(formatter="json")

        # call the plot function
        self.call_plot_func(
            "dhi.plots.pulls_impacts.plot_pulls_impacts",
            path=output.path,
            data=data,
            parameters_per_page=self.parameters_per_page,
            skip_parameters=self.skip_parameters,
            order_parameters=self.order_parameters,
            order_by_impact=self.order_by_impact,
            pull_range=self.pull_range,
            impact_range=self.impact_range,
            labels=nuisance_labels,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
