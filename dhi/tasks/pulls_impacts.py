# coding: utf-8

"""
Tasks related to pulls and impacts.
"""

from collections import OrderedDict

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow, view_output_plots
from dhi.tasks.combine import CombineCommandTask, POITask, POIPlotTask, CreateWorkspace
from dhi.config import poi_data, nuisance_labels
from dhi.datacard_tools import get_workspace_parameters


class PullsAndImpactsBase(POITask):

    only_parameters = law.CSVParameter(
        default=(),
        description="comma-separated parameter names to include; supports patterns; skips all "
        "others; no default",
    )
    skip_parameters = law.CSVParameter(
        default=(),
        description="comma-separated parameter names to be skipped; supports patterns; no default",
    )
    mc_stats = luigi.BoolParameter(
        default=False,
        description="when True, calculate pulls and impacts for MC stats nuisances as well; "
        "default: False",
    )

    mc_stats_patterns = ["*prop_bin*"]

    force_n_pois = 1
    allow_parameter_values_in_pois = True


class PullsAndImpacts(PullsAndImpactsBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

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
            if self.only_parameters:
                params = [p for p in params if law.util.multi_match(p, self.only_parameters)]
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
            return "--seed {self.branch} --toys {self.toys}".format(self=self)

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

        if self.mc_stats:
            postfix += "_mcstats"
        if self.only_parameters:
            postfix += "_only" + law.util.create_hash(sorted(self.only_parameters))
        if self.skip_parameters:
            postfix += "_skip" + law.util.create_hash(sorted(self.skip_parameters))

        return postfix


class MergePullsAndImpacts(PullsAndImpactsBase):

    def requires(self):
        return PullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = []
        if self.mc_stats:
            parts.append("mcstats")
        if self.only_parameters:
            parts.append("only_" + law.util.create_hash(sorted(self.only_parameters)))
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
        poi = self.pois[0]

        # extract all fit results
        fit_results = {}
        fail_info = []
        for b, name in branch_map.items():
            inp = inputs[b]
            if not inp.exists():
                self.logger.warning("input of branch {} at {} does not exist".format(
                    b, inp.path))
                continue

            tree = inp.load(formatter="uproot")["limit"]
            values = tree.arrays([poi] if b == 0 else [poi, name])
            # the fit converged when there are 3 values in the parameter array
            converged = values[poi if b == 0 else name].size == 3
            if not converged:
                fail_info.append((b, name, inp.path))
            else:
                fit_results[b] = values

        # throw an error with instructions when a fit failed
        if fail_info:
            failing_branches = [b for b, _, _ in fail_info]
            working_branches = [b for b in branch_map if b not in failing_branches]
            working_branches = law.util.range_join(working_branches, to_str=True)
            c = law.util.colored
            msg = "{} failing parameter fit(s) detected:\n".format(len(fail_info))
            for b, name, _ in fail_info:
                msg += "  {} (branch {})\n".format(name, b)
            msg += c("\nYou have two options\n\n", style=("underlined", "bright"))
            msg += "  " + c("1.", "magenta")
            msg += " You can try to remove the corresponding output files via\n\n"
            for _, _, path in fail_info:
                msg += "       rm {}\n".format(path)
            msg += "\n     and then add different options such as\n\n"
            msg += c("       --PullsAndImpacts-custom-args='--X-rtd MINIMIZER_no_analytic'\n",
                style="bright")
            msg += "\n     to your 'law run ...' command.\n\n"
            msg += "  " + c("2.", "magenta")
            msg += " You can proceed with the converging fits only by adding\n\n"
            msg += c("       --PullsAndImpacts-branches {}\n\n".format(working_branches),
                style="bright")
            msg += "     which effectively skips all failing fits.\n"
            raise Exception(msg)

        # merge values and parameter infos into data structure similar to the one produced by
        # CombineHarvester the only difference is that two sided impacts are stored as well
        data = OrderedDict(method="default")

        # load and store nominal value
        data["POIs"] = [{"name": poi, "fit": fit_results[0][poi][[1, 0, 2]].tolist()}]
        self.publish_message("read nominal values")

        # load and store parameter results
        data["params"] = []
        for b, vals in fit_results.items():
            # skip the nominal fit
            if b == 0:
                continue

            d = OrderedDict()
            name = branch_map[b]
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


class PlotPullsAndImpacts(PullsAndImpactsBase, POIPlotTask):

    hide_best_fit = luigi.BoolParameter(
        default=False,
        significant=False,
        description="do not show the label of the best fit value; should be considered when also "
        "using --unblinded; default: False",
    )
    parameters_per_page = luigi.IntParameter(
        default=-1,
        significant=False,
        description="number of parameters per page; creates a single page when < 1; only applied "
        "for file type 'pdf'; default: -1",
    )
    page = luigi.IntParameter(
        default=-1,
        description="the number of the page to print (starting at 0) when --parameters-per-page is "
        "set; prints all pages when negative; default: -1",
    )
    order_parameters = law.CSVParameter(
        default=(),
        significant=False,
        description="list of parameters or files containing parameters line-by-line for ordering; "
        "supports patterns; no default",
    )
    order_by_impact = luigi.BoolParameter(
        default=False,
        significant=False,
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
    label_size = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="size of the nuisance labels on the y-axis; uses the default of the plot when "
        "empty; no default",
    )

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None

    def __init__(self, *args, **kwargs):
        super(PlotPullsAndImpacts, self).__init__(*args, **kwargs)

        # complain when parameters_per_page is set for non pdf file types
        if self.parameters_per_page > 0 and self.file_type != "pdf":
            self.logger.warning(
                "parameters_per_page is not supported for file_type {}".format(self.file_type)
            )
            self.parameters_per_page = -1

        # show a warning when unblinded and not hiding the best fit value
        if self.unblinded and not self.hide_best_fit:
            self.logger.warning("running unblinded but not hiding the best fit value")

    def requires(self):
        return MergePullsAndImpacts.req(self)

    def output(self):
        # build the output file postfix
        parts = []
        if self.mc_stats:
            parts.append("mcstats")
        if self.only_parameters:
            parts.append("only_" + law.util.create_hash(sorted(self.only_parameters)))
        if self.skip_parameters:
            parts.append("skip_" + law.util.create_hash(sorted(self.skip_parameters)))
        if self.page >= 0:
            parts.append("page{}".format(self.page))

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
            selected_page=self.page,
            only_parameters=self.only_parameters,
            skip_parameters=self.skip_parameters,
            order_parameters=self.order_parameters,
            order_by_impact=self.order_by_impact,
            pull_range=self.pull_range,
            impact_range=self.impact_range,
            best_fit_value=not self.hide_best_fit,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            labels=nuisance_labels,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
        )
