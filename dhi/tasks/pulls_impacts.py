# coding: utf-8

"""
Tasks related to pulls and impacts.
"""

from collections import OrderedDict

import law
import luigi
import numpy as np

from dhi.tasks.base import HTCondorWorkflow, BoxPlotTask, view_output_plots
from dhi.tasks.combine import (
    CombineCommandTask, POITask, POIPlotTask, CreateWorkspace, MultiDatacardTask,
    POIMultiTask,
)
from dhi.tasks.snapshot import Snapshot, SnapshotUser
from dhi.datacard_tools import get_workspace_parameters


class PullsAndImpactsBase(POITask, SnapshotUser):

    method = luigi.ChoiceParameter(
        choices=("default", "hesse", "robust"),
        default="default",
        description="the computation method; 'default' means no approximation; "
        "choices: default,hesse,robust; default: default",
    )
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

    def get_output_postfix(self, join=True):
        parts = super(PullsAndImpactsBase, self).get_output_postfix(join=False)

        if self.method != "default":
            parts.append(self.method)
        if self.use_snapshot:
            parts.append("fromsnapshot")

        return self.join_postfix(parts) if join else parts

    def get_selected_parameters(self, workspace_parameters):
        if not workspace_parameters:
            return []

        # start with all except pois
        params = [p for p in workspace_parameters if p != self.pois[0]]

        # remove mc stats parameters if requested, otherwise move them to the end
        is_mc_stats = lambda p: law.util.multi_match(p, self.mc_stats_patterns, mode=any)
        if self.mc_stats:
            sort_fn = lambda p: params.index(p) * (100000 if is_mc_stats(p) else 1)
            params = sorted(params, key=sort_fn)
        else:
            params = [p for p in params if not is_mc_stats(p)]

        # filter and skip
        if self.only_parameters:
            params = [p for p in params if law.util.multi_match(p, self.only_parameters)]
        if self.skip_parameters:
            params = [p for p in params if not law.util.multi_match(p, self.skip_parameters)]

        return params


class PullsAndImpacts(PullsAndImpactsBase, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    run_command_in_tmp = True

    def __init__(self, *args, **kwargs):
        super(PullsAndImpacts, self).__init__(*args, **kwargs)

        self._cache_branches = False

    @law.cached_workflow_property(setter=False, empty_value=law.no_value)
    def workspace_parameters(self):
        ws_input = CreateWorkspace.req(self, branch=0).output()
        if not ws_input.exists():
            return law.no_value
        return get_workspace_parameters(ws_input.path)

    def create_branch_map(self):
        # the first branch (index 0) is the nominal fit of the poi
        branches = [self.pois[0]]

        # only the default method needs additional parameter fits
        if self.method == "default":
            params = self.get_selected_parameters(self.workspace_parameters)
            if params:
                # add to branches
                branches.extend(params)

                # mark that the branch map is cached from now on
                self._cache_branches = True

        return branches

    def workflow_requires(self):
        reqs = super(PullsAndImpacts, self).workflow_requires()
        reqs["workspace"] = CreateWorkspace.req(self)
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self)
        return reqs

    def requires(self):
        reqs = {"workspace": CreateWorkspace.req(self, branch=0)}
        if self.use_snapshot:
            reqs["snapshot"] = Snapshot.req(self, branch=0)
        return reqs

    def output(self):
        parts = [self.branch_data]
        name = lambda prefix: self.join_postfix([prefix, self.get_output_postfix(), parts]) + ".root"

        # the default fit result
        outputs = {"result": self.local_target(name("fit"))}

        # additional output files, depending on method and branch
        if self.method == "default":
            if self.branch == 0:
                outputs["multidimfit"] = self.local_target(name("multidimfit"))
        elif self.method == "hesse":
            outputs["multidimfit"] = self.local_target(name("multidimfit"))
        elif self.method == "robust":
            outputs["robusthesse"] = self.local_target(name("robusthesse"))
            outputs["hessian"] = self.local_target(name("hessian"))

        return outputs

    def build_command(self):
        outputs = self.output()

        # get the workspace to use and define snapshot args
        if self.use_snapshot:
            workspace = self.input()["snapshot"].path
            snapshot_args = " --snapshotName MultiDimFit"
        else:
            workspace = self.input()["workspace"].path
            snapshot_args = ""

        # args for blinding / unblinding
        if self.unblinded:
            blinded_args = "--seed {self.branch}".format(self=self)
        else:
            blinded_args = "--seed {self.branch} --toys {self.toys}".format(self=self)

        # define output files as (src, dst, optional) to be moved after command execution
        mv_files = [(
            "higgsCombineTest.MultiDimFit.mH{self.mass_int}.{self.branch}.root".format(self=self),
            outputs["result"].path,
            False,
        )]

        # define branch and method dependent options
        if self.method == "default":
            if self.branch == 0:
                # initial fit
                branch_opts = (
                    " --algo singles"
                    " --saveFitResult"
                )
                mv_files.append(("multidimfitTest.root", outputs["multidimfit"].path, False))
            else:
                # nuisance fits
                branch_opts = (
                    " --algo impact"
                    " --parameters {}"
                    " --floatOtherPOIs 1"
                    " --saveInactivePOI 1"
                ).format(self.branch_data)
        elif self.method == "hesse":
            # setup a single fit
            branch_opts = (
                " --algo none"
                " --floatOtherPOIs 1"
                " --saveInactivePOI 1"
                " --saveFitResult"
            )
            mv_files.append(("multidimfitTest.root", outputs["multidimfit"].path, False))
        elif self.method == "robust":
            # setup a single fit
            branch_opts = (
                " --algo none"
                " --floatOtherPOIs 1"
                " --saveInactivePOI 1"
                " --robustHesse 1"
                " --robustHesseSave hessian.root"
            )
            mv_files.append(("robustHesseTest.root", outputs["robusthesse"].path, False))
            mv_files.append(("hessian.root", outputs["hessian"].path, False))
        else:
            raise NotImplementedError

        # move statements for saving output files
        mv_cmd = " && ".join(
            ("( mv {} {} || true )" if opt else "mv {} {}").format(src, dst)
            for src, dst, opt in mv_files
        )

        # define the basic command
        cmd = (
            "combine -M MultiDimFit {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " {snapshot_args}"
            " --redefineSignalPOIs {self.pois[0]}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --setParameters {self.joined_parameter_values}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --saveNLL"
            " {self.combine_optimization_args}"
            " {branch_opts}"
            " && "
            "{mv_cmd}"
        ).format(
            self=self,
            workspace=workspace,
            snapshot_args=snapshot_args,
            blinded_args=blinded_args,
            branch_opts=branch_opts,
            mv_cmd=mv_cmd,
        )

        return cmd

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

    keep_failures = luigi.BoolParameter(
        default=False,
        description="keep failed fits and mark them as invalid; default: False",
    )

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

        # extract all fit results, depending on the method
        if self.method == "default":
            # default method: loop through fit result per branch
            fit_results = {}
            fail_info = []
            for b, name in branch_map.items():
                inp = inputs[b]["result"]
                if not inp.exists():
                    self.logger.warning("input {} of branch {} does not exist".format(inp.path, b))
                    continue

                # load the result
                values = self.load_default_fit(inp, poi, name, keep_failures=self.keep_failures)
                if values:
                    fit_results[name] = values
                else:
                    fail_info.append((b, name, inp.path))

            # throw an error with instructions when a fit failed
            if fail_info:
                failing_branches = [b for b, _, _ in fail_info]
                working_branches = [b for b in branch_map if b not in failing_branches]
                working_branches = law.util.range_join(working_branches, to_str=True)
                c = law.util.colored
                msg = "{} failing parameter fit(s) detected:\n".format(len(fail_info))
                for b, name, _ in fail_info:
                    msg += "  {} (branch {})\n".format(name, b)
                msg += "\nBranches: {}".format(",".join(map(str, failing_branches)))
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
                msg += "     which effectively skips all failing fits in your case.\n"
                raise Exception(msg)

        elif self.method == "hesse":
            # load all parameter fits from the mdf result
            fit_results = self.load_hesse_fits(inputs[0]["multidimfit"], poi,
                self.get_selected_parameters(params))

        elif self.method == "robust":
            # load all parameter fits from the robustHesse result
            fit_results = self.load_robust_fits(inputs[0]["robusthesse"], poi,
                self.get_selected_parameters(params))

        else:
            raise NotImplementedError

        # merge values and parameter infos into data structure similar to the one produced by
        # CombineHarvester the only difference is that two sided impacts are stored as well
        data = OrderedDict()

        # save the method
        data["method"] = self.method

        # load and store nominal value
        data["POIs"] = [{"name": poi, "fit": fit_results[poi][poi][[1, 0, 2]].tolist()}]
        self.publish_message("read nominal values")

        # load and store parameter results
        data["params"] = []
        for name, vals in fit_results.items():
            # skip the nominal fit
            if name == poi:
                continue

            d = OrderedDict()
            # parameter name
            d["name"] = name
            # parameter pdf type
            d["type"] = params[name]["type"]
            # list of groups
            d["groups"] = params[name]["groups"]
            # prefit values in a harvester-compatible format, i.e., down, nominal, up
            d["prefit"] = params[name]["prefit"]
            # postfit values in a harvester-compatible format, i.e., down, nominal, up
            d["fit"] = vals[name][[1, 0, 2]].tolist()
            # POI impact values
            d[poi] = vals[poi][[1, 0, 2]].tolist()
            d["impacts"] = {
                poi: [
                    d[poi][1] - d[poi][0],
                    d[poi][2] - d[poi][1],
                ]
            }
            # maximum impact on that POI
            d["impact_" + poi] = max(map(abs, d["impacts"][poi]))

            data["params"].append(d)

        self.output().dump(data, indent=4, formatter="json")

    @classmethod
    def load_default_fit(cls, target, poi, param, keep_failures=False):
        tree = target.load(formatter="uproot")["limit"]
        values = tree.arrays([poi] if param == poi else [poi, param])

        # the fit converged when there are 3 values in the parameter array
        converged = values[param].size == 3
        if converged:
            return values
        elif keep_failures:
            return {
                poi: np.array([np.nan, np.nan, np.nan]),
                param: np.array([np.nan, np.nan, np.nan]),
            }

        # error case
        return None

    @classmethod
    def load_hesse_fits(cls, target, poi, params):
        res = OrderedDict()

        # load the roofit result
        with target.load(formatter="root") as f:
            rfr = f.Get("fit_mdf")
            float_params = rfr.floatParsFinal()

            # check the fit status
            if rfr.covQual() != 3:
                raise Exception("inaccurate covariance matrix")

            # load the fit results per parameter
            # logic bluntly copied from CombineHarvester/CombineTools/python/combine/utils.py
            for param in [poi] + params:
                res[param] = OrderedDict()
                for p in [poi, param]:
                    pj = float_params.find(p)
                    if not pj:
                        raise Exception("parameter {} not in floatParsFinal result".format(p))

                    vj = pj.getVal()
                    ej = pj.getError()
                    c = rfr.correlation(param, p)
                    res[param][p] = np.array([vj, vj - ej * c, vj + ej * c])

        return res

    @classmethod
    def load_robust_fits(cls, target, poi, params):
        res = OrderedDict()

        # load the roofit result
        with target.load(formatter="root") as f:
            float_params = f.Get("floatParsFinal")
            corr = f.Get("h_correlation")

            # load the fit results per parameter
            # logic bluntly copied from CombineHarvester/CombineTools/python/combine/utils.py
            for param in [poi] + params:
                if not float_params.find(param):
                    continue

                res[param] = OrderedDict()
                idx_p = corr.GetXaxis().FindBin(param)
                for p in [poi, param]:
                    pj = float_params.find(p)
                    vj = pj.getVal()
                    ej = pj.getError()
                    idx = corr.GetXaxis().FindBin(p)
                    c = corr.GetBinContent(idx_p, idx)
                    res[param][p] = np.array([vj, vj - ej * c, vj + ej * c])

        return res


class PlotPullsAndImpacts(PullsAndImpactsBase, POIPlotTask, BoxPlotTask):

    keep_failures = MergePullsAndImpacts.keep_failures
    show_best_fit = luigi.BoolParameter(
        default=True,
        significant=False,
        description="when True, show the label of the best fit value; default: True",
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
    labels = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a python file containing a function 'rename_nuisance' taking a single "
        "argument, or a json file containing a mapping 'name' -> 'new_name' to translate nuisance "
        "parameter names with (ROOT) latex support; key-value pairs support pattern replacement "
        r"with regular expressions, e.g. '^prefix_(.*)$' -> 'new_\1', where keys are forced to "
        r"start with '^' and end with '$', and '\n' in values are replaced with the n-th match; "
        "no default",
    )

    x_min = None
    x_max = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None
    save_hep_data = None

    def __init__(self, *args, **kwargs):
        super(PlotPullsAndImpacts, self).__init__(*args, **kwargs)

        # complain when parameters_per_page is set for non pdf file types
        if self.parameters_per_page > 0 and self.page < 0 and "pdf" not in self.file_types:
            self.logger.warning("parameters_per_page is only supported for file_type 'pdf', but "
                "got {}".format(self.file_types))
            self.parameters_per_page = -1

        # show a warning when unblinded, not in paper mode and not hiding the best fit value
        if self.unblinded and not self.paper and self.show_best_fit:
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

        names = self.create_plot_names(["pulls_impacts", self.get_output_postfix(), parts])
        return [self.local_target(name) for name in names]

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        # load input data
        data = self.input().load(formatter="json")

        # call the plot function
        self.call_plot_func(
            "dhi.plots.pulls_impacts.plot_pulls_impacts",
            paths=[outp.path for outp in outputs],
            data=data,
            parameters_per_page=self.parameters_per_page,
            selected_page=self.page,
            only_parameters=self.only_parameters,
            skip_parameters=self.skip_parameters,
            order_parameters=self.order_parameters,
            order_by_impact=self.order_by_impact,
            pull_range=self.pull_range,
            impact_range=self.impact_range,
            best_fit_value=self.show_best_fit,
            labels=None if self.labels == law.NO_STR else self.labels,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            pad_width=None if self.pad_width == law.NO_INT else self.pad_width,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            right_margin=None if self.right_margin == law.NO_INT else self.right_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            paper=self.paper,
        )


class PlotMultiplePullsAndImpacts(PlotPullsAndImpacts, POIMultiTask, MultiDatacardTask):

    # fix some parameters
    order_by_impact = False
    show_best_fit = False
    mc_stats = False
    parameters_per_page = False

    compare_multi_sequence = "multi_datacards"

    @classmethod
    def modify_param_values(cls, params):
        params = PlotPullsAndImpacts.modify_param_values.__func__.__get__(cls)(params)
        params = MultiDatacardTask.modify_param_values.__func__.__get__(cls)(params)
        return params

    def requires(self):
        return {
            name: MergePullsAndImpacts.req(self, datacards=datacards, **kwargs)
            for name, datacards, kwargs in zip(
                self.datacard_names, self.multi_datacards, self.get_multi_task_kwargs(),
            )
        }

    @law.decorator.log
    @law.decorator.notify
    @view_output_plots
    @law.decorator.safe_output
    def run(self):
        # prepare the output
        outputs = self.output()
        outputs[0].parent.touch()

        inp = self.input()
        # load input data
        data = OrderedDict({
            name: inp[name].load(formatter="json")
            for name in sorted(self.input().keys())
        })

        # add datacard_name as tag
        for name, info in data.items():
            params = info["params"]
            for entry in params:
                entry["tag"] = name

        # we can only compare those, which are there for all datacard groups
        # (and those which were choosen through cli args)
        only_parameters = set.intersection(*[
            set([p["name"] for p in d["params"]])
            for d in data.values()
        ])
        if self.only_parameters:
            only_parameters &= set(self.only_parameters)
        only_parameters = tuple(only_parameters)

        # flatten now, take first for 'POIs' and 'method',
        # just as placeholder to preserve 'data' format for plot function
        first = next(iter(data.values()))
        data = {
            "POIs": first["POIs"],
            "params": sum([d["params"] for d in data.values()], []),
            "method": first["method"],
        }

        # abuse and call the plot function
        self.call_plot_func(
            "dhi.plots.pulls_impacts.plot_pulls_impacts",
            paths=[outp.path for outp in outputs],
            data=data,
            parameters_per_page=len(self.datacard_names),
            selected_page=self.page,
            only_parameters=only_parameters,
            skip_parameters=self.skip_parameters,
            order_parameters=self.order_parameters,
            order_by_impact=self.order_by_impact,
            pull_range=self.pull_range,
            impact_range=self.impact_range,
            best_fit_value=self.show_best_fit,
            labels=None if self.labels == law.NO_STR else self.labels,
            label_size=None if self.label_size == law.NO_INT else self.label_size,
            pad_width=None if self.pad_width == law.NO_INT else self.pad_width,
            left_margin=None if self.left_margin == law.NO_INT else self.left_margin,
            right_margin=None if self.right_margin == law.NO_INT else self.right_margin,
            entry_height=None if self.entry_height == law.NO_INT else self.entry_height,
            campaign=self.campaign if self.campaign != law.NO_STR else None,
            paper=self.paper,
        )
