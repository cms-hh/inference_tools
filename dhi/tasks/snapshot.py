# coding: utf-8

"""
Tasks related to defining and using snapshots.
"""

import copy

import law
import luigi

from dhi.tasks.base import HTCondorWorkflow
from dhi.tasks.combine import (
    CombineCommandTask,
    POITask,
    CreateWorkspace,
)


class Snapshot(POITask, CombineCommandTask, law.LocalWorkflow, HTCondorWorkflow):

    pois = copy.copy(POITask.pois)
    pois._default = ("r",)

    force_scan_parameters_equal_pois = True
    allow_parameter_values_in_pois = True
    run_command_in_tmp = True

    exclude_params_req_get = {"start_branch", "end_branch", "branches", "workflow"}
    prefer_params_cli = {
        "toys", "frozen_parameters", "frozen_groups", "minimizer", "parameter_values",
        "parameter_ranges", "workflow", "max_runtime",
    }

    def create_branch_map(self):
        # single branch that does not need special data
        return [None]

    def workflow_requires(self):
        reqs = super(Snapshot, self).workflow_requires()
        reqs["workspace"] = CreateWorkspace.req(self)
        return reqs

    def requires(self):
        return CreateWorkspace.req(self, branch=0)

    def output(self):
        name = self.join_postfix(["snapshot", self.get_output_postfix()]) + ".root"
        return self.local_target(name)

    def build_command(self, fallback_level):
        # args for blinding / unblinding
        if self.unblinded:
            blinded_args = "--seed {self.branch}".format(self=self)
        else:
            blinded_args = "--seed {self.branch} --toys {self.toys}".format(self=self)

        # build the command
        cmd = (
            "combine -M MultiDimFit {workspace}"
            " {self.custom_args}"
            " --verbose 1"
            " --mass {self.mass}"
            " {blinded_args}"
            " --redefineSignalPOIs {self.joined_pois}"
            " --setParameters {self.joined_parameter_values}"
            " --setParameterRanges {self.joined_parameter_ranges}"
            " --freezeParameters {self.joined_frozen_parameters}"
            " --freezeNuisanceGroups {self.joined_frozen_groups}"
            " --saveWorkspace"
            " --saveNLL"
            " {self.combine_optimization_args}"
            " && "
            "mv higgsCombineTest.MultiDimFit.mH{self.mass_int}.{self.branch}.root {output}"
        ).format(
            self=self,
            workspace=self.input().path,
            output=self.output().path,
            blinded_args=blinded_args,
        )

        return cmd


class SnapshotUser(object):

    use_snapshot = luigi.BoolParameter(
        default=False,
        description="when set, perform a MultiDimFit first and use its workspace as a snapshot; "
        "default: False",
    )
