# coding: utf-8

"""
Miscellaneous tasks.
"""

import os
import law
import luigi

from dhi.tasks.nlo.base import CombineCommandTask
from dhi.tasks.nlo.inference import CreateWorkspace, ImpactsPulls


# currently disabled since we decided to use a non-CMSSW environment which does not allow to run any
# CombineHarvester tool such as ValidateDatacards.py; this is, however, required to be run by HIG
# (https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWG/HiggsPAGPreapprovalChecks?rev=19) so we might
# want to revisit this in the future using (e.g.) task sandboxing
# class ValidateDatacard(CombineCommandTask):
#
#     mass = 125
#     input_card = luigi.Parameter(description="path to the input datacard")
#     verbosity = luigi.ChoiceParameter(var_type=int, default=1, choices=list(range(4)))
#
#     version = None
#
#     def output(self):
#         return self.local_target("validation.json")
#
#     def build_command(self):
#         return (
#             "ValidateDatacards.py {self.input_card}"
#             " --mass {self.mass}"
#             " --printLevel {self.verbosity}"
#             " --jsonFile {out}"
#         ).format(
#             self=self,
#             out=self.output().path,
#         )


class PostFitShapes(CombineCommandTask):
    def requires(self):
        return CreateWorkspace.req(self)

    def output(self):
        return self.local_target("fitDiagnostics.root")

    def build_command(self):
        return (
            "combine -M FitDiagnostics {workspace}"
            " -t -1"
            " -v 2"
            " -m {self.mass}"
            " --expectSignal 1"
            " --saveShapes --saveWithUncertainties"
            " --X-rtd MINIMIZER_analytic"
            " {self.combine_stable_options}"
            " {params}"
        ).format(
            self=self,
            workspace=self.input().path,
            params=ImpactsPulls.params,
        )


class CompareNuisances(CombineCommandTask):

    format = luigi.ChoiceParameter(default="html", choices=("html", "latex", "text"))

    format_to_ext = {
        "html": "html",
        "latex": "tex",
        "text": "txt",
    }

    def requires(self):
        return PostFitShapes.req(self)

    def output(self):
        return self.local_target("nuisances.{}".format(self.format_to_ext[self.format]))

    def build_command(self):
        script = "$DHI_SOFTWARE/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
        return "python {script} -a -f {self.format} {input} > {output}".format(
            self=self,
            script=script,
            input=self.input().path,
            output=self.output().path,
        )
