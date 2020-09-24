# coding: utf-8

"""
Miscellaneous tasks.
"""

import os
import law
import luigi

from dhi.tasks.base import CHBase
from dhi.tasks.nlo.inference import CombDatacards, NLOT2W, ImpactsPulls


class ValidateDatacard(CHBase):
    version = None

    input_card = luigi.Parameter(description="Path to input datacard")

    verbosity = luigi.ChoiceParameter(default="1", choices=("0", "1", "2", "3"))

    def output(self):
        return self.local_target("validation.json")

    @property
    def cmd(self):
        return "ValidateDatacards.py {input_card} --mass {mass} --printLevel {verbosity} --jsonFile {out}".format(
            input_card=self.input_card,
            mass=self.mass,
            verbosity=self.verbosity,
            out=self.output().path,
        )


class PostFitShapes(CHBase):
    def requires(self):
        return NLOT2W.req(self)

    def output(self):
        return self.local_target("fitDiagnostics.root")

    @property
    def cmd(self):
        return (
            "combine -M FitDiagnostics {workspace} -t -1 --expectSignal 1 "
            "--X-rtd MINIMIZER_analytic -m {mass} -v 2 {params} "
            "{options} --saveShapes --saveWithUncertainties "
        ).format(
            workspace=self.input().path,
            mass=self.mass,
            params=ImpactsPulls.params,
            options=self.stable_options,
        )


class CompareNuisances(CHBase):
    exe = "$CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"

    format = luigi.ChoiceParameter(default="html", choices=("html", "latex", "text"))

    def requires(self):
        return PostFitShapes.req(self)

    def output(self):
        return self.local_target("nuisances.{}".format(self.ext))

    def __init__(self, *args, **kwargs):
        super(CompareNuisances, self).__init__(*args, **kwargs)
        if self.format == "html":
            self.ext = "html"
        if self.format == "latex":
            self.ext = "tex"
        if self.format == "text":
            self.ext = "txt"

    @property
    def cmd(self):
        return "python {exe} -a -f {format} {fitfile} > {output}".format(
            exe=self.exe,
            format=self.format,
            fitfile=self.input().path,
            output=self.output().path,
        )
