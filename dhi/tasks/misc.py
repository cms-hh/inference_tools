# coding: utf-8

import law
import luigi

from dhi.tasks.base import CHBase


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
