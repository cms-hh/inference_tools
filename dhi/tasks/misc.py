# coding: utf-8

import os
import glob
import itertools

import law
import luigi
import luigi.util

from dhi.tasks.base import CHBase
from dhi.utils.util import *

import CombineHarvester.CombineTools.ch as ch


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
