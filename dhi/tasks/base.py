# coding: utf-8

from __future__ import absolute_import

__all__ = ["BaseTask", "AnalysisTask", "CMSSWSandboxTask", "CHBase"]


import os
from collections import OrderedDict

import luigi
import six
import law

law.contrib.load("matplotlib")


class BaseTask(law.Task):

    output_collection_cls = law.SiblingFileCollection

    def store_parts(self):
        parts = OrderedDict()
        parts["task_class"] = self.__class__.__name__
        return parts

    def local_path(self, *path):
        parts = tuple(self.store_parts().values()) + path
        return os.path.join(os.environ["DHI_STORE"], *(str(p) for p in parts))

    def local_target(self, *path, **kwargs):
        cls = law.LocalFileTarget if not kwargs.pop("dir", False) else law.LocalDirectoryTarget
        return cls(self.local_path(*path), **kwargs)


class AnalysisTask(BaseTask):

    version = luigi.Parameter(description="task version")

    task_namespace = "dhi"

    def store_parts(self):
        parts = super(AnalysisTask, self).store_parts()
        if self.version is not None:
            parts["version"] = self.version
        return parts

    @classmethod
    def modify_param_values(cls, params):
        return params


class CHBase(AnalysisTask):

    mass = luigi.IntParameter(default=125)

    stable_options = r"--cminDefaultMinimizerType Minuit2 --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,0:1.0"

    exclude_index = True

    def store_parts(self):
        parts = super(CHBase, self).store_parts()
        parts["mass"] = self.mass
        return parts

    @property
    def cmd(self):
        raise NotImplementedError

    @law.decorator.safe_output
    def run(self):
        # find the directory of the first registered output
        output_dir = law.util.flatten(self.output())[0].parent
        output_dir.touch()

        with self.publish_step(law.util.colored(self.cmd, color="light_cyan")):
            code = law.util.interruptable_popen(
                self.cmd, shell=True, executable="/bin/bash", cwd=output_dir.path,
            )[0]
            if code != 0:
                raise Exception("{} failed with exit code {}".format(self.cmd, code))
