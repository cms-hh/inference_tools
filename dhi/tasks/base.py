# coding: utf-8

from __future__ import absolute_import

__all__ = ["BaseTask", "AnalysisTask", "CMSSWSandboxTask", "CHBase"]


import os
from collections import OrderedDict

import luigi
import six
import law

law.contrib.load("matplotlib", "htcondor", "root", "tasks")


class BaseTask(law.Task):

    output_collection_cls = law.SiblingFileCollection
    default_store = "$DHI_STORE"

    def store_parts(self):
        parts = OrderedDict()
        parts["task_class"] = self.__class__.__name__
        return parts

    def local_path(self, *path, **kwargs):
        store = kwargs.get("store") or self.default_store
        store = os.path.expandvars(os.path.expanduser(store))
        parts = tuple(self.store_parts().values()) + path
        return os.path.join(store, *(str(p) for p in parts))

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
                self.cmd,
                shell=True,
                executable="/bin/bash",
                cwd=output_dir.path,
            )[0]
            if code != 0:
                raise Exception("{} failed with exit code {}".format(self.cmd, code))


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):
    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return law.LocalDirectoryTarget(self.local_path(store="$DHI_LOCAL_STORE"))

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return os.path.expandvars("$DHI_BASE/dhi/tasks/bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # render_variables are rendered into all files sent with a job
        config.render_variables["dhi_base"] = os.environ["DHI_BASE"]
        config.render_variables["dhi_env_path"] = os.environ["PATH"]
        # force to run on CC7, http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice
        config.custom_content.append(("requirements", '(OpSysAndVer =?= "CentOS7")'))
        # copy the entire environment
        config.custom_content.append(("getenv", "true"))
        # the CERN htcondor setup requires a "log" config, but we can safely set it to /dev/null
        # if you are interested in the logs of the batch system itself, set a meaningful value here
        config.custom_content.append(("log", "/dev/null"))
        # schedule for 1 day runtime
        config.custom_content.append(("+JobFlavour", "tomorrow"))
        return config

    def htcondor_use_local_scheduler(self):
        return True
