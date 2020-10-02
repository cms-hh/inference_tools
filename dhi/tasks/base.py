# coding: utf-8

"""
Generic base tasks.
"""

import os
import re
import math
from collections import OrderedDict

import luigi
import law

law.contrib.load("git", "htcondor", "matplotlib", "numpy", "root", "tasks", "wlcg")


class BaseTask(law.Task):

    print_command = law.CSVParameter(
        default=(),
        significant=False,
        description="print the command that this task would execute but do not run any task; this "
        "CSV parameter accepts a single integer value which sets the task recursion depth to also "
        "print the commands of required tasks (0 means non-recursive)",
    )

    interactive_params = law.Task.interactive_params + ["print_command"]
    task_namespace = os.getenv("DHI_TASK_NAMESPACE")

    def _print_command(self, args):
        max_depth = int(args[0])

        print("print task commands with max_depth {}".format(max_depth))

        for dep, _, depth in self.walk_deps(max_depth=max_depth, order="pre"):
            offset = depth * ("|" + law.task.interactive.ind)
            print(offset)

            print("{}> {}".format(offset, dep.repr(color=True)))
            offset += "|" + law.task.interactive.ind

            if isinstance(dep, CommandTask):
                # when dep is a workflow, take the first branch
                text = law.util.colored("command", style="bright")
                if isinstance(dep, law.BaseWorkflow) and dep.is_workflow():
                    dep = dep.as_branch(0)
                    text += " (from branch {})".format(law.util.colored("0", "red"))
                text += ": "

                cmd = dep.build_command()
                if cmd:
                    cmd = law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd
                    text += law.util.colored(cmd, "cyan")
                else:
                    text += law.util.colored("empty", "red")
                print(offset + text)
            else:
                print(offset + law.util.colored("not a CommandTask", "yellow"))


class AnalysisTask(BaseTask):

    version = luigi.Parameter(description="task version")

    output_collection_cls = law.SiblingFileCollection
    default_store = "$DHI_STORE"

    def store_parts(self):
        parts = OrderedDict()
        parts["task_class"] = self.__class__.__name__
        return parts

    def store_parts_ext(self):
        parts = OrderedDict()
        if self.version is not None:
            parts["version"] = self.version
        return parts

    def local_path(self, *path, **kwargs):
        store = kwargs.get("store") or self.default_store
        store = os.path.expandvars(os.path.expanduser(store))

        parts = tuple(self.store_parts().values()) + tuple(self.store_parts_ext().values()) + path

        return os.path.join(store, *(str(p) for p in parts))

    def local_target(self, *path, **kwargs):
        cls = law.LocalFileTarget if not kwargs.pop("dir", False) else law.LocalDirectoryTarget
        store = kwargs.pop("store", None)

        return cls(self.local_path(*path, store=store), **kwargs)


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):

    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory, default: True",
    )
    max_runtime = law.DurationParameter(
        default=2.0,
        unit="h",
        significant=False,
        description="maximum runtime, default unit is hours, default: 2",
    )
    htcondor_flavor = luigi.ChoiceParameter(
        default="cern",
        choices=("cern",),
        significant=False,
        description="the 'flavor' (i.e. configuration name) of the batch system, choices: cern",
    )
    htcondor_getenv = luigi.BoolParameter(
        default=False,
        significant=False,
        description="whether to use htcondor's getenv feature to set the job enrivonment to the "
        "current one, instead of using bundled versions of the repository and software, "
        "default: False",
    )

    def htcondor_workflow_requires(self):
        reqs = law.htcondor.HTCondorWorkflow.htcondor_workflow_requires(self)

        # add repo and software bundling as requirements when getenv is not requested
        if not self.htcondor_getenv:
            reqs["repo"] = BundleRepo.req(self, replicas=1)
            reqs["software"] = BundleSoftware.req(self, replicas=1)

        return reqs

    def htcondor_output_directory(self):
        # the directory where submission meta data should be stored
        return self.local_target(store="$DHI_STORE_REPO", dir=True)

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        return os.path.expandvars("$DHI_BASE/dhi/tasks/remote_bootstrap.sh")

    def htcondor_job_config(self, config, job_num, branches):
        # use cc7 at CERN (http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice)
        if self.htcondor_flavor == "cern":
            config.custom_content.append(("requirements", '(OpSysAndVer =?= "CentOS7")'))
        # copy the entire environment when requests
        if self.htcondor_getenv:
            config.custom_content.append(("getenv", "true"))
        # the CERN htcondor setup requires a "log" config, but we can safely set it to /dev/null
        # if you are interested in the logs of the batch system itself, set a meaningful value here
        config.custom_content.append(("log", "/dev/null"))
        # max runtime
        config.custom_content.append(("+MaxRuntime", int(math.floor(self.max_runtime * 3600)) - 1))

        # render_variables are rendered into all files sent with a job
        if self.htcondor_getenv:
            config.render_variables["dhi_bootstrap_name"] = "htcondor_getenv"
        else:
            reqs = self.htcondor_workflow_requires()
            config.render_variables["dhi_bootstrap_name"] = "htcondor_standalone"
            config.render_variables["dhi_software_pattern"] = reqs["software"].get_file_pattern()
            config.render_variables["dhi_repo_pattern"] = reqs["repo"].get_file_pattern()
        config.render_variables["dhi_env_path"] = os.environ["PATH"]
        config.render_variables["dhi_env_pythonpath"] = os.environ["PYTHONPATH"]
        config.render_variables["dhi_htcondor_flavor"] = self.htcondor_flavor
        config.render_variables["dhi_base"] = os.environ["DHI_BASE"]
        config.render_variables["dhi_user"] = os.environ["DHI_USER"]
        config.render_variables["dhi_store"] = os.environ["DHI_STORE"]
        config.render_variables["dhi_task_namespace"] = os.environ["DHI_TASK_NAMESPACE"]
        config.render_variables["dhi_local_scheduler"] = os.environ["DHI_LOCAL_SCHEDULER"]

        return config

    def htcondor_use_local_scheduler(self):
        # remote jobs should not communicate with ther central scheduler but with a local one
        return True


class BundleRepo(AnalysisTask, law.git.BundleGitRepository, law.tasks.TransferLocalFile):

    replicas = luigi.IntParameter(
        default=10,
        description="number of replicas to generate, default: 10",
    )

    exclude_files = ["docs", "githooks", ".data", ".law", ".setups"]

    version = None
    task_namespace = None
    default_store = "$DHI_STORE_BUNDLES"

    def get_repo_path(self):
        # required by BundleGitRepository
        return os.environ["DHI_BASE"]

    def single_output(self):
        repo_base = os.path.basename(self.get_repo_path())
        return self.local_target("{}.{}.tgz".format(repo_base, self.checksum))

    def get_file_pattern(self):
        path = os.path.expandvars(os.path.expanduser(self.single_output().path))
        return self.get_replicated_path(path, i=None if self.replicas <= 0 else "*")

    def output(self):
        return law.tasks.TransferLocalFile.output(self)

    @law.decorator.safe_output
    def run(self):
        # create the bundle
        bundle = law.LocalFileTarget(is_tmp="tgz")
        self.bundle(bundle)

        # log the size
        self.publish_message(
            "bundled repository archive, size is {:.2f} {}".format(
                *law.util.human_bytes(bundle.stat.st_size)
            )
        )

        # transfer the bundle
        self.transfer(bundle)


class BundleSoftware(AnalysisTask, law.tasks.TransferLocalFile, law.tasks.RunOnceTask):

    replicas = luigi.IntParameter(
        default=10,
        description="number of replicas to generate, default: 10",
    )
    force_upload = luigi.BoolParameter(default=False, description="force uploading")

    version = None
    default_store = "$DHI_STORE_BUNDLES"

    def complete(self):
        if self.force_upload and not self.has_run:
            return False
        else:
            return AnalysisTask.complete(self)

    def single_output(self):
        return self.local_target("software.tgz")

    def get_file_pattern(self):
        path = os.path.expandvars(os.path.expanduser(self.single_output().path))
        return self.get_replicated_path(path, i=None if self.replicas <= 0 else "*")

    @law.decorator.safe_output
    def run(self):
        software_path = os.environ["DHI_SOFTWARE"]

        # create the local bundle
        bundle = law.LocalFileTarget(software_path + ".tgz", is_tmp=True)

        def _filter(tarinfo):
            if re.search(r"(\.pyc|\/\.git|\.tgz|__pycache__|black)$", tarinfo.name):
                return None
            return tarinfo

        # create the archive with a custom filter
        bundle.dump(software_path, filter=_filter)

        # log the size
        self.publish_message(
            "bundled software archive, size is {:.2f} {}".format(
                *law.util.human_bytes(bundle.stat.st_size)
            )
        )

        # transfer the bundle and mark the task as complete
        self.transfer(bundle)
        self.mark_complete()


class CommandTask(AnalysisTask):
    """
    A task that provides convenience methods to work with shell commands, i.e., printing them on the
    command line and executing them with error handling.
    """

    exclude_index = True

    run_command_in_tmp = False

    def build_command(self):
        # this method should build and return the command to run
        raise NotImplementedError

    def run_command(self, cmd, optional=False, **kwargs):
        # proper command encoding
        cmd = law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path

        # call it
        with self.publish_step("running '{}' ...".format(law.util.colored(cmd, "cyan"))):
            code, out, err = law.util.interruptable_popen(
                cmd, shell=True, executable="/bin/bash", **kwargs
            )

        # raise an exception when the call failed and optional is not True
        if code != 0 and not optional:
            msg = "command failed with exit code {}: {}".format(code, cmd)
            if out:
                msg += "\noutput: {}".format(out)
            if err:
                msg += "\nerror: {}".format(err)
            raise Exception(msg)

        return code, out, err

    def touch_output_dirs(self):
        # keep track of created uris so we can avoid creating them twice
        handled_parent_uris = set()

        for outp in law.util.flatten(self.output()):
            # get the parent directory target
            parent = None
            if isinstance(outp, law.SiblingFileCollection):
                parent = outp.dir
            elif isinstance(outp, law.FileSystemFileTarget):
                parent = outp.parent

            # create it
            if parent and parent.uri() not in handled_parent_uris:
                parent.touch()
                handled_parent_uris.add(parent.uri())

    @law.decorator.safe_output
    def run(self, **kwargs):
        # default run implementation
        # first, create all output directories
        self.touch_output_dirs()

        # build the command
        cmd = self.build_command()

        # run it
        self.run_command(cmd, **kwargs)


class CombineCommandTask(CommandTask):

    mass = luigi.FloatParameter(
        default=125.0,
        description="mass of the underlying resonance, default: 125.",
    )

    combine_stable_options = " ".join(
        [
            "--cminDefaultMinimizerType Minuit2",
            "--cminDefaultMinimizerStrategy 0",
            "--cminFallbackAlgo Minuit2,0:1.0",
        ]
    )

    def store_parts(self):
        parts = super(CombineCommandTask, self).store_parts()
        parts["mass"] = "m{}".format(self.mass)
        return parts
