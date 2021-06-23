# coding: utf-8

"""
Generic base tasks.
"""

import os
import re
import math
import importlib
from collections import OrderedDict

import luigi
import law
import six


law.contrib.load("git", "htcondor", "matplotlib", "numpy", "slack", "telegram", "root", "tasks")


class LocalTarget(law.LocalTarget):
    def __init__(self, *args, **kwargs):
        self._repr_path = kwargs.pop("repr_path", None)
        super(LocalTarget, self).__init__(*args, **kwargs)

    def _repr_pairs(self, *args, **kwargs):
        pairs = super(LocalTarget, self)._repr_pairs(*args, **kwargs)
        if self._repr_path:
            pairs = [(key, self._repr_path if key == "path" else value) for key, value in pairs]
        return pairs

    def _parent_args(self):
        args, kwargs = super(LocalTarget, self)._parent_args()
        if self._repr_path:
            kwargs["repr_path"] = os.path.dirname(self._repr_path)
        return args, kwargs


class LocalFileTarget(LocalTarget, law.LocalFileTarget):
    pass


class LocalDirectoryTarget(LocalTarget, law.LocalDirectoryTarget):

    def _child_args(self, path):
        args, kwargs = law.LocalDirectoryTarget._child_args(self, path)
        if self._repr_path:
            basename = os.path.relpath(path, self.path)
            if os.sep not in basename:
                kwargs["repr_path"] = os.path.join(self._repr_path, basename)
        return args, kwargs


LocalTarget.file_class = LocalFileTarget
LocalTarget.directory_class = LocalDirectoryTarget


class SiblingFileCollection(law.SiblingFileCollection):

    def _repr_pairs(self, *args, **kwargs):
        pairs = super(SiblingFileCollection, self)._repr_pairs(*args, **kwargs)
        if pairs[-1][0] == "dir" and getattr(self.dir, "_repr_path", None):
            pairs[-1] = ("dir", self.dir._repr_path)
        return pairs


class BaseTask(law.Task):

    print_command = law.CSVParameter(
        default=(),
        significant=False,
        description="print the command that this task would execute but do not run any task; this "
        "CSV parameter accepts a single integer value which sets the task recursion depth to also "
        "print the commands of required tasks (0 means non-recursive)",
    )
    notify_slack = law.slack.NotifySlackParameter(significant=False)
    notify_telegram = law.telegram.NotifyTelegramParameter(significant=False)

    exclude_params_req = {"notify_slack", "notify_telegram"}

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

    version = luigi.Parameter(description="mandatory version that is encoded into output paths")

    output_collection_cls = SiblingFileCollection
    default_store = "$DHI_STORE"
    store_by_family = False

    @classmethod
    def modify_param_values(cls, params):
        return params

    @classmethod
    def req_params(cls, inst, **kwargs):
        # always prefer certain parameters given as task family parameters (--TaskFamily-parameter)
        _prefer_cli = law.util.make_list(kwargs.get("_prefer_cli", []))
        if "version" not in _prefer_cli:
            _prefer_cli.append("version")
        kwargs["_prefer_cli"] = _prefer_cli

        return super(AnalysisTask, cls).req_params(inst, **kwargs)

    def store_parts(self):
        parts = OrderedDict()
        parts["task_class"] = self.task_family if self.store_by_family else self.__class__.__name__
        return parts

    def store_parts_ext(self):
        parts = OrderedDict()
        if self.version is not None:
            parts["version"] = self.version
        return parts

    def local_path(self, *path, **kwargs):
        store = kwargs.get("store") or self.default_store
        parts = tuple(self.store_parts().values()) + tuple(self.store_parts_ext().values()) + path
        repr_path = os.path.join(store, *(str(p) for p in parts))
        local_path = os.path.expandvars(os.path.expanduser(repr_path))
        return (local_path, repr_path) if kwargs.get("repr_path") else local_path

    def local_target(self, *path, **kwargs):
        cls = LocalFileTarget if not kwargs.pop("dir", False) else LocalDirectoryTarget
        store = kwargs.pop("store", None)
        local_path, repr_path = self.local_path(*path, store=store, repr_path=True)
        return cls(local_path, repr_path=repr_path, **kwargs)

    def join_postfix(self, parts, sep1="__", sep2="_"):
        repl = lambda s: re.sub(r"[^a-zA-Z0-9\.\_\-\+]", "", str(s))
        return sep1.join(
            (sep2.join(repl(p) for p in part) if isinstance(part, (list, tuple)) else repl(part))
            for part in parts
            if (isinstance(part, int) or part)
        )


class HTCondorWorkflow(law.htcondor.HTCondorWorkflow):

    transfer_logs = luigi.BoolParameter(
        default=True,
        significant=False,
        description="transfer job logs to the output directory; default: True",
    )
    max_runtime = law.DurationParameter(
        default=2.0,
        unit="h",
        significant=False,
        description="maximum runtime; default unit is hours; default: 2",
    )
    htcondor_cpus = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="number of CPUs to request; empty value leads to the cluster default setting; "
        "no default",
    )
    htcondor_flavor = luigi.ChoiceParameter(
        default=os.getenv("DHI_HTCONDOR_FLAVOR", "cern"),
        choices=("cern",),
        significant=False,
        description="the 'flavor' (i.e. configuration name) of the batch system; choices: cern; "
        "default: {}".format(os.getenv("DHI_HTCONDOR_FLAVOR", "cern")),
    )
    htcondor_getenv = luigi.BoolParameter(
        default=False,
        significant=False,
        description="whether to use htcondor's getenv feature to set the job enrivonment to the "
        "current one, instead of using repository and software bundling; default: False",
    )
    htcondor_group = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="the name of an accounting group on the cluster to handle user priority; not "
        "used when empty; no default",
    )

    exclude_params_branch = {
        "max_runtime", "htcondor_cpus", "htcondor_flavor", "htcondor_getenv", "htcondor_group",
    }

    def htcondor_workflow_requires(self):
        reqs = law.htcondor.HTCondorWorkflow.htcondor_workflow_requires(self)

        # add repo and software bundling as requirements when getenv is not requested
        if not self.htcondor_getenv:
            reqs["repo"] = BundleRepo.req(self, replicas=3)
            reqs["software"] = BundleSoftware.req(self, replicas=3)

        return reqs

    def htcondor_output_directory(self):
        # the directory where submission meta data and logs should be stored
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

        # request cpus
        if self.htcondor_cpus > 0:
            config.custom_content.append(("RequestCpus", self.htcondor_cpus))

        # accounting group for priority on the cluster
        if self.htcondor_group and self.htcondor_group != law.NO_STR:
            config.custom_content.append(("+AccountingGroup", self.htcondor_group))

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
        description="number of replicas to generate; default: 10",
    )

    exclude_files = ["docs", "data", ".law", ".setups", "datacards_run2/*"]

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


class BundleSoftware(AnalysisTask, law.tasks.TransferLocalFile):

    replicas = luigi.IntParameter(
        default=10,
        description="number of replicas to generate; default: 10",
    )

    version = None
    default_store = "$DHI_STORE_BUNDLES"

    def __init__(self, *args, **kwargs):
        super(BundleSoftware, self).__init__(*args, **kwargs)

        self._checksum = None

    @property
    def checksum(self):
        if not self._checksum:
            # read content of all software flag files and create a hash
            contents = []
            for flag_file in os.environ["DHI_SOFTWARE_FLAG_FILES"].strip().split():
                if os.path.exists(flag_file):
                    with open(flag_file, "r") as f:
                        contents.append((flag_file, f.read().strip()))
            self._checksum = law.util.create_hash(contents)

        return self._checksum

    def single_output(self):
        return self.local_target("software.{}.tgz".format(self.checksum))

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

        # transfer the bundle
        self.transfer(bundle)


class CommandTask(AnalysisTask):
    """
    A task that provides convenience methods to work with shell commands, i.e., printing them on the
    command line and executing them with error handling.
    """

    custom_args = luigi.Parameter(
        default="",
        description="custom arguments that are forwarded to the underlying command; they might not "
        "be encoded into output file paths; no default",
    )

    exclude_index = True
    exclude_params_req = {"custom_args"}

    run_command_in_tmp = False

    def build_command(self):
        # this method should build and return the command to run
        raise NotImplementedError

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

    def run_command(self, cmd, optional=False, **kwargs):
        # proper command encoding
        cmd = (law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd).strip()

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path
        self.publish_message("cwd: {}".format(kwargs.get("cwd", os.getcwd())))

        # call it
        with self.publish_step("running '{}' ...".format(law.util.colored(cmd, "cyan"))):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            for line in lines:
                print(line)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            raise Exception("command failed with exit code {}: {}".format(p.returncode, cmd))

        return p

    @law.decorator.log
    @law.decorator.notify
    @law.decorator.safe_output
    def run(self, **kwargs):
        self.pre_run_command()

        # default run implementation
        # first, create all output directories
        self.touch_output_dirs()

        # build the command
        cmd = self.build_command()

        # run it
        self.run_command(cmd, **kwargs)

        self.post_run_command()

    def pre_run_command(self):
        return

    def post_run_command(self):
        return


class PlotTask(AnalysisTask):

    file_types = law.CSVParameter(
        default=("pdf",),
        description="comma-separated types of the output plot files; default: pdf",
    )
    plot_postfix = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="an arbitrary postfix that is added with to underscores to all paths of "
        "produced plots; no default",
    )
    view_cmd = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a command to execute after the task has run to visualize plots right in the "
        "terminal; no default",
    )
    campaign = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="the campaign name used for plotting; no default",
    )
    x_min = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the lower x-axis limit; no default",
    )
    x_max = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the upper x-axis limit; no default",
    )
    y_min = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the lower y-axis limit; no default",
    )
    y_max = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the upper y-axis limit; no default",
    )
    z_min = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the lower z-axis limit; no default",
    )
    z_max = luigi.FloatParameter(
        default=-1000.0,
        significant=False,
        description="the upper z-axis limit; no default",
    )

    def get_axis_limit(self, value):
        if isinstance(value, six.string_types):
            value = getattr(self, value)
        return None if value == -1000.0 else value

    def create_plot_names(self, parts):
        if self.plot_postfix and self.plot_postfix != law.NO_STR:
            parts.append((self.plot_postfix,))

        return ["{}.{}".format(self.join_postfix(parts), ext) for ext in self.file_types]

    def get_plot_func(self, func_id):
        if "." not in func_id:
            raise ValueError("invalid func_id format: {}".format(func_id))
        module_id, name = func_id.rsplit(".", 1)

        try:
            mod = importlib.import_module(module_id)
        except ImportError as e:
            raise ImportError(
                "cannot import plot function {} from module {}: {}".format(name, module_id, e)
            )

        func = getattr(mod, name, None)
        if func is None:
            raise Exception("module {} does not contain plot function {}".format(module_id, name))

        return func

    def call_plot_func(self, func_id, **kwargs):
        self.get_plot_func(func_id)(**(self.update_plot_kwargs(kwargs)))

    def update_plot_kwargs(self, kwargs):
        return kwargs


class ModelParameters(luigi.Parameter):

    def __init__(self, *args, **kwargs):
        self._unique = kwargs.pop("unique", False)
        self._sort = kwargs.pop("sort", False)
        self._min_len = kwargs.pop("min_len", None)
        self._max_len = kwargs.pop("max_len", None)

        # ensure that the default value is a tuple
        if "default" in kwargs:
            kwargs["default"] = law.util.make_tuple(kwargs["default"])

        super(ModelParameters, self).__init__(*args, **kwargs)

    def _make_unique(self, value):
        if not self._unique:
            return value

        # keep only the first occurence of a parameter, identified by name
        _value = []
        names = set()
        for v in value:
            if v[0] not in names:
                value.append(v)
                names.add(v[0])

        return tuple(_value)

    def _sort_by_name(self, value):
        if not self._sort:
            return value

        _value = sorted(value, key=lambda v: v[0])

        return tuple(_value)

    def _check_len(self, value):
        s = lambda v: str(v[0]) if len(v) == 1 else "{}={}".format(v[0], ",".join(map(str, v[1:])))

        for v in value:
            if self._min_len is not None and len(v) - 1 < self._min_len:
                raise ValueError("the parameter '{}' contains {} value(s), but a minimum of {} is "
                    "required".format(s(v), len(v) - 1, self._min_len))

            if self._max_len is not None and len(v) - 1 > self._max_len:
                raise ValueError("the parameter '{}' contains {} value(s), but a maximum of {} is "
                    "required".format(s(v), len(v) - 1, self._max_len))

    def parse(self, inp):
        if not inp or inp == law.NO_STR:
            value = tuple()
        elif isinstance(inp, (tuple, list)) or law.util.is_lazy_iterable(inp):
            value = law.util.make_tuple(inp)
        elif isinstance(inp, six.string_types):
            value = []
            for s in inp.split(":"):
                v = []
                if "=" in s:
                    name, s = s.split("=", 1)
                    v.append(name)
                v.extend(s.split(","))
                value.append(tuple(v))
            value = tuple(value)
        else:
            value = (inp,)

        # apply uniqueness, sort, length and choices checks
        value = self._make_unique(value)
        value = self._sort_by_name(value)
        self._check_len(value)

        return value

    def serialize(self, value):
        if not value:
            value = tuple()

        value = law.util.make_tuple(value)

        # apply uniqueness, sort, length and choices checks
        value = self._make_unique(value)
        value = self._sort_by_name(value)
        self._check_len(value)

        return ":".join(
            str(v[0]) if len(v) == 1 else ("{}={}".format(v[0], ",".join(map(str, v[1:]))))
            for v in value
        )


@law.decorator.factory(accept_generator=True)
def view_output_plots(fn, opts, task, *args, **kwargs):
    def before_call():
        return None

    def call(state):
        return fn(task, *args, **kwargs)

    def after_call(state):
        view_cmd = getattr(task, "view_cmd", None)
        if not view_cmd or view_cmd == law.NO_STR:
            return

        # prepare the view command
        if "{}" not in view_cmd:
            view_cmd += " {}"

        # collect all paths to view
        view_paths = []
        outputs = law.util.flatten(task.output())
        while outputs:
            output = outputs.pop(0)
            if isinstance(output, law.TargetCollection):
                outputs.extend(output._flat_target_list)
                continue
            if not getattr(output, "path", None):
                continue
            if output.path.endswith((".pdf", ".png")) and output.path not in view_paths:
                view_paths.append(output.path)

        # loop through paths and view them
        for path in view_paths:
            task.publish_message("showing {}".format(path))
            law.util.interruptable_popen(view_cmd.format(path), shell=True, executable="/bin/bash")

    return before_call, call, after_call
