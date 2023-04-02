# coding: utf-8

"""
Generic base tasks.
"""

import os
import sys
import re
import enum
import importlib
import getpass
from collections import OrderedDict

import luigi
import law
import six

from dhi import dhi_remote_job, dhi_has_gfal
from dhi.util import call_hook
from dhi.config import cms_postfix as default_cms_postfix


logger = law.logger.get_logger(__name__)


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

    def _print_command(self, args):
        from law.task.interactive import fmt_chars, _print_wrapped
        from law.util import colored, get_terminal_width

        max_depth = int(args[0])

        print("print task commands with max_depth {}".format(max_depth))
        print("")

        # get the format chars
        fmt_name = law.config.get_expanded("task", "interactive_format")
        fmt = fmt_chars.get(fmt_name, fmt_chars["fancy"])

        # get the line break setting
        break_lines = law.config.get_expanded_bool("task", "interactive_line_breaks")
        out_width = law.config.get_expanded_int("task", "interactive_line_width")
        print_width = (out_width if out_width > 0 else get_terminal_width()) if break_lines else None
        _print = lambda line, offset, br=1: _print_wrapped(line, print_width if br else None, offset)

        # walk through deps
        parents_last_flags = []
        for dep, next_deps, depth, is_last in self.walk_deps(
            max_depth=max_depth,
            order="pre",
            yield_last_flag=True,
        ):
            del parents_last_flags[depth:]
            next_deps_shown = bool(next_deps) and (max_depth < 0 or depth < max_depth)

            # determine the print common offset
            offset = [(" " if f else fmt["|"]) + fmt["ind"] * " " for f in parents_last_flags[1:]]
            offset = "".join(offset)
            parents_last_flags.append(is_last)

            # print free space
            free_offset = offset + fmt["|"]
            free_lines = "\n".join(fmt["free"] * [free_offset])
            if depth > 0 and free_lines:
                print(free_lines)

            # determine task offset and prefix
            task_offset = offset
            if depth > 0:
                task_offset += fmt["l" if is_last else "t"] + fmt["ind"] * fmt["-"]
            task_prefix = "{} {} ".format(depth, fmt[">"])

            # determine text offset and prefix
            text_offset = offset
            if depth > 0:
                text_offset += (" " if is_last else fmt["|"]) + fmt["ind"] * " "
            text_prefix = (len(task_prefix) - 1) * " "
            text_offset += (fmt["|"] if next_deps_shown else " ") + text_prefix
            text_offset_ind = text_offset + fmt["ind"] * " "

            # print the task line
            _print(task_offset + task_prefix + dep.repr(color=True), text_offset)

            # stop when dep has no command
            if not isinstance(dep, CommandTask):
                _print(text_offset_ind + colored("not a CommandTask", "yellow"), text_offset_ind)
                continue

            # when dep is a workflow, take the first branch
            text = law.util.colored("command", style="bright")
            if isinstance(dep, law.BaseWorkflow) and dep.is_workflow():
                dep = dep.as_branch(0)
                text += " (from branch {})".format(law.util.colored("0", "red"))
            text += ": "

            cmd = dep.get_command(fallback_level=0)
            if cmd:
                # when cmd is a 2-tuple, i.e. the real command and a representation for printing
                # pick the second one
                if isinstance(cmd, tuple) and len(cmd) == 2:
                    cmd = cmd[1]
                else:
                    if isinstance(cmd, list):
                        cmd = law.util.quote_cmd(cmd)
                    # defaut highlighting
                    cmd = law.util.colored(cmd, "cyan")
                text += cmd
            else:
                text += law.util.colored("empty", "red")
            _print(text_offset_ind + text, text_offset_ind, br=False)

    def _repr_params(self, *args, **kwargs):
        params = super(BaseTask, self)._repr_params(*args, **kwargs)

        # remove empty params by default
        for key, value in list(params.items()):
            if not value and value != 0:
                del params[key]

        return params


class OutputLocation(enum.Enum):
    """
    Output location flag.
    """

    config = "config"
    local = "local"
    wlcg = "wlcg"


class AnalysisTask(BaseTask):

    version = luigi.Parameter(
        description="mandatory version that is encoded into output paths",
    )

    # defaults for targets
    output_collection_cls = law.SiblingFileCollection
    default_store = "$DHI_STORE"
    default_wlcg_fs = law.config.get_expanded("target", "default_wlcg_fs")
    default_output_location = "config"
    store_by_family = False

    @classmethod
    def modify_param_values(cls, params):
        return params

    @classmethod
    def req_params(cls, inst, **kwargs):
        # always prefer certain parameters given as task family parameters (--TaskFamily-parameter)
        _prefer_cli = law.util.make_list(kwargs.get("_prefer_cli", []))
        if "version" not in _prefer_cli:
            _prefer_cli += [
                "version", "workflow", "job_workers", "poll_interval", "walltime", "max_runtime",
                "retries", "acceptance", "tolerance", "parallel_jobs", "shuffle_jobs",
            ]
        kwargs["_prefer_cli"] = set(_prefer_cli) | cls.prefer_params_cli

        return super(AnalysisTask, cls).req_params(inst, **kwargs)

    def __init__(self, *args, **kwargs):
        super(AnalysisTask, self).__init__(*args, **kwargs)

        # generic hook to change task parameters in a customizable way
        self.call_hook("init_analysis_task")

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
        """ local_path(*path, store=None, fs=None)
        """
        # if no fs is set, determine the main store directory
        parts = ()
        if not kwargs.pop("fs", None):
            store = kwargs.get("store") or self.default_store
            parts += (store,)

        parts += tuple(self.store_parts().values())
        parts += tuple(self.store_parts_ext().values())
        parts += path

        return os.path.join(*map(str, parts))

    def local_target(self, *path, **kwargs):
        """ local_target(*path, dir=False, store=None, fs=None, **kwargs)
        """
        _dir = kwargs.pop("dir", False)
        store = kwargs.pop("store", None)
        fs = kwargs.get("fs", None)

        # select the target class
        cls = law.LocalDirectoryTarget if _dir else law.LocalFileTarget

        # create the local path
        path = self.local_path(*path, store=store, fs=fs)

        # create the target instance and return it
        return cls(path, **kwargs)

    def wlcg_path(self, *path):
        # concatenate all parts that make up the path and join them
        parts = ()
        parts += tuple(self.store_parts().values())
        parts += tuple(self.store_parts_ext().values())
        parts += path

        return os.path.join(*map(str, parts))

    def wlcg_target(self, *path, **kwargs):
        """ wlcg_target(*path, dir=False, fs=default_wlcg_fs, **kwargs)
        """
        _dir = kwargs.pop("dir", False)
        if not kwargs.get("fs"):
            kwargs["fs"] = self.default_wlcg_fs

        # select the target class
        cls = law.wlcg.WLCGDirectoryTarget if _dir else law.wlcg.WLCGFileTarget

        # create the local path
        path = self.wlcg_path(*path)

        # create the target instance and return it
        return cls(path, **kwargs)

    def target(self, *path, **kwargs):
        """ target(*path, location=None, **kwargs)
        """
        # get the default location
        location = kwargs.pop("location", self.default_output_location)

        # parse it and obtain config values if necessary
        if isinstance(location, str):
            location = OutputLocation[location]
        if location == OutputLocation.config:
            location = law.config.get_expanded("outputs", self.task_family, split_csv=True)
            if not location:
                self.logger.debug(
                    "no option 'outputs::{}' found in law.cfg to obtain target "
                    "location, falling back to 'local'".format(self.task_family),
                )
                location = ["local"]
            location[0] = OutputLocation[location[0]]
        location = law.util.make_list(location)

        # forward to correct function
        if location[0] == OutputLocation.wlcg:
            # check if gfal exists
            if not dhi_has_gfal:
                logger.warning_once(
                    "gfal2_missing_for_wlcg_target",
                    "cannot create wlcg target, gfal2 is missing",
                )
                location[0] = OutputLocation.local
            else:
                # get other options
                (fs,) = (location[1:] + [None])[:1]
                kwargs.setdefault("fs", fs)
                return self.wlcg_target(*path, **kwargs)

        if location[0] == OutputLocation.local:
            # get other options
            (loc,) = (location[1:] + [None])[:1]
            loc_key = "fs" if (loc and law.config.has_section(loc)) else "store"
            kwargs.setdefault(loc_key, loc)
            return self.local_target(*path, **kwargs)

        raise Exception("cannot determine output location based on '{}'".format(location))

    def join_postfix(self, parts, sep1="__", sep2="_"):
        def repl(s):
            # replace certain characters
            s = str(s).replace("*", "X").replace("?", "Y")
            # remove remaining unknown characters
            s = re.sub(r"[^a-zA-Z0-9\.\_\-\+]", "", s)
            return s

        return sep1.join(
            (sep2.join(repl(p) for p in part) if isinstance(part, (list, tuple)) else repl(part))
            for part in parts
            if (isinstance(part, int) or part)
        )

    def get_output_postfix(self, join=True):
        parts = []
        return self.join_postfix(parts) if join else parts

    def call_hook(self, name, **kwargs):
        return call_hook(name, self, **kwargs)


class UserTask(AnalysisTask):

    user = luigi.Parameter(
        default=getpass.getuser(),
    )

    exclude_params_index = {"user"}
    exclude_params_req = {"user"}


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
    test_timing = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when set, a log file is created along the outputs containing timing and "
        "memory usage information; default: False",
    )

    exclude_index = True
    exclude_params_req = {"custom_args"}

    # by default, do not run in a tmp dir
    run_command_in_tmp = False

    # by default, do not cleanup tmp dirs on error, except when running as a remote job
    cleanup_tmp_on_error = dhi_remote_job

    def build_command(self, fallback_level):
        # this method should build and return the command to run
        raise NotImplementedError

    def get_command(self, *args, **kwargs):
        # this method is returning the actual, possibly cleaned command
        return self.build_command(*args, **kwargs)

    def allow_fallback_command(self, errors):
        # by default, never allow fallbacks
        return False

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

    def run_command(self, cmd, highlighted_cmd=None, optional=False, **kwargs):
        # proper command encoding
        cmd = (law.util.quote_cmd(cmd) if isinstance(cmd, (list, tuple)) else cmd).strip()

        # prepend timing instructions
        if self.test_timing:
            log_file = self.join_postfix(["timing", self.get_output_postfix()]) + ".log"
            timing_cmd = (
                "/usr/bin/time "
                "-ao {log_file} "
                "-f \"TIME='Elapsed %e User %U Sys %S Mem %M' time\""
            ).format(
                log_file=self.local_target(log_file).path,
            )
            cmd = "{} {}".format(timing_cmd, cmd)
            print("adding timing command")

        # default highlighted command
        if not highlighted_cmd:
            highlighted_cmd = law.util.colored(cmd, "cyan")

        # when no cwd was set and run_command_in_tmp is True, create a tmp dir
        tmp_dir = None
        if "cwd" not in kwargs and self.run_command_in_tmp:
            tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
            tmp_dir.touch()
            kwargs["cwd"] = tmp_dir.path
        self.publish_message("cwd: {}".format(kwargs.get("cwd", os.getcwd())))

        # log the timing command
        if self.test_timing:
            self.publish_message("timing: {}".format(timing_cmd))

        # call it
        with self.publish_step("running '{}' ...".format(highlighted_cmd)):
            p, lines = law.util.readable_popen(cmd, shell=True, executable="/bin/bash", **kwargs)
            for line in lines:
                print(line)

        # raise an exception when the call failed and optional is not True
        if p.returncode != 0 and not optional:
            # when requested, make the tmp_dir non-temporary to allow for checks later on
            if tmp_dir and not self.cleanup_tmp_on_error:
                tmp_dir.is_tmp = False

            # raise exception
            raise CommandException(cmd, p.returncode, kwargs.get("cwd"))

        return p

    @law.decorator.log
    @law.decorator.notify
    @law.decorator.localize
    def run(self, **kwargs):
        self.pre_run_command()

        # default run implementation
        # first, create all output directories
        self.touch_output_dirs()

        # start command building and execution in a fallback loop
        errors = []
        while True:
            # get the command
            cmd = self.get_command(fallback_level=len(errors))
            if isinstance(cmd, tuple) and len(cmd) == 2:
                kwargs["highlighted_cmd"] = cmd[1]
                cmd = cmd[0]

            # run it
            try:
                self.run_command(cmd, **kwargs)
                break
            except CommandException as e:
                errors.append(e)
                if self.allow_fallback_command(errors):
                    self.logger.warning(str(e))
                    self.logger.info("starting fallback command {}".format(len(errors)))
                else:
                    six.reraise(*sys.exc_info())

        self.post_run_command()

    def pre_run_command(self):
        return

    def post_run_command(self):
        return


class PlotTask(AnalysisTask):

    plot_function = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="a custom plot function to use; when empty, the default plot function is used "
        "instead; no default",
    )
    plot_postfix = luigi.Parameter(
        default=law.NO_STR,
        significant=False,
        description="an arbitrary postfix that is added with to underscores to all paths of "
        "produced plots; no default",
    )
    file_types = law.CSVParameter(
        default=("pdf",),
        description="comma-separated types of the output plot files; default: pdf",
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
    style = law.CSVParameter(
        default=(),
        description="one or more comma-separated names of a custom styles as provided by the "
        "underlying plot function; no default",
    )
    cms_postfix = luigi.Parameter(
        default=default_cms_postfix,
        significant=False,
        description="postfix to show after the CMS label; default: '{}'".format(default_cms_postfix),
    )
    save_hep_data = luigi.BoolParameter(
        default=False,
        description="save plot and meta data in a yaml file, compatible with the HEPData 'data' "
        "file syntax; default: False",
    )
    save_plot_data = luigi.BoolParameter(
        default=False,
        description="save arguments that are passed to the plot function also in a pkl file; "
        "default: False",
    )

    default_plot_function = None

    def get_axis_limit(self, value):
        if isinstance(value, six.string_types):
            value = getattr(self, value)
        return None if value == -1000.0 else value

    def create_plot_names(self, parts):
        plot_file_types = ["pdf", "png", "root", "c", "eps"]
        if any(t not in plot_file_types for t in self.file_types):
            raise Exception("plot names only allowed for file types {}, got {}".format(
                ",".join(plot_file_types), ",".join(self.file_types),
            ))

        if self.style:
            parts.append(("style", "_".join(self.style)))
        if self.plot_postfix and self.plot_postfix != law.NO_STR:
            parts.append((self.plot_postfix,))

        return ["{}.{}".format(self.join_postfix(parts), ext) for ext in self.file_types]

    @property
    def plot_function_id(self):
        if self.plot_function not in (None, law.NO_STR):
            return self.plot_function

        return self.default_plot_function

    def get_plot_func(self, func_id=None):
        # get the func_id
        if func_id is None:
            func_id = self.plot_function_id
        if func_id is None:
            raise ValueError(
                "cannot determine plot function, func_id is not set and no plot_function_id is "
                "defined on instance level",
            )

        if "." not in func_id:
            raise ValueError("invalid func_id format: {}".format(func_id))
        module_id, name = func_id.rsplit(".", 1)

        try:
            mod = importlib.import_module(module_id)
        except ImportError as e:
            raise ImportError(
                "cannot import plot function {} from module {}: {}".format(name, module_id, e),
            )

        func = getattr(mod, name, None)
        if func is None:
            raise Exception("module {} does not contain plot function {}".format(module_id, name))

        return func

    def call_plot_func(self, func_id=None, dump_target=None, **kwargs):
        plot_kwargs = self.update_plot_kwargs(kwargs)

        # dump here
        if dump_target:
            print("saved plot data in pkl file (loadable with encoding='latin1')")
            dump_target.dump(plot_kwargs, formatter="pickle")

        self.get_plot_func(func_id=func_id)(**plot_kwargs)

    def update_plot_kwargs(self, kwargs):
        return kwargs


class BoxPlotTask(PlotTask):

    pad_width = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="width of the pad in pixels; uses the default of the plot when empty; no "
        "default",
    )
    left_margin = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="left margin of the pad in pixels; uses the default of the plot when empty; no "
        "default",
    )
    right_margin = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="right margin of the pad in pixels; uses the default of the plot when empty; no "
        "default",
    )
    entry_height = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="vertical height of each entry in pixels; uses the default of the plot when "
        "empty; no default",
    )
    label_size = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="size of the nuisance labels on the y-axis; uses the default of the plot when "
        "empty; no default",
    )
    legend_columns = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="Force a number for legend columns "
        "empty; no default",
    )
    legend_x2 = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="Force a number for legend starting point "
        "empty; no default",
    )


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
                raise ValueError(
                    "the parameter '{}' contains {} value(s), but a minimum of {} is "
                    "required".format(s(v), len(v) - 1, self._min_len),
                )

            if self._max_len is not None and len(v) - 1 > self._max_len:
                raise ValueError(
                    "the parameter '{}' contains {} value(s), but a maximum of {} is "
                    "required".format(s(v), len(v) - 1, self._max_len),
                )

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

        # make unique, sort and apply length checks
        value = self._make_unique(value)
        value = self._sort_by_name(value)
        self._check_len(value)

        return value

    def serialize(self, value):
        if not value:
            value = tuple()

        value = law.util.make_tuple(value)

        # make unique, sort and apply length checks
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
        view_targets = OrderedDict()
        outputs = law.util.flatten(task.output())
        while outputs:
            output = outputs.pop(0)
            if isinstance(output, law.TargetCollection):
                outputs.extend(output._flat_target_list)
                continue
            if not getattr(output, "path", None):
                continue
            if output.path.endswith((".pdf", ".png")) and output.uri() not in view_targets:
                view_targets[output.uri()] = output

        # loop through targets and view them
        for target in view_targets.values():
            task.publish_message("showing {}".format(target.path))
            with target.localize("r") as tmp:
                law.util.interruptable_popen(
                    view_cmd.format(tmp.path),
                    shell=True,
                    executable="/bin/bash",
                )

    return before_call, call, after_call


class CommandException(Exception):

    def __init__(self, cmd, returncode, cwd=None):
        self.cmd = cmd
        self.returncode = returncode
        self.cwd = cwd or os.getcwd()

        msg = "command execution failed"
        msg += "\nexit code: {}".format(self.returncode)
        msg += "\ncwd      : {}".format(self.cwd)
        msg += "\ncommand  : {}".format(self.cmd)

        super(CommandException, self).__init__(msg)
