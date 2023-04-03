# coding: utf-8

"""
Tasks that handle remote job submission.
"""

import os
import re
import math

import luigi
import law

from dhi import dhi_htcondor_flavor
from dhi.tasks.base import AnalysisTask, UserTask
from dhi.util import expand_path


logger = law.logger.get_logger(__name__)


class BundleRepo(UserTask, law.git.BundleGitRepository, law.tasks.TransferLocalFile):

    replicas = luigi.IntParameter(
        default=10,
        description="number of replicas to generate; default: 10",
    )

    exclude_files = ["docs", "data", ".law", ".setups", "datacards_run2*/*", "*~", "*.pyc"]

    version = None
    task_namespace = None
    default_store = "$DHI_STORE_BUNDLES"

    def get_repo_path(self):
        # required by BundleGitRepository
        return os.environ["DHI_BASE"]

    def single_output(self):
        repo_base = os.path.basename(self.get_repo_path())
        return self.target("{}.{}.tgz".format(repo_base, self.checksum))

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
        self.publish_message("bundled repository archive, size is {}".format(
            law.util.human_bytes(bundle.stat().st_size, fmt=True),
        ))

        # transfer the bundle
        self.transfer(bundle)


class BundleSoftware(UserTask, law.tasks.TransferLocalFile):

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
        return self.target("software.{}.tgz".format(self.checksum))

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
            if re.search(r"^(.+/)?CMSSW_\d+_\d+_\d+", tarinfo.name):
                return None
            if re.search(r"^(.+/)?combine_.*_amd64_gcc\d+", tarinfo.name):
                return None
            return tarinfo

        # create the archive with a custom filter
        bundle.dump(software_path, add_kwargs={"filter": _filter})

        # log the size
        self.publish_message("bundled software archive, size is {}".format(
            law.util.human_bytes(bundle.stat().st_size, fmt=True),
        ))

        # transfer the bundle
        self.transfer(bundle)


class BundleCMSSW(UserTask, law.cms.BundleCMSSW, law.tasks.TransferLocalFile):

    replicas = luigi.IntParameter(
        default=10,
        description="number of replicas to generate; default: 10",
    )

    version = None
    task_namespace = None
    exclude = "^src/tmp"
    default_store = "$DHI_STORE_BUNDLES"

    def get_cmssw_path(self):
        return os.environ["CMSSW_BASE"]

    def single_output(self):
        path = "{}.{}.tgz".format(os.path.basename(self.get_cmssw_path()), self.checksum)
        return self.target(path)

    def get_file_pattern(self):
        path = os.path.expandvars(os.path.expanduser(self.single_output().path))
        return self.get_replicated_path(path, i=None if self.replicas <= 0 else "*")

    def output(self):
        return law.tasks.TransferLocalFile.output(self)

    def run(self):
        # create the bundle
        bundle = law.LocalFileTarget(is_tmp="tgz")
        self.bundle(bundle)

        # log the size
        self.publish_message("bundled CMSSW archive, size is {}".format(
            law.util.human_bytes(bundle.stat().st_size, fmt=True),
        ))

        # transfer the bundle
        self.transfer(bundle)


class HTCondorWorkflow(AnalysisTask, law.htcondor.HTCondorWorkflow):

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
    htcondor_share_software = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, all software (except the code repository itself) is assumed to be "
        "accessible via network mounts and sourced from there; default: False",
    )
    htcondor_cpus = luigi.IntParameter(
        default=law.NO_INT,
        significant=False,
        description="number of CPUs to request; empty value leads to the cluster default setting; "
        "no default",
    )
    htcondor_mem = law.BytesParameter(
        default=law.NO_FLOAT,
        unit="GB",
        significant=False,
        description="amount of memory to request; the default unit is GB; empty value leads to the "
        "cluster default setting; no default",
    )
    htcondor_flavor = luigi.ChoiceParameter(
        default=dhi_htcondor_flavor,
        choices=("cern", "naf", "infn"),
        significant=False,
        description="the 'flavor' (i.e. configuration name) of the batch system; choices: "
        "cern,naf,infn; default: {}".format(dhi_htcondor_flavor),
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
        "max_runtime", "htcondor_cpus", "htcondor_mem", "htcondor_flavor", "htcondor_getenv",
        "htcondor_group",
    }

    def htcondor_workflow_requires(self):
        reqs = law.htcondor.HTCondorWorkflow.htcondor_workflow_requires(self)

        # add repo and software bundling as requirements when getenv is not requested
        if not self.htcondor_getenv:
            reqs["repo"] = BundleRepo.req(self, replicas=3)
            if not self.htcondor_share_software:
                reqs["software"] = BundleSoftware.req(self, replicas=3)
                reqs["cmssw"] = BundleCMSSW.req(self, replicas=3)

        return reqs

    def htcondor_output_directory(self):
        # the directory where submission meta data and logs should be stored
        return self.local_target(store="$DHI_STORE_JOBS", dir=True)

    def htcondor_bootstrap_file(self):
        # each job can define a bootstrap file that is executed prior to the actual job
        # in order to setup software and environment variables
        bootstrap_file = os.path.expandvars("$DHI_BASE/dhi/tasks/remote_bootstrap.sh")
        return law.JobInputFile(bootstrap_file, share=True, render_job=True)

    def htcondor_job_config(self, config, job_num, branches):
        # add the user proxy when existing
        voms_proxy_file = law.wlcg.get_voms_proxy_file()
        if os.path.exists(voms_proxy_file):
            config.input_files["voms_proxy_file"] = law.JobInputFile(
                voms_proxy_file,
                share=True,
                render=False,
            )
            config.render_variables["dhi_x509_cert_dir"] = os.getenv("X509_CERT_DIR", "")
            config.render_variables["dhi_x509_voms_dir"] = os.getenv("X509_VOMS_DIR", "")

        # use cc7 at CERN (http://batchdocs.web.cern.ch/batchdocs/local/submit.html#os-choice)
        # and NAF
        if self.htcondor_flavor in ("cern", "naf"):
            config.custom_content.append(("requirements", '(OpSysAndVer =?= "CentOS7")'))
        # architecture at INFN
        if self.htcondor_flavor == "infn":
            config.custom_content.append((
                "requirements",
                'TARGET.OpSys == "LINUX" && (TARGET.Arch != "DUMMY")',
            ))

        # copy the entire environment when requests
        if self.htcondor_getenv:
            config.custom_content.append(("getenv", "true"))

        # include the wlcg specific tools script in the input sandbox
        tools_file = law.util.law_src_path("contrib/wlcg/scripts/law_wlcg_tools.sh")
        config.input_files["wlcg_tools"] = law.JobInputFile(tools_file, share=True, render=False)

        # the CERN htcondor setup requires a "log" config, but we can safely set it to /dev/null
        # if you are interested in the logs of the batch system itself, set a meaningful value here
        config.custom_content.append(("log", "/dev/null"))

        # max runtime
        max_runtime = int(math.floor(self.max_runtime * 3600)) - 1
        config.custom_content.append(("+MaxRuntime", max_runtime))
        config.custom_content.append(("+RequestRuntime", max_runtime))

        # request cpus
        if self.htcondor_cpus > 0:
            if self.htcondor_flavor == "naf":
                self.logger.warning(
                    "--htcondor-cpus has no effect on NAF resources, use --htcondor-mem instead",
                )
            else:
                config.custom_content.append(("RequestCpus", self.htcondor_cpus))

        # request memory
        if self.htcondor_mem > 0:
            if self.htcondor_flavor == "cern":
                self.logger.warning(
                    "--htcondor-mem has no effect on CERN resources, use --htcondor-cpus instead",
                )
            elif self.htcondor_flavor == "naf":
                # NAF uses MB
                config.custom_content.append(("RequestMemory", self.htcondor_mem * 1024))
            else:
                config.custom_content.append(("RequestMemory", self.htcondor_mem))

        # accounting group for priority on the cluster
        if self.htcondor_group and self.htcondor_group != law.NO_STR:
            config.custom_content.append(("+AccountingGroup", self.htcondor_group))

        # helper to return uris and a file pattern for replicated bundles
        reqs = self.htcondor_workflow_requires()
        def get_bundle_info(task):
            uris = task.output().dir.uri(base_name="filecopy", return_all=True)
            pattern = os.path.basename(task.get_file_pattern())
            return ",".join(uris), pattern

        # prepare the hook file location
        hook_file = os.getenv("DHI_HOOK_FILE", "")
        if hook_file:
            hook_file = expand_path(hook_file)
            dhi_base = expand_path("$DHI_BASE")
            if hook_file.startswith(dhi_base):
                hook_file = os.path.relpath(hook_file, dhi_base)

        # render_variables are rendered into all files sent with a job
        config.render_variables["dhi_env_path"] = os.environ["PATH"]
        config.render_variables["dhi_env_pythonpath"] = os.environ["PYTHONPATH"]
        config.render_variables["dhi_htcondor_flavor"] = self.htcondor_flavor
        config.render_variables["dhi_base"] = os.environ["DHI_BASE"]
        config.render_variables["dhi_user"] = os.environ["DHI_USER"]
        config.render_variables["dhi_store"] = os.environ["DHI_STORE"]
        config.render_variables["dhi_local_scheduler"] = os.environ["DHI_LOCAL_SCHEDULER"]
        config.render_variables["dhi_hook_file"] = hook_file
        if self.htcondor_getenv:
            config.render_variables["dhi_bootstrap_name"] = "htcondor_getenv"
        else:
            config.render_variables["dhi_bootstrap_name"] = "htcondor_standalone"
            config.render_variables["dhi_lcg_dir"] = os.environ["DHI_LCG_DIR"]
            if self.htcondor_share_software:
                config.render_variables["dhi_software"] = os.environ["DHI_SOFTWARE"]

            # add repo bundle variables
            uris, pattern = get_bundle_info(reqs["repo"])
            config.render_variables["dhi_repo_uris"] = uris
            config.render_variables["dhi_repo_pattern"] = pattern

            # add software bundle variables
            if "software" in reqs:
                uris, pattern = get_bundle_info(reqs["software"])
                config.render_variables["dhi_software_uris"] = uris
                config.render_variables["dhi_software_pattern"] = pattern

            # add cmssw bundle variables
            if "cmssw" in reqs:
                uris, pattern = get_bundle_info(reqs["cmssw"])
                config.render_variables["dhi_cmssw_uris"] = uris
                config.render_variables["dhi_cmssw_pattern"] = pattern
                config.render_variables["dhi_scram_arch"] = os.environ["DHI_SCRAM_ARCH"]
                config.render_variables["dhi_cmssw_version"] = os.environ["DHI_CMSSW_VERSION"]
                config.render_variables["dhi_combine_version"] = os.environ["DHI_COMBINE_VERSION"]

        return config

    def htcondor_use_local_scheduler(self):
        # remote jobs should not communicate with ther central scheduler but with a local one
        return True

    def control_output_postfix(self):
        parts = [
            super(HTCondorWorkflow, self).control_output_postfix(),
            self.get_output_postfix(),
        ]
        return "__".join(list(map(str, filter(bool, parts))))
