#!/usr/bin/env bash

# Bootstrap file that is executed by remote jobs submitted by law to set up the environment. This
# file contains multiple different bootstrap functions of which only one is invoked by the last line
# of this file. So-called render variables, denoted by "{{name}}", are replaced with variables
# configured in the remote workflow tasks, e.g. in HTCondorWorkflow.htcondor_job_config(), upon job
# submission.

# Bootstrap function for htcondor jobs that have the getenv feature enabled, i.e., environment
# variables of the submitting shell and the job node will be identical.
bootstrap_htcondor_getenv() {
    # on the CERN HTCondor batch, the PATH and PYTHONPATH variables are changed even though "getenv"
    # is set, in the job file, so set them manually to the desired values
    if [ "{{dhi_htcondor_flavor}}" = "cern" ]; then
        export PATH="{{dhi_env_path}}"
        export PYTHONPATH="{{dhi_env_pythonpath}}"
    fi

    # set env variables
    export DHI_ON_HTCONDOR="1"
    export DHI_REMOTE_JOB="1"

    return "0"
}

# Bootstrap function for standalone htcondor jobs, i.e., each jobs fetches a software and repository
# code bundle and unpacks them to have a standalone environment, independent of the submitting one.
# The setup script of the repository is sourced with a few environment variables being set before,
# tailored for remote jobs.
bootstrap_htcondor_standalone() {
    # set env variables
    export DHI_BASE="$LAW_JOB_HOME/repo"
    export DHI_DATA="$LAW_JOB_HOME/dhi_data"
    export DHI_SOFTWARE="$DHI_DATA/software"
    export DHI_USER="{{dhi_user}}"
    export DHI_STORE="{{dhi_store}}"
    export DHI_COMBINE_STANDALONE="{{dhi_combine_standalone}}"
    export DHI_TASK_NAMESPACE="{{dhi_task_namespace}}"
    export DHI_LOCAL_SCHEDULER="{{dhi_local_scheduler}}"
    export DHI_HOOK_FILE="{{dhi_hook_file}}"
    export DHI_ON_HTCONDOR="1"
    export DHI_REMOTE_JOB="1"

    # source the law wlcg tools, mainly for law_wlcg_get_file
    source "law_wlcg_tools{{file_postfix}}.sh" ""

    # start loading bundles
    mkdir -p "$DHI_SOFTWARE"

    # load the cmssw bundle
    if [ "$DHI_COMBINE_STANDALONE" != "True" ]; then
        cd "$DHI_SOFTWARE"
        source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" || return "$?"
        export SCRAM_ARCH="{{dhi_scram_arch}}"
        export CMSSW_VERSION="{{dhi_cmssw_version}}"
        scramv1 project CMSSW "$CMSSW_VERSION" || return "$?"
        cd "$CMSSW_VERSION"
        law_wlcg_get_file "{{dhi_cmssw_uris}}" "{{dhi_cmssw_pattern}}" "$PWD/cmssw.tgz" || return "$?"
        tar -xzf cmssw.tgz || return "$?"
        cd "src" || return "$?"
        eval "$( scramv1 runtime -sh )" || return "$?"
        scram b || return "$?"
        cd "$LAW_JOB_HOME"
    fi

    # load the software bundle
    cd "$DHI_SOFTWARE"
    law_wlcg_get_file "{{dhi_software_uris}}" "{{dhi_software_pattern}}" "$PWD/software.tgz" || return "$?"
    tar -xzf "software.tgz" || return "$?"
    rm "software.tgz"
    cd "$LAW_JOB_HOME"

    # load the repo bundle
    mkdir -p "$DHI_BASE"
    cd "$DHI_BASE"
    law_wlcg_get_file "{{dhi_repo_uris}}" "{{dhi_repo_pattern}}" "$PWD/repo.tgz" || return "$?"
    tar -xzf "repo.tgz" || return "$?"
    rm "repo.tgz"
    cd "$LAW_JOB_HOME"

    # source the repo setup
    source "$DHI_BASE/setup.sh" "default" || return "$?"

    return "0"
}

bootstrap_{{dhi_bootstrap_name}} "$@"
