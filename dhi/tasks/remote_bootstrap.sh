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
    export DHI_STORE="{{dhi_store}}"
    export DHI_USER="{{dhi_user}}"
    export DHI_TASK_NAMESPACE="{{dhi_task_namespace}}"
    export DHI_LOCAL_SCHEDULER="{{dhi_local_scheduler}}"
    export DHI_ON_HTCONDOR="1"
    export DHI_REMOTE_JOB="1"

    # load the software bundle
    mkdir -p "$DHI_SOFTWARE"
    cd "$DHI_SOFTWARE"
    fetch_local_file "{{dhi_software_pattern}}" software.tgz || return "$?"
    tar -xzf "software.tgz" || return "$?"
    rm "software.tgz"
    cd "$LAW_JOB_HOME"

    # load the repo bundle
    mkdir -p "$DHI_BASE"
    cd "$DHI_BASE"
    fetch_local_file "{{dhi_repo_pattern}}" repo.tgz || return "$?"
    tar -xzf "repo.tgz" || return "$?"
    rm "repo.tgz"
    cd "$LAW_JOB_HOME"

    # source the repo setup
    source "$DHI_BASE/setup.sh" "default" || return "$?"

    return "0"
}

# Copies a local, potentially replicated file to a certain location. When the file to copy contains
# pattern characters, e.g. "/path/to/some/file.*.tgz", a random existing file matching that pattern
# is selected.
# Arguments:
#   1. src_pattern: Path of a file or pattern matching multiple files of which one is copied.
#   2. dst_path   : Path where the source file should be copied to.
fetch_local_file() {
    # get arguments
    local src_pattern="$1"
    local dst_path="$2"

    # select one random file matched by pattern
    local src_path="$( ls $src_pattern | shuf -n 1 )"
    if [ -z "$src_path" ]; then
        2>&1 echo "could not determine random src file from pattern $src_pattern"
        return "1"
    fi
    echo "using source file $src_path"

    # create the target directory if it does not exist yet
    local dst_dir="$( dirname "$dst_path" )"
    [ ! -d "$dst_dir" ] && mkdir -p "$dst_dir"

    # copy the file
    cp "$src_path" "$dst_path"
}

bootstrap_{{dhi_bootstrap_name}} "$@"
