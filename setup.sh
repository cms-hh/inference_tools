#!/usr/bin/env bash

action() {
    #
    # prepare local variables
    #

    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local orig="$PWD"
    local setup_name="${1:-default}"


    #
    # global variables
    # (DHI = Di Higgs Inference)
    #

    export DHI_BASE="$this_dir"
    interactive_setup "$setup_name"
    export DHI_STORE_REPO="$DHI_BASE/data/store"
    export DHI_BLACK_PATH="$DHI_SOFTWARE/black"
    export DHI_EXAMPLE_CARDS="/afs/cern.ch/user/m/mfackeld/public/datacards/*/*.txt"
    export DHI_ORIG_PATH="$PATH"
    export DHI_ORIG_PYTHONPATH="$PYTHONPATH"
    export DHI_ORIG_PYTHON3PATH="$PYTHON3PATH"
    export DHI_ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

    # lang defaults
    [ -z "$LANGUAGE" ] && export LANGUAGE="en_US.UTF-8"
    [ -z "$LANG" ] && export LANG="en_US.UTF-8"
    [ -z "$LC_ALL" ] && export LC_ALL="en_US.UTF-8"


    #
    # helper functions
    #

    # helper to reset some environment variables
    dhi_reset_env() {
        export PATH="$DHI_ORIG_PATH"
        export PYTHONPATH="$DHI_ORIG_PYTHONPATH"
        export PYTHON3PATH="$DHI_ORIG_PYTHON3PATH"
        export LD_LIBRARY_PATH="$DHI_ORIG_LD_LIBRARY_PATH"
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_reset_env

    # pip install helper
    dhi_pip_install() {
        PYTHONUSERBASE="$DHI_SOFTWARE" pip install --user --no-cache-dir "$@"
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_pip_install

    # generic wrapper to run black in a virtual env in case a user prefers python 2
    # which black already dropped
    dhi_black() {
        (
            dhi_reset_env
            source "$DHI_BLACK_PATH/bin/activate" "" || return "$?"
            black "$@"
        )
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_black

    # linting helper
    dhi_lint() {
        local fix="${1:-}"
        if [ "$fix" = "fix" ]; then
            dhi_black --line-length 100 "$DHI_BASE/dhi"
        else
            dhi_black --line-length 100 --check --diff "$DHI_BASE/dhi"
        fi
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_lint


    #
    # combine setup
    #

    local combine_flag_file="$DHI_SOFTWARE/.combine_good"
    if [ ! -f "$combine_flag_file" ]; then
        echo "installing combine into $DHI_SOFTWARE/HiggsAnalysis/CombinedLimit"
        mkdir -p "$DHI_SOFTWARE"

        (
            cd "$DHI_SOFTWARE"
            rm -rf HiggsAnalysis/CombinedLimit
            # TODO: the following branch is based on v8.1.0 and adds the --points2 parameter to
            # likelihood scans to improve control over the scan grid, so switch back again to the
            # original repo once merged, https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/pull/623
            git clone --depth 1 --branch control_2d_grid https://github.com/riga/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit || return "1"
            cd HiggsAnalysis/CombinedLimit
            source env_standalone.sh "" || return "2"
            make -j
            make || return "3"
        )

        touch "$combine_flag_file"
    fi

    cd "$DHI_SOFTWARE/HiggsAnalysis/CombinedLimit"
    source env_standalone.sh ""
    cd "$orig"


    #
    # minimal local software stack
    #

    # source externals
    for pkg in \
            py2-pip/9.0.3 \
            py2-numpy/1.16.2-pafccj \
            py2-scipy/1.2.1-pafccj \
            py2-sympy/1.3-pafccj \
            py2-matplotlib/2.2.3 \
            py2-cycler/0.10.0-gnimlf \
            py2-uproot/2.6.19-gnimlf \
            py2-requests/2.20.0 \
            py2-urllib3/1.25.3 \
            py2-idna/2.8 \
            py2-chardet/3.0.4-gnimlf \
            py2-ipython/5.5.0-ogkkac \
            py2-backports/1.0 \
            ; do
        source "/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/$pkg/etc/profile.d/init.sh" ""
    done

    # update paths and flags
    local pyv="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
    export PATH="$DHI_BASE/bin:$DHI_BASE/dhi/scripts:$DHI_SOFTWARE/bin:$PATH"
    export PYTHONPATH="$DHI_BASE:$DHI_SOFTWARE/lib/python${pyv}/site-packages:$DHI_SOFTWARE/lib64/python${pyv}/site-packages:$PYTHONPATH"
    export PYTHONWARNINGS="ignore"
    export GLOBUS_THREAD_MODEL="none"
    ulimit -s unlimited

    # local stack
    local sw_flag_file="$DHI_SOFTWARE/.sw_good"
    if [ ! -f "$sw_flag_file" ]; then
        echo "installing software stack into $DHI_SOFTWARE"
        mkdir -p "$DHI_SOFTWARE"

        # python packages
        dhi_pip_install luigi==2.8.2 || return "$?"
        LAW_INSTALL_EXECUTABLE="$DHI_PYTHON" dhi_pip_install --no-deps git+https://github.com/riga/law.git || return "$?"
        dhi_pip_install --no-deps git+https://github.com/riga/scinum.git || return "$?"

        # virtual env for black which requires python 3
        echo -e "\nsetting up black in virtual environment at $DHI_BLACK_PATH"
        (
            dhi_reset_env
            rm -rf "$DHI_BLACK_PATH"
            virtualenv -p python3 "$DHI_BLACK_PATH" || return "$?"
            source "$DHI_BLACK_PATH/bin/activate" "" || return "$?"
            pip install -U pip || return "$?"
            pip install black==20.8b1 || return "$?"
        )

        touch "$sw_flag_file"
    fi

    # gfal2 bindings
    local lcg_dir="/cvmfs/grid.cern.ch/centos7-ui-4.0.3-1_umd4v3/usr"
    if [ ! -d "$lcg_dir" ]; then
        2>&1 echo "lcg directory $lcg_dir not existing, cannot setup gfal2 bindings"
        return "4"
    fi

    export DHI_GFAL_DIR="$DHI_SOFTWARE/gfal2"
    export GFAL_PLUGIN_DIR="$DHI_GFAL_DIR/plugins"
    export PYTHONPATH="$DHI_GFAL_DIR:$PYTHONPATH"

    local gfal_flag_file="$DHI_SOFTWARE/.gfal_good"
    if [ ! -f "$gfal_flag_file" ]; then
        echo "linking gfal2 bindings"

        mkdir -p "$GFAL_PLUGIN_DIR"
        ln -s $lcg_dir/lib64/python2.7/site-packages/gfal2.so "$DHI_GFAL_DIR"
        ln -s $lcg_dir/lib64/gfal2-plugins/libgfal_plugin_* "$GFAL_PLUGIN_DIR"

        touch "$gfal_flag_file"
    fi


    #
    # law setup
    #

    export LAW_HOME="$DHI_BASE/.law"
    export LAW_CONFIG_FILE="$DHI_BASE/law.cfg"

    # source law's bash completion scipt
    which law &> /dev/null && source "$( law completion )" ""


    #
    # synchronize git hooks
    #

    # this is disabled for the moment to avoid difficulties related to formatting
    # in the initial phase of the combination tools development
    # for hook in "$( ls "$this_dir/githooks" )"; do
    #     if [ ! -f "$this_dir/.git/hooks/$hook" ]; then
    #         ln -s "../../githooks/$hook" "$this_dir/.git/hooks/$hook" &> /dev/null
    #     fi
    # done


    echo -e "\x1b[0;49;35mHH inference tools successfully setup\x1b[0m"
}

interactive_setup() {
    local setup_name="${1:-default}"
    local env_file="${2:-$DHI_BASE/.setups/$setup_name.sh}"
    local env_file_tmp="$env_file.tmp"

    # check if the setup is the default one
    local setup_is_default="false"
    [ "$setup_name" = "default" ] && setup_is_default="true"

    # when the setup already exists and it's not the default one,
    # source the corresponding env file and stop
    if ! $setup_is_default && [ -f "$env_file" ]; then
        echo "using setup variables from $env_file"
        source "$env_file" ""
        return "0"
    fi

    export_and_save() {
        local varname="$1"
        local value="$2"

        export $varname="$value"
        ! $setup_is_default && echo "export $varname=\"$value\"" >> "$env_file_tmp"
    }

    query() {
        local varname="$1"
        local text="$2"
        local default="$3"
        local default_text="${4:-$default}"

        # when the setup is the default one, use the default value when the env variable is empty,
        # otherwise, query interactively
        local value="$default"
        if $setup_is_default; then
            [ ! -z "${!varname}" ] && value="${!varname}"
        else
            printf "$text (\x1b[1;49;39m$varname\x1b[0m, default \x1b[1;49;39m$default_text\x1b[0m):  "
            read query_response
            [ "X$query_response" = "X" ] && query_response="$default"

            # repeat for boolean flags that were not entered correctly
            while true; do
                ( [ "$default" != "True" ] && [ "$default" != "False" ] ) && break
                ( [ "$query_response" = "True" ] || [ "$query_response" = "False" ] ) && break
                printf "please enter either '\x1b[1;49;39mTrue\x1b[0m' or '\x1b[1;49;39mFalse\x1b[0m':  " query_response
                read query_response
                [ "X$query_response" = "X" ] && query_response="$default"
            done

            # save the expanded value
            value="$( eval "echo $query_response" )"
            # strip " and '
            value=${value%\"}
            value=${value%\'}
            value=${value#\"}
            value=${value#\'}
        fi

        export_and_save "$varname" "$value"
    }

    # prepare the tmp env file
    if ! $setup_is_default; then
        rm -rf "$env_file_tmp"
        mkdir -p "$( dirname "$env_file_tmp" )"

        echo -e "Start querying variables for setup '$setup_name', press enter to accept default values\n"
    fi

    # start querying for variables
    query DHI_USER "Username on lxplus" "$( whoami )"
    query DHI_DATA "Local data directory" "$DHI_BASE/data" "./data"
    query DHI_STORE "Default local output store" "$DHI_DATA/store" "\$DHI_DATA/store"
    query DHI_STORE_BUNDLES "Output store for software bundles when submitting jobs" "$DHI_STORE" "\$DHI_STORE"
    query DHI_STORE_EOSUSER "Optional output store in EOS user directory" "/eos/user/${DHI_USER:0:1}/${DHI_USER}/dhi/store"
    query DHI_SOFTWARE "Directory for installing software" "$DHI_DATA/software" "\$DHI_DATA/software"
    query DHI_JOB_DIR "Directory for storing job files" "$DHI_DATA/jobs" "\$DHI_DATA/jobs"
    query DHI_TASK_NAMESPACE "Namespace (i.e. the prefix) of law tasks" ""
    query DHI_LOCAL_SCHEDULER "Use a local scheduler for law tasks" "True"
    if [ "$DHI_LOCAL_SCHEDULER" != "True" ]; then
        query DHI_SCHEDULER_HOST "Address of a central scheduler for law tasks" "hh:cmshhcombr2@hh-scheduler1.cern.ch"
        query DHI_SCHEDULER_PORT "Port of a central scheduler for law tasks" "80"
    else
        export_and_save DHI_SCHEDULER_PORT "80"
    fi

    # move the env file to the correct location for later use
    if ! $setup_is_default; then
        mv "$env_file_tmp" "$env_file"
        echo -e "\nsetup variables written to $env_file"
    fi
}

action "$@"
