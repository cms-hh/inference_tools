#!/usr/bin/env bash

setup() {
    #
    # prepare local variables
    #

    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local orig="$PWD"
    local setup_name="${1:-default}"
    local setup_is_default="false"
    [ "$setup_name" = "default" ] && setup_is_default="true"


    #
    # global variables
    # (DHI = Di Higgs Inference)
    #

    export DHI_BASE="$this_dir"
    interactive_setup "$setup_name" || return "$?"
    export DHI_STORE_REPO="$DHI_BASE/data/store"
    export DHI_EXAMPLE_CARDS_GGF="$( echo /afs/cern.ch/user/m/mfackeld/public/datacards/dnn_score_max/*_dnn_node_HH_2B2VTo2L2Nu_GluGlu_NLO/datacard.txt | sed 's/ /,/g' )"
    export DHI_EXAMPLE_CARDS_VBF="$( echo /afs/cern.ch/user/m/mfackeld/public/datacards/dnn_score_max/*_dnn_node_HH_2B2VTo2L2Nu_VBF_NLO/datacard.txt | sed 's/ /,/g' )"
    export DHI_EXAMPLE_CARDS="$DHI_EXAMPLE_CARDS_GGF,$DHI_EXAMPLE_CARDS_VBF"
    export DHI_ORIG_PATH="$PATH"
    export DHI_ORIG_PYTHONPATH="$PYTHONPATH"
    export DHI_ORIG_PYTHON3PATH="$PYTHON3PATH"
    export DHI_ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    export DHI_SLACK_TOKEN="${DHI_SLACK_TOKEN:-}"
    export DHI_SLACK_CHANNEL="${DHI_SLACK_CHANNEL:-}"
    export DHI_TELEGRAM_TOKEN="${DHI_TELEGRAM_TOKEN:-}"
    export DHI_TELEGRAM_CHAT="${DHI_TELEGRAM_CHAT:-}"

    # lang defaults
    [ -z "$LANGUAGE" ] && export LANGUAGE="en_US.UTF-8"
    [ -z "$LANG" ] && export LANG="en_US.UTF-8"
    [ -z "$LC_ALL" ] && export LC_ALL="en_US.UTF-8"


    #
    # helper functions
    #

    # pip install helper
    dhi_pip_install() {
        PYTHONUSERBASE="$DHI_SOFTWARE" pip install --user --no-cache-dir "$@"
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_pip_install


    #
    # combine setup
    #

    local combine_version="1"
    local flag_file_combine="$DHI_SOFTWARE/.combine_good"
    [ "$DHI_REINSTALL_COMBINE" = "1" ] && rm -f "$flag_file_combine"
    if [ ! -f "$flag_file_combine" ]; then
        echo "installing combine into $DHI_SOFTWARE/HiggsAnalysis/CombinedLimit"
        mkdir -p "$DHI_SOFTWARE"

        (
            cd "$DHI_SOFTWARE"
            rm -rf HiggsAnalysis/CombinedLimit
            git clone --depth 1 --branch 102x https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit && \
            cd HiggsAnalysis/CombinedLimit && \
            source env_standalone.sh "" && \
            make -j && \
            make
        ) || return "$?"

        date "+%s" > "$flag_file_combine"
        echo "version $combine_version" >> "$flag_file_combine"
    fi
    export DHI_SOFTWARE_FLAG_FILES="$flag_file_combine"

    # check the version in the combine flag file and show a warning when there was an update
    if [ "$( cat "$flag_file_combine" | grep -Po "version \K\d+.*" )" != "$combine_version" ]; then
        2>&1 echo ""
        2>&1 echo "WARNING: your local combine installation is not up to date, please consider updating it in a new shell with"
        2>&1 echo "         > DHI_REINSTALL_COMBINE=1 source setup.sh $( $setup_is_default || echo "$setup_name" )"
        2>&1 echo ""
    fi

    # source it
    cd "$DHI_SOFTWARE/HiggsAnalysis/CombinedLimit"
    source env_standalone.sh "" || return "$?"
    # the setup script appends to PATH, but we need to prepend since some htcondor nodes seem to
    # have an other "combine" executable that gets picked instead
    export PATH="$PWD/exe:$PWD/scripts:$PATH"
    cd "$orig"


    #
    # minimal local software stack
    #

    # source externals
    for pkg in \
            libjpeg-turbo/1.3.1-omkpbe2 \
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
        source "/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/$pkg/etc/profile.d/init.sh" "" || return "$?"
    done

    # update paths and flags
    local pyv="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
    export PATH="$DHI_BASE/bin:$DHI_BASE/dhi/scripts:$DHI_BASE/modules/law/bin:$DHI_SOFTWARE/bin:$PATH"
    export PYTHONPATH="$DHI_BASE:$DHI_BASE/modules/law:$DHI_BASE/modules/plotlib:$DHI_SOFTWARE/lib/python${pyv}/site-packages:$DHI_SOFTWARE/lib64/python${pyv}/site-packages:$PYTHONPATH"
    export PYTHONWARNINGS="ignore"
    export PYTHONNOUSERSITE="1"
    export GLOBUS_THREAD_MODEL="none"
    ulimit -s unlimited

    # local stack
    local sw_version="1"
    local flag_file_sw="$DHI_SOFTWARE/.sw_good"
    [ "$DHI_REINSTALL_SOFTWARE" = "1" ] && rm -f "$flag_file_sw"
    if [ ! -f "$flag_file_sw" ]; then
        echo "installing software stack into $DHI_SOFTWARE"
        rm -rf "$DHI_SOFTWARE/lib"
        mkdir -p "$DHI_SOFTWARE"

        # python packages
        dhi_pip_install six==1.15.0 || return "$?"
        dhi_pip_install luigi==2.8.13 || return "$?"
        dhi_pip_install --no-deps git+https://github.com/riga/scinum.git || return "$?"
        dhi_pip_install tabulate==0.8.7 || return "$?"

        # optional packages, disabled at the moment
        # dhi_pip_install python-telegram-bot==12.3.0

        date "+%s" > "$flag_file_sw"
        echo "version $sw_version" >> "$flag_file_sw"
    fi
    export DHI_SOFTWARE_FLAG_FILES="$DHI_SOFTWARE_FLAG_FILES $flag_file_sw"

    # check the version in the sw flag file and show a warning when there was an update
    if [ "$( cat "$flag_file_sw" | grep -Po "version \K\d+.*" )" != "$sw_version" ]; then
        2>&1 echo ""
        2>&1 echo "WARNING: your local software stack is not up to date, please consider updating it in a new shell with"
        2>&1 echo "         > DHI_REINSTALL_SOFTWARE=1 source setup.sh $( $setup_is_default || echo "$setup_name" )"
        2>&1 echo ""
    fi

    # gfal2 bindings (optional)
    local lcg_dir="/cvmfs/grid.cern.ch/centos7-ui-4.0.3-1_umd4v3/usr"
    if [ ! -d "$lcg_dir" ]; then
        2>&1 echo "lcg directory $lcg_dir not existing, cannot setup gfal2 bindings"
    else
        export DHI_GFAL_DIR="$DHI_SOFTWARE/gfal2"
        export GFAL_PLUGIN_DIR="$DHI_GFAL_DIR/plugins"
        export PYTHONPATH="$DHI_GFAL_DIR:$PYTHONPATH"

        local flag_file_gfal="$DHI_SOFTWARE/.gfal_good"
        [ "$DHI_REINSTALL_GFAL" = "1" ] && rm -f "$flag_file_gfal"
        if [ ! -f "$flag_file_gfal" ]; then
            echo "linking gfal2 bindings"

            rm -rf "$DHI_GFAL_DIR"
            mkdir -p "$GFAL_PLUGIN_DIR"

            ln -s $lcg_dir/lib64/python2.7/site-packages/gfal2.so "$DHI_GFAL_DIR" || return "$?"
            ln -s $lcg_dir/lib64/gfal2-plugins/libgfal_plugin_* "$GFAL_PLUGIN_DIR" || return "$?"

            date "+%s" > "$flag_file_gfal"
        fi
        export DHI_SOFTWARE_FLAG_FILES="$DHI_SOFTWARE_FLAG_FILES $flag_file_gfal"
    fi


    #
    # initialze some submodules
    #

    if [ -d "$DHI_BASE/.git" ]; then
        for m in law plotlib; do
            local mpath="$DHI_BASE/modules/$m"
            # update and potentially initialize the submodule when the directory is either empty
            # or there are no changes
            local mfiles=( "$mpath"/* )
            if [ "${#mfiles}" = "0" ] || [ "$( cd "$mpath"; git status --porcelain=v1 2>/dev/null | wc -l )" = "0" ]; then
                git submodule update --init --recursive "$mpath"
            fi
        done
    fi


    #
    # law setup
    #

    export LAW_HOME="$DHI_BASE/.law"
    export LAW_CONFIG_FILE="$DHI_BASE/law.cfg"

    # source law's bash completion scipt
    which law &> /dev/null && source "$( law completion )" ""
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
    if ! $setup_is_default; then
        if [ -f "$env_file" ]; then
            echo -e "using variables for setup '\x1b[0;49;35m$setup_name\x1b[0m' from $env_file"
            source "$env_file" ""
            return "0"
        else
            echo -e "no setup file $env_file found for setup '\x1b[0;49;35m$setup_name\x1b[0m'"
        fi
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

        echo -e "Start querying variables for setup '\x1b[0;49;35m$setup_name\x1b[0m', press enter to accept default values\n"
    fi

    # start querying for variables
    query DHI_USER "CERN / WLCG username" "$( whoami )"
    query DHI_DATA "Local data directory" "$DHI_BASE/data" "./data"
    query DHI_STORE "Default local output store" "$DHI_DATA/store" "\$DHI_DATA/store"
    query DHI_STORE_BUNDLES "Output store for software bundles when submitting jobs" "$DHI_STORE" "\$DHI_STORE"
    query DHI_STORE_EOSUSER "Optional output store in EOS user directory" "/eos/user/${DHI_USER:0:1}/${DHI_USER}/dhi/store"
    query DHI_SOFTWARE "Directory for installing software" "$DHI_DATA/software" "\$DHI_DATA/software"
    query DHI_DATACARDS_RUN2 "Location of the datacards_run2 repository (optional)" "" "''"
    query DHI_JOB_DIR "Directory for storing job files" "$DHI_DATA/jobs" "\$DHI_DATA/jobs"
    query DHI_TASK_NAMESPACE "Namespace (i.e. the prefix) of law tasks" "" "''"
    query DHI_LOCAL_SCHEDULER "Use a local scheduler for law tasks" "True"
    if [ "$DHI_LOCAL_SCHEDULER" != "True" ]; then
        query DHI_SCHEDULER_HOST "Address of a central scheduler for law tasks" "hh:cmshhcombr2@hh-scheduler1.cern.ch"
        query DHI_SCHEDULER_PORT "Port of a central scheduler for law tasks" "80"
    else
        export_and_save DHI_SCHEDULER_HOST "hh:cmshhcombr2@hh-scheduler1.cern.ch"
        export_and_save DHI_SCHEDULER_PORT "80"
    fi

    # move the env file to the correct location for later use
    if ! $setup_is_default; then
        mv "$env_file_tmp" "$env_file"
        echo -e "\nsetup variables written to $env_file"
    fi
}

action() {
    if setup "$@"; then
        echo -e "\x1b[0;49;35mHH inference tools successfully setup\x1b[0m"
        return "0"
    else
        local code="$?"
        echo -e "\x1b[0;49;31msetup failed with code $code\x1b[0m"
        return "$code"
    fi
}
action "$@"
