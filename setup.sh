#!/usr/bin/env bash

action() {
    #
    # prepare local variables
    #
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local orig="$PWD"


    #
    # global variables
    # (DHI = Di Higgs Inference)
    # some of them can be exported before this script is sourced for customization
    #

    export DHI_BASE="$this_dir"
    [ -z "$DHI_USER" ] && export DHI_USER="$( whoami )"
    [ -z "$DHI_DATA" ] && export DHI_DATA="/eos/user/${DHI_USER:0:1}/${DHI_USER}/dhi"
    [ -z "$DHI_SOFTWARE" ] && export DHI_SOFTWARE="/afs/cern.ch/work/${DHI_USER:0:1}/${DHI_USER}/dhi_software"
    [ -z "$DHI_STORE" ] && export DHI_STORE="$DHI_DATA/store"
    [ -z "$DHI_LOCAL_STORE" ] && export DHI_LOCAL_STORE="/afs/cern.ch/work/${DHI_USER:0:1}/${DHI_USER}/dhi_store"
    [ -z "$DHI_JOB_DIR" ] && export DHI_JOB_DIR="$DHI_BASE/data/jobs"
    [ -z "$DHI_N_CORES" ] && export DHI_N_CORES="$( grep -c ^processor /proc/cpuinfo )"
    export DHI_BLACK_PATH="$DHI_SOFTWARE/black"
    export DHI_EXAMPLE_CARDS="/afs/cern.ch/user/m/mfackeld/public/datacards/*/*.txt"
    export DHI_ORIG_PATH="$PATH"
    export DHI_ORIG_PYTHONPATH="$PYTHONPATH"
    export DHI_ORIG_PYTHON3PATH="$PYTHON3PATH"
    export DHI_ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    [ -z "$LANGUAGE" ] && export LANGUAGE=en_US.UTF-8
    [ -z "$LANG" ] && export LANG=en_US.UTF-8
    [ -z "$LC_ALL" ] && export LC_ALL=en_US.UTF-8


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
            git clone --depth 1 --branch v8.1.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit || return "1"
            cd HiggsAnalysis/CombinedLimit
            source env_standalone.sh "" || return "2"
            make -j "$DHI_N_CORES"
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
            py2-numpy/1.14.1-omkpbe2 \
            py2-scipy/1.1.0-ogkkac2 \
            py2-sympy/1.1.1-gnimlf \
            py2-matplotlib/1.5.2-omkpbe4 \
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
    export PATH="$DHI_BASE/bin:$DHI_SOFTWARE/bin:$PATH"
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


    #
    # law setup
    #

    export LAW_HOME="$DHI_BASE/.law"
    export LAW_CONFIG_FILE="$DHI_BASE/law.cfg"
    [ -z "$DHI_SCHEDULER_PORT" ] && export DHI_SCHEDULER_PORT="80"
    if [ -z "$DHI_LOCAL_SCHEDULER" ]; then
        export DHI_LOCAL_SCHEDULER="$( [ -z "$DHI_SCHEDULER_HOST" ] && echo True || echo False )"
    fi

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


    echo -e "\x1b[0;49;32mHH inference tools successfully setup\x1b[0m"
}
action "$@"
