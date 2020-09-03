#!/usr/bin/env bash

action() {
    #
    # prepare local variables
    #

    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"


    #
    # global variables
    # (DHI = Di Higgs Inference)
    #

    export DHI_BASE="$this_dir"
    [ -z "$DHI_USER" ] && export DHI_USER="$( whoami )"
    [ -z "$DHI_DATA" ] && export DHI_DATA="/eos/user/${DHI_USER:0:1}/${DHI_USER}/dhi"
    [ -z "$DHI_SOFTWARE" ] && export DHI_SOFTWARE="/afs/cern.ch/work/${DHI_USER:0:1}/${DHI_USER}/dhi_software"
    [ -z "$DHI_STORE" ] && export DHI_STORE="$DHI_DATA/store"
    [ -z "$DHI_LOCAL_STORE" ] && export DHI_LOCAL_STORE="/afs/cern.ch/work/${DHI_USER:0:1}/${DHI_USER}/dhi_store"
    [ -z "$DHI_DIST_VERSION" ] && export DHI_DIST_VERSION="slc7"
    export DHI_BLACK_PATH="$DHI_SOFTWARE/black"
    export DHI_EXAMPLE_CARDS="/afs/cern.ch/user/m/mfackeld/public/datacards/*/*.txt"
    export DHI_ORIG_PATH="$PATH"
    export DHI_ORIG_PYTHONPATH="$PYTHONPATH"
    export DHI_ORIG_PYTHON3PATH="$PYTHON3PATH"
    export DHI_ORIG_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

    # helper to reset the environment
    dhi_reset_env() {
        export PATH="$DHI_ORIG_PATH"
        export PYTHONPATH="$DHI_ORIG_PYTHONPATH"
        export PYTHON3PATH="$DHI_ORIG_PYTHON3PATH"
        export LD_LIBRARY_PATH="$DHI_ORIG_LD_LIBRARY_PATH"
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_reset_env

    [ -z "$LANGUAGE" ] && export LANGUAGE=en_US.UTF-8
    [ -z "$LANG" ] && export LANG=en_US.UTF-8
    [ -z "$LC_ALL" ] && export LC_ALL=en_US.UTF-8


    #
    # CMSSW setup
    #

    [ -z "$SCRAM_ARCH" ] && export SCRAM_ARCH="${DHI_DIST_VERSION}_amd64_gcc700"
    [ -z "$CMSSW_VERSION" ] && export CMSSW_VERSION="CMSSW_10_2_20_UL"
    [ -z "$CMSSW_BASE" ] && export CMSSW_BASE="$DHI_SOFTWARE/cmssw/$CMSSW_VERSION"

    source "/cvmfs/cms.cern.ch/cmsset_default.sh" ""

    if [ ! -f "$CMSSW_BASE/.good" ]; then
        local scram_cores="$( grep -c ^processor /proc/cpuinfo )"
        [ -z "$scram_cores" ] && scram_cores="1"

        if [ -d "$CMSSW_BASE" ]; then
            echo "remove already installed software in $CMSSW_BASE"
            rm -rf "$CMSSW_BASE"
        fi

        echo "setting up $CMSSW_VERSION with $SCRAM_ARCH in $CMSSW_BASE"

        (
            mkdir -p "$( dirname "$CMSSW_BASE" )"
            cd "$( dirname "$CMSSW_BASE" )"
            scramv1 project CMSSW "$CMSSW_VERSION" || return "$?"
            cd "$CMSSW_VERSION/src"
            eval `scramv1 runtime -sh` || return "$?"
            scram b || return "$?"

            # install CombinedLimit and CombineHarvester
            git clone --depth 1 -b v8.0.1 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
            # TODO: use CombineHarvester from cms-analysis again when branch is merged
            git clone --depth 1 -b feature/datacard_stacking_tool https://github.com/riga/CombineHarvester.git CombineHarvester

            # compile
            scram b -j "$scram_cores" || return "$?"
        )

        touch "$CMSSW_BASE/.good"
    fi

    local origin="$( pwd )"
    cd "$CMSSW_BASE/src" || return "$?"
    eval `scramv1 runtime -sh` || return "$?"
    cd "$origin"


    #
    # local software stack
    #

    # update paths
    local pyv="$( python -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
    export PATH="$DHI_BASE/bin:$DHI_SOFTWARE/bin:$PATH"
    export PYTHONPATH="$DHI_BASE:$DHI_SOFTWARE/lib/python${pyv}/site-packages:$DHI_SOFTWARE/lib64/python${pyv}/site-packages:$PYTHONPATH"

    # external software setup
    ulimit -s unlimited
    export GLOBUS_THREAD_MODEL="none"

    # pip install helper
    dhi_pip_install() {
        pip install --ignore-installed --no-cache-dir --prefix "$DHI_SOFTWARE" "$@"
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_pip_install

    # linting helper
    dhi_black() {
        (
            dhi_reset_env
            source "$DHI_BLACK_PATH/bin/activate" || return "$?"
            black "$@"
        )
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_black

    dhi_lint() {
        local fix="${1:-}"
        if [ "$fix" = "fix" ]; then
            dhi_black "$DHI_BASE/dhi"
        else
            dhi_black --check --diff "$DHI_BASE/dhi"
        fi
    }
    [ ! -z "$BASH_VERSION" ] && export -f dhi_lint

    # local stack
    if [ ! -f "$DHI_SOFTWARE/.good" ]; then
        echo "installing software stack into $DHI_SOFTWARE"
        mkdir -p "$DHI_SOFTWARE"

        # python packages
        rm -rf "$DHI_SOFTWARE/{bin,lib*}"
        LAW_INSTALL_EXECUTABLE=env dhi_pip_install git+https://github.com/riga/law.git --no-binary law || return "$?"

        # virtual env for black which requires python 3
        echo "setting up black in virtual environment at $DHI_BLACK_PATH"
        (
            dhi_reset_env
            rm -rf "$DHI_BLACK_PATH"
            virtualenv -p python3 "$DHI_BLACK_PATH" || return "$?"
            source "$DHI_BLACK_PATH/bin/activate" || return "$?"
            pip install -U pip || return "$?"
            pip install black==19.10b0 || return "$?"
        )

        touch "$DHI_SOFTWARE/.good"
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
    source "$( law completion )" ""


    #
    # synchronize git hooks
    #

    for hook in "$( ls "$this_dir/githooks" )"; do
        if [ ! -f "$this_dir/.git/hooks/$hook" ]; then
            ln -s "../../githooks/$hook" "$this_dir/.git/hooks/$hook" &> /dev/null
        fi
    done
}
action "$@"
