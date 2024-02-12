#!/usr/bin/env bash

setup() {
    #
    # prepare local variables
    #

    local this_file="$( [ ! -z "${ZSH_VERSION}" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local setup_name="${1:-default}"
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"


    #
    # global variables
    # (DHI = Di Higgs Inference)
    #

    export DHI_BASE="${this_dir}"
    interactive_setup "${setup_name}" || return "$?"
    export DHI_SETUP_NAME="${setup_name}"
    export DHI_STORE_REPO="${DHI_BASE}/data/store"
    export DHI_ON_HTCONDOR="${DHI_ON_HTCONDOR:-0}"
    export DHI_ON_SLURM="${DHI_ON_SLURM:-0}"
    export DHI_ON_CRAB="${DHI_ON_CRAB:-0}"
    export DHI_REMOTE_JOB="${DHI_REMOTE_JOB:-0}"

    export DHI_SLACK_TOKEN="${DHI_SLACK_TOKEN:-}"
    export DHI_SLACK_CHANNEL="${DHI_SLACK_CHANNEL:-}"
    export DHI_TELEGRAM_TOKEN="${DHI_TELEGRAM_TOKEN:-}"
    export DHI_TELEGRAM_CHAT="${DHI_TELEGRAM_CHAT:-}"

    export DHI_ORIG_PATH="${PATH}"
    export DHI_ORIG_PYTHONPATH="${PYTHONPATH}"
    export DHI_ORIG_PYTHON3PATH="${PYTHON3PATH}"
    export DHI_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"

    # backwards compatible settings queried during interactive setup only recently
    export DHI_WLCG_CACHE_ROOT="${DHI_WLCG_CACHE_ROOT:-}"
    export DHI_WLCG_USE_CACHE="${DHI_WLCG_USE_CACHE:-false}"

    # change some defaults in remote jobs
    if [ "${DHI_REMOTE_JOB}" = "1" ]; then
        export DHI_REINSTALL_COMBINE="0"
        export DHI_REINSTALL_SOFTWARE="0"
    fi

    # lang defaults
    [ -z "${LANGUAGE}" ] && export LANGUAGE="en_US.UTF-8"
    [ -z "${LANG}" ] && export LANG="en_US.UTF-8"
    [ -z "${LC_ALL}" ] && export LC_ALL="en_US.UTF-8"


    #
    # example cards
    #

    [ -z "$DHI_EXAMPLE_CARDS_BASE" ] && export DHI_EXAMPLE_CARDS_BASE="/afs/cern.ch/user/m/mrieger/public/hh/inference/examplecards/v2"

    expand_cards() {
        local pattern="$1"
        eval "echo ${DHI_EXAMPLE_CARDS_BASE}/${pattern}" | sed 's/ /,/g'
    }

    export DHI_EXAMPLE_CARDS="$( expand_cards 'sm/datacard*.txt' )"
    export DHI_EXAMPLE_CARDS_1="$( expand_cards 'sm/datacard1.txt' )"
    export DHI_EXAMPLE_CARDS_2="$( expand_cards 'sm/datacard2.txt' )"
    export DHI_EXAMPLE_CARDS_3="$( expand_cards 'sm/datacard3.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_C2="$( expand_cards 'eft_c2/datacard*.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_C2_1="$( expand_cards 'eft_c2/datacard1.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_C2_2="$( expand_cards 'eft_c2/datacard2.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_C2_3="$( expand_cards 'eft_c2/datacard3.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_BM="$( expand_cards 'eft_bm/datacard1_*.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_BM_1="$( expand_cards 'eft_bm/datacard1_*.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_BM_2="$( expand_cards 'eft_bm/datacard2_*.txt' )"
    export DHI_EXAMPLE_CARDS_EFT_BM_3="$( expand_cards 'eft_bm/datacard3_*.txt' )"
    export DHI_EXAMPLE_CARDS_RES="$( expand_cards 'res/datacard1_*.txt' )"
    export DHI_EXAMPLE_CARDS_RES_1="$( expand_cards 'res/datacard1_*.txt' )"
    export DHI_EXAMPLE_CARDS_RES_2="$( expand_cards 'res/datacard2_*.txt' )"
    export DHI_EXAMPLE_CARDS_RES_3="$( expand_cards 'res/datacard3_*.txt' )"


    #
    # helper functions
    #

    # pip install helper
    dhi_pip_install() {
        PYTHONUSERBASE="${DHI_SOFTWARE}" pip3 install --user --no-cache-dir "$@"
    }
    [ ! -z "${BASH_VERSION}" ] && export -f dhi_pip_install

    # remove cache locks on cached wlcg targets
    dhi_remove_cache_locks() {
        if [ ! -z "${DHI_WLCG_CACHE_ROOT}" ] && [ -d "${DHI_WLCG_CACHE_ROOT}" ]; then
            find -L "${DHI_WLCG_CACHE_ROOT}" . -name '*.lock' | xargs rm
        fi
    }
    [ ! -z "${BASH_VERSION}" ] && export -f dhi_remove_cache_locks


    #
    # CMSSW & combine setup
    # see https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit
    #

    export DHI_SCRAM_ARCH="${DHI_SCRAM_ARCH:-slc7_amd64_gcc12}"
    export DHI_CMSSW_VERSION="${DHI_CMSSW_VERSION:-CMSSW_14_0_0_pre0}"
    export DHI_COMBINE_VERSION="${DHI_COMBINE_VERSION:-14x-comb2023}"
    export DHI_CMSSW_BASE="${DHI_SOFTWARE}/combine_${DHI_COMBINE_VERSION}_${DHI_SCRAM_ARCH}"

    local flag_file_combine="${DHI_CMSSW_BASE}/.combine_${DHI_CMSSW_VERSION}_good"
    local combine_version="6"

    source "/cvmfs/cms.cern.ch/cmsset_default.sh" "" || return "$?"
    export SCRAM_ARCH="${DHI_SCRAM_ARCH}"

    # reset combine if requested
    if [ "${DHI_REMOTE_JOB}" != "1" ] && [ "${DHI_REINSTALL_COMBINE}" = "1" ]; then
        rm -f "${flag_file_combine}"
    fi

    if [ ! -f "${flag_file_combine}" ]; then
        mkdir -p "${DHI_CMSSW_BASE}"

        # local env
        if [ "${DHI_REMOTE_JOB}" != "1" ]; then
            # raw CMSSW setup
            (
                echo "installing combine at ${DHI_CMSSW_BASE}/${DHI_CMSSW_VERSION}"
                cd "${DHI_CMSSW_BASE}"
                rm -rf "${DHI_CMSSW_VERSION}"
                scramv1 project CMSSW "${DHI_CMSSW_VERSION}" && \
                cd "${DHI_CMSSW_VERSION}/src" && \
                eval "$( scramv1 runtime -sh )" && \
                scram b
            ) || return "$?"

            # add combine
            (
                cd "${DHI_CMSSW_BASE}/${DHI_CMSSW_VERSION}/src"
                eval "$( scramv1 runtime -sh )" && \
                git clone --branch "${DHI_COMBINE_VERSION}" https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit && \
                cd HiggsAnalysis/CombinedLimit && \
                chmod ug+x test/diffNuisances.py && \
                ( [ "${DHI_COMBINE_PYTHON_ONLY}" = "1" ] && scram b python || scram b -j "${DHI_INSTALL_CORES}" )
            ) || return "$?"
        fi

        # remote env
        if [ "${DHI_REMOTE_JOB}" = "1" ]; then
            # TODO: no need to do this in crab jobs!
            # fetch the bundle and unpack it
            (
                cd "${DHI_CMSSW_BASE}" && \
                scramv1 project CMSSW "${DHI_CMSSW_VERSION}" && \
                cd "${DHI_CMSSW_VERSION}" && \
                law_wlcg_get_file "${DHI_JOB_CMSSW_URIS}" "${DHI_JOB_CMSSW_PATTERN}" "${PWD}/cmssw.tgz" && \
                tar -xzf cmssw.tgz && \
                rm cmssw.tgz && \
                cd "src" && \
                eval "$( scramv1 runtime -sh )" && \
                scram b
            ) || return "$?"
        fi

        date "+%s" > "${flag_file_combine}"
        echo "version ${combine_version}" >> "${flag_file_combine}"
    fi
    export DHI_SOFTWARE_FLAG_FILES="${flag_file_combine}"

    # check the version in the combine flag file and show a warning when there was an update
    if [ "$( cat "${flag_file_combine}" | grep -Po "version \K\d+.*" )" != "${combine_version}" ]; then
        >&2 echo ""
        >&2 echo "WARNING: your local combine installation is not up to date, please consider updating it in a new shell with"
        >&2 echo "         > DHI_REINSTALL_COMBINE=1 source setup.sh $( ${setup_is_default} || echo "${setup_name}" )"
        >&2 echo ""
    fi

    # source it
    cd "${DHI_CMSSW_BASE}/${DHI_CMSSW_VERSION}/src" || return "$?"
    eval "$( scramv1 runtime -sh )" || return "$?"
    export PATH="${PWD}/HiggsAnalysis/CombinedLimit/exe:${PWD}/HiggsAnalysis/CombinedLimit/scripts:${PWD}/HiggsAnalysis/CombinedLimit/test:${PATH}"
    cd "${orig}"


    #
    # minimal local software stack
    #

    # update paths and flags
    local pyv="$( python3 -c "import sys; print('{0.major}.{0.minor}'.format(sys.version_info))" )"
    export PATH="${DHI_BASE}/bin:${DHI_BASE}/dhi/scripts:${DHI_BASE}/modules/law/bin:${DHI_SOFTWARE}/bin:$PATH"
    export PYTHONPATH="${DHI_BASE}:${DHI_BASE}/modules/law:${DHI_BASE}/modules/plotlib:${DHI_SOFTWARE}/lib/python${pyv}/site-packages:${DHI_SOFTWARE}/lib64/python${pyv}/site-packages:${PYTHONPATH}"
    export PYTHONWARNINGS="ignore"
    export PYTHONNOUSERSITE="1"

    # unlimited stack size (as fallback, set soft-limit only)
    ulimit -s unlimited 2> /dev/null
    [ "$?" != "0" ] && ulimit -S -s unlimited

    # local stack
    local sw_version="8"
    local flag_file_sw="${DHI_SOFTWARE}/.sw_good"

    # reset software if requested
    if [ "${DHI_REMOTE_JOB}" != "1" ] && [ "${DHI_REINSTALL_SOFTWARE}" = "1" ]; then
        rm -f "${flag_file_sw}"
    fi

    if [ ! -f "${flag_file_sw}" ]; then
        # local env
        if [ "${DHI_REMOTE_JOB}" != "1" ]; then
            echo "installing software stack at ${DHI_SOFTWARE}"
            rm -rf "${DHI_SOFTWARE}/lib"
            mkdir -p "${DHI_SOFTWARE}"

            # python packages
            dhi_pip_install 'six==1.16.0' || return "$?"
            dhi_pip_install 'luigi==3.2.1' || return "$?"
            dhi_pip_install 'scinum==2.0.2' || return "$?"
            dhi_pip_install 'tabulate==0.9.0' || return "$?"
            dhi_pip_install 'uproot==5.0.5' || return "$?"
            dhi_pip_install 'awkward==2.1.1' || return "$?"
            dhi_pip_install 'mplhep==0.3.31' || return "$?"
            dhi_pip_install 'cvxpy==1.4.1' || return "$?"
            dhi_pip_install 'PyYAML==6.0' || return "$?"
            dhi_pip_install 'mermaidmro==0.2.1' || return "$?"
            dhi_pip_install 'flake8==6.0.0' || return "$?"
            dhi_pip_install 'flake8-commas==2.1.0' || return "$?"
            dhi_pip_install 'flake8-quotes==3.3.2' || return "$?"

            # optional packages, disabled at the moment
            # dhi_pip_install python-telegram-bot==12.3.0
        fi

        # remote env
        if [ "${DHI_REMOTE_JOB}" = "1" ]; then
            # fetch the bundle and unpack it
            (
                cd "${DHI_SOFTWARE}" && \
                law_wlcg_get_file "${DHI_JOB_SOFTWARE_URIS}" "${DHI_JOB_SOFTWARE_PATTERN}" "${PWD}/software.tgz" && \
                tar -xzf "software.tgz" && \
                rm "software.tgz"
            ) || return "$?"
        fi

        date "+%s" > "${flag_file_sw}"
        echo "version ${sw_version}" >> "${flag_file_sw}"
    fi
    export DHI_SOFTWARE_FLAG_FILES="${DHI_SOFTWARE_FLAG_FILES} ${flag_file_sw}"

    # check the version in the sw flag file and show a warning when there was an update
    if [ "$( cat "${flag_file_sw}" | grep -Po "version \K\d+.*" )" != "${sw_version}" ]; then
        >&2 echo ""
        >&2 echo "WARNING: your local software stack is not up to date, please consider updating it in a new shell with"
        >&2 echo "         > DHI_REINSTALL_SOFTWARE=1 source setup.sh $( ${setup_is_default} || echo "${setup_name}" )"
        >&2 echo ""
    fi

    # gfal2 bindings, disabled until we need full gfal2 support
    # however, DHI_LCG_DIR is needed by remote bootstrap script to fetch software bundles
    export DHI_HAS_GFAL="0"
    export DHI_LCG_DIR="${DHI_LCG_DIR:-/cvmfs/grid.cern.ch/centos7-ui-200122}"
    # if [ ! -d "${DHI_LCG_DIR}" ]; then
    #     >&2 echo "lcg directory ${DHI_LCG_DIR} not existing, skip gfal2 bindings setup"
    # else
    #     source "${DHI_LCG_DIR}/etc/profile.d/setup-c7-ui-python3-example.sh" "" || return "$?"
    #     export DHI_HAS_GFAL="1"
    # fi


    #
    # initialze some submodules
    #

    dhi_update_submodules() {
        [ ! -d "${DHI_BASE}/.git" ] && return "1"

        for m in law plotlib; do
            local mpath="${DHI_BASE}/modules/$m"
            # initialize the submodule when the directory is empty
            if [ "$( ls -1q "${mpath}" )" = "0" ]; then
                git submodule update --init --recursive "${mpath}"
            else
                # update when not on a working branch and there are no changes
                local detached_head="$( ( cd "${mpath}"; git symbolic-ref -q HEAD &> /dev/null ) && echo true || echo false )"
                local changed_files="$( cd "${mpath}"; git status --porcelain=v1 2> /dev/null | wc -l )"
                if ! ${detached_head} && [ "${changed_files}" = "0" ]; then
                    git submodule update --init --recursive "${mpath}"
                fi
            fi
        done
    }
    [ ! -z "${BASH_VERSION}" ] && export -f dhi_update_submodules
    dhi_update_submodules


    #
    # law setup
    #

    export LAW_HOME="${DHI_BASE}/.law"
    export LAW_CONFIG_FILE="${DHI_BASE}/law.cfg"

    if which law &> /dev/null; then
        # source law's bash completion scipt
        source "$( law completion )" ""

        # silently index
        law index -q
    fi


    #
    # custom user post-setup hook
    #

    if [ "$( type -t DHI_POST_SETUP )" = "function" ]; then
        echo "calling post setup function"
        DHI_POST_SETUP
    fi
}

interactive_setup() {
    local setup_name="${1:-default}"
    local env_file="${2:-${DHI_BASE}/.setups/${setup_name}.sh}"
    local env_file_tmp="${env_file}.tmp"

    # check if the setup is the default one
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"

    # when the setup already exists and it's not the default one,
    # source the corresponding env file and set a flag to use defaults of missing vars below
    local env_file_exists="false"
    if ! ${setup_is_default}; then
        if [ -f "${env_file}" ]; then
            echo -e "using variables for setup '\x1b[0;49;35m${setup_name}\x1b[0m' from $env_file"
            source "${env_file}" ""
            env_file_exists="true"
        else
            echo -e "no setup file ${env_file} found for setup '\x1b[0;49;35m${setup_name}\x1b[0m'"
        fi
    fi

    variable_exists() {
        local varname="$1"
        eval [ ! -z "\${${varname}+x}" ]
    }

    export_and_save() {
        local varname="$1"
        local value="$2"

        # nothing to do when the env file exists and already contains the value
        if ${env_file_exists} && variable_exists "${varname}"; then
            return "0"
        fi

        # strip " and '
        value=${value%\"}
        value=${value%\'}
        value=${value#\"}
        value=${value#\'}

        if ${env_file_exists}; then
            # write into the existing file
            echo "export ${varname}=\"${value}\"" >> "${env_file}"
        elif ! $setup_is_default; then
            # write into the tmp file
            echo "export ${varname}=\"${value}\"" >> "${env_file_tmp}"
        fi

        # expand and export
        value="$( eval "echo ${value}" )"
        export $varname="${value}"
    }

    query() {
        local varname="$1"
        local text="$2"
        local default="$3"
        local default_text="${4:-${default}}"

        # when the setup is the default one, use the default value when the env variable is empty,
        # otherwise, query interactively
        local value="${default}"
        if ${setup_is_default} || ${env_file_exists}; then
            variable_exists "${varname}" && value="$( eval echo "\$${varname}" )"
        else
            printf "${text} (\x1b[1;49;39m${varname}\x1b[0m, default \x1b[1;49;39m${default_text}\x1b[0m): "
            read query_response
            [ "X${query_response}" = "X" ] && query_response="${default_text}"

            # repeat for boolean flags that were not entered correctly
            while true; do
                ( [ "${default}" != "True" ] && [ "${default}" != "False" ] ) && break
                ( [ "${query_response}" = "True" ] || [ "${query_response}" = "False" ] ) && break
                printf "please enter either '\x1b[1;49;39mTrue\x1b[0m' or '\x1b[1;49;39mFalse\x1b[0m': " query_response
                read query_response
                [ "X${query_response}" = "X" ] && query_response="${default_text}"
            done

            value="${query_response}"
        fi

        export_and_save "${varname}" "${value}"
    }

    # prepare the tmp env file
    if ! ${setup_is_default} && ! ${env_file_exists}; then
        rm -f "${env_file_tmp}"
        mkdir -p "$( dirname "${env_file_tmp}" )"

        echo -e "\nStart querying variables for setup '\x1b[0;49;35m${setup_name}\x1b[0m'"
        echo -e "Please use \x1b[1;49;39mabsolute\x1b[0m values for paths, and press enter to accept default values\n"
    fi

    # start querying for variables
    query DHI_USER "CERN / WLCG username" "$( whoami )"
    export_and_save DHI_USER_FIRSTCHAR "\${DHI_USER:0:1}"
    query DHI_DATA "Local data directory" "${DHI_BASE}/data" "./data"
    query DHI_STORE "Default local output store" "${DHI_DATA}/store" "\$DHI_DATA/store"
    query DHI_STORE_JOBS "Default local store for job files (should not be on /eos when submitting via lxplus!)" "${DHI_STORE}" "\$DHI_STORE"
    query DHI_STORE_BUNDLES "Output store for software bundles when submitting jobs" "${DHI_STORE}" "\$DHI_STORE"
    local eos_user_home="/eos/user/${DHI_USER_FIRSTCHAR}/${DHI_USER}"
    local eos_user_store="${eos_user_home}/dhi/store"
    local eos_user_store_repr=""
    if [ "${DHI_STORE:0:${#eos_user_home}}" = "${eos_user_home}" ]; then
        eos_user_store="${DHI_STORE}"
        eos_user_store_repr="\$DHI_STORE"
    fi
    query DHI_STORE_EOSUSER "Optional output store in EOS user directory" "${eos_user_store}" "${eos_user_store_repr}"
    query DHI_SOFTWARE "Directory for installing software" "${DHI_DATA}/software" "\$DHI_DATA/software"
    query DHI_CMSSW_VERSION "Version of CMSSW to be used" "CMSSW_14_0_0_pre0"
    query DHI_COMBINE_VERSION "Version of combine to be used (tag name)" "14x-comb2023"
    query DHI_DATACARDS_RUN2 "Location of the datacards_run2 repository (optional)" "" "''"
    query DHI_WLCG_CACHE_ROOT "Local directory for caching remote files" "" "''"
    export_and_save DHI_WLCG_USE_CACHE "$( [ -z "${DHI_WLCG_CACHE_ROOT}" ] && echo false || echo true )"
    query DHI_HOOK_FILE "Location of a file with custom hooks (optional)" "" "''"
    query DHI_LOCAL_SCHEDULER "Use a local scheduler for law tasks" "True"
    if [ "$DHI_LOCAL_SCHEDULER" != "True" ]; then
        query DHI_SCHEDULER_HOST "Address of a central scheduler for law tasks" "hh:cmshhcombr2@hh-scheduler1.cern.ch"
        query DHI_SCHEDULER_PORT "Port of a central scheduler for law tasks" "80"
    else
        export_and_save DHI_SCHEDULER_HOST "hh:cmshhcombr2@hh-scheduler1.cern.ch"
        export_and_save DHI_SCHEDULER_PORT "80"
    fi

    # move the env file to the correct location for later use
    if ! ${setup_is_default} && ! ${env_file_exists}; then
        mv "${env_file_tmp}" "${env_file}"
        echo -e "\nsetup variables written to ${env_file}"
    fi
}

action() {
    if setup "$@"; then
        echo -e "\x1b[0;49;35mHH inference tools successfully setup\x1b[0m"
        return "0"
    else
        local code="$?"
        echo -e "\x1b[0;49;31msetup failed with code ${code}\x1b[0m"
        return "${code}"
    fi
}
action "$@"
