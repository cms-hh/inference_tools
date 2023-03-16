#!/usr/bin/env bash

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    (
        cd "${this_dir}" && \
        flake8 dhi --exclude dhi/tasks/postfit.py,dhi/scripts/postfit_plots.py
    )
}
action "$@"
