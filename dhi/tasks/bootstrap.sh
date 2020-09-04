#!/usr/bin/env bash

action() {
    # on the CERN HTCondor batch, the PATH variable is changed even though "getenv" is set
    # in the job file, so set the PATH manually to the desired same value
    echo "HERE1"
    export PATH="{{dhi_env_path}}"
    echo "HERE2"
    # set the HMC_ON_HTCONDOR which is recognized by the setup script below
    export DHI_ON_HTCONDOR="1"
    echo "HERE3"

    # source the main setup
    source "{{dhi_base}}/setup.sh" ""
    echo "HERE4"
}
action "$@"
