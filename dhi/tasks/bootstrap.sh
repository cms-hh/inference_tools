#!/usr/bin/env bash

action() {
    # on the CERN HTCondor batch, the PATH variable is changed even though "getenv" is set
    # in the job file, so set the PATH manually to the desired same value
    export PATH="{{dhi_env_path}}"
    # set the HMC_ON_HTCONDOR which is recognized by the setup script below
    export DHI_ON_HTCONDOR="1"
    echo "Setting up environment"
    # source the main setup
    source "{{dhi_base}}/setup.sh" ""
    echo "Done."
}
action "$@"
