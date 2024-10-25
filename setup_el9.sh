#!/bin/bash
export APPTAINER_BINDPATH=/afs,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/cvmfs/grid.cern.ch/etc/grid-security/vomses:/etc/vomses,/eos,/etc/pki/ca-trust,/etc/tnsnames.ora,/run/user,/tmp,/var/run/user,/etc/sysconfig,/etc:/orig/etc
schedd=`myschedd show -j | jq .currentschedd | tr -d '"'`

apptainer -s exec /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cat/cmssw-lxplus/cmssw-el7-lxplus:latest/ sh -c "source /app/setupCondor.sh && export _condor_SCHEDD_HOST=$schedd && export _condor_SCHEDD_NAME=$schedd && export _condor_CREDD_HOST=$schedd && export DHI_EL9_SETUP=${1} && export DHI_REINSTALL_SOFTWARE=${DHI_REINSTALL_SOFTWARE} && export DHI_REINSTALL_COMBINE=${DHI_REINSTALL_COMBINE} && /bin/bash --init-file setup.sh"
