#!/bin/bash
# SMQawa worker payload wrapper (package data; rendered by qawa.condor.submit).
# Semantics preserved verbatim from the old brewer's script_TEMPLATE: activate the
# venv, export the proxy, run the coffea payload, and FAIL (exit 1) if the output
# histogram pickle is missing so DAGMan RETRY / on_exit_remove can resubmit.
#
# Placeholder tokens are filled with str.format by render_worker_script().
# Runtime args: 1 = jobid (ProcId), 2 = input file.
export X509_USER_PROXY={proxy}
export XRD_REQUESTTIMEOUT=6400
export XRD_REDIRECTLIMIT=64
export INSTALL_LOC_EXTERNAL={install_loc_external}
export COFFEA_IMAGE={coffea_image}
export FULL_IMAGE={full_image}

voms-proxy-info -all
voms-proxy-info -all -file {proxy}

echo "----- COFFEA_IMAGE :"
echo COFFEA_IMAGE $COFFEA_IMAGE
echo FULL_IMAGE $FULL_IMAGE

echo "----- Sourcing virtual environment :"
echo source $INSTALL_LOC_EXTERNAL/.env/bin/activate
source $INSTALL_LOC_EXTERNAL/.env/bin/activate
echo "which python3"
which python3
echo '$PYTHONPATH'
echo $PYTHONPATH
echo awkward, uproot, coffea, qawa versions:
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show coffea
$INSTALL_LOC_EXTERNAL/.env/bin/python3 -m pip show qawa

echo "----- JOB STARTS @" `date "+%Y-%m-%d %H:%M:%S"`
echo "----- X509_USER_PROXY    : $X509_USER_PROXY"
ls -lthr

echo "----- processing the files : "
$INSTALL_LOC_EXTERNAL/.env/bin/python3 brewer-remote-inclusive.py --jobNum=$1 --isMC={ismc} --era={era} --analysis={analysis} --zzdd={zzdd} --infile=$2 --executor={executor} --copyInput {split_by_charge}

echo "----- directory after running :"
ls -lthr
if [ ! -f "histogram_$1.pkl.gz" ]; then
  echo "No output histogram pickle file found";
  exit 1;
fi
echo " ------ THE END (everyone dies !) ----- "
