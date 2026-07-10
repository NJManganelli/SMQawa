#!/usr/bin/env zsh

if [[ "$1" == "zsh" ]]; then
    cat <<EOF > shell
#!/usr/bin/env zsh
autoload bashcompinit
bashcompinit
source SMQawa/call_host.zsh

export INSTALL_LOC_EXTERNAL=\$PWD
export INSTALL_LOC=/srv/
export ZDOTDIR=\$INSTALL_LOC

EOF
else
    cat <<EOF > shell
#!/usr/bin/env bash
source SMQawa/call_host.sh

export INSTALL_LOC_EXTERNAL=\$PWD
export INSTALL_LOC=/srv/

EOF
fi

if [[ "$2" == "lpc" ]]; then
    cat <<EOF >> shell
# Needed to setup cluster for LPC
export CONDOR_CONFIG=\$INSTALL_LOC.condor_config
export APPTAINERENV_CONDOR_CONFIG=\$CONDOR_CONFIG
grep -v '^include' /etc/condor/config.d/01_cmslpc_interactive > .condor_config

# Need all our bind addresses
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/eos,/etc/pki/ca-trust,/run/user,/var/run/user,\$(readlink -f \$PWD)

EOF
else
    cat <<EOF >> shell
# Needed for the HTCondor python bindings inside the container (lxplus):
# snapshot the host's fully-resolved condor config (includes and the
# myschedd-assigned SCHEDD_HOST already baked in) into the bound workdir.
# LOCAL_CONFIG_* are filtered so the container doesn't re-read host-only paths.
condor_config_val -dump 2>/dev/null | grep -Ev '^(LOCAL_CONFIG_FILE|LOCAL_CONFIG_DIR|REQUIRE_LOCAL_CONFIG_FILE)\b' > .condor_config
echo 'REQUIRE_LOCAL_CONFIG_FILE = false' >> .condor_config
export CONDOR_CONFIG=\$INSTALL_LOC.condor_config
export APPTAINERENV_CONDOR_CONFIG=\$CONDOR_CONFIG

# Need all our bind addresses
export APPTAINER_BINDPATH=/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/eos,/etc/pki/ca-trust,/etc/tnsnames.ora,/run/user,/var/run/user,\$(readlink -f \$PWD)

EOF
fi

cat <<EOF >> shell
voms-proxy-init -voms cms --valid 192:00 --out \$HOME/x509up_u\$UID
export X509_USER_PROXY=\$HOME/x509up_u\$UID

if [[ "\$1" == "" ]]; then
  export COFFEA_IMAGE="coffeateam/coffea-dask-almalinux9:2025.12.0-py3.12"
else
  export COFFEA_IMAGE="\$1"
fi

export FULL_IMAGE="/cvmfs/unpacked.cern.ch/registry.hub.docker.com/"\$COFFEA_IMAGE
EOF

if [[ "$1" == "zsh" ]]; then
    cat <<EOF >> shell
SINGULARITY_SHELL=\$(which zsh) apptainer exec -B \${PWD}:/srv --pwd /srv \${FULL_IMAGE} $(which zsh)
EOF
else
    cat <<EOF >> shell
SINGULARITY_SHELL=\$(which bash) apptainer exec -B \${PWD}:/srv --pwd /srv \${FULL_IMAGE} $(which bash) --rcfile /srv/.bashrc
EOF
fi

if [[ "$1" == "zsh" ]]; then
    cat <<EOF > .zshrc
if [ ! -d "SMQawa" ]; then
  echo "SMQawa must already be cloned, e.g. via 'git clone -b <branch> git@github.com:<githubusername>/SMQawa.git'"
  echo "the bootstrap.zsh script should be run from the parent folder of SMQawa to allow editable install of coffea and other packages alongside it."
  echo "clean the virtual env before re-attempting install."
fi
# Source the call_host script again inside the container
source SMQawa/call_host.zsh

# To get dasgoclient
export PATH=\$PATH:/cvmfs/cms.cern.ch/common
export PYTHONPATH=\$INSTALL_LOC.env/bin

export XRDPARALLELEVTLOOP=16 #This might only work in development environments, but should increase the throughput...
# export INSTALL_LOC=\$PWD/ #this could potentially be VIRTUAL_ENV, but creating that prior to activation may cause unforeseen problems... note TRAILING slash
if [[ -z "\$INSTALL_LOC" ]]; then
  echo "INSTALL_LOC not set, check shell script or export the variable for where the virtual environment should be installed/found"
else
  echo "INSTALL_LOC=" \$INSTALL_LOC
fi

test -e \${ZDOTDIR}/.iterm2_shell_integration.zsh && source \${ZDOTDIR}/.iterm2_shell_integration.zsh

patch_venv_pths() {
    echo Patching VIRTUAL_ENV .pth files
    for x in \$(find \$VIRTUAL_ENV -name "*.pth"); do
        echo \${x} && sed -i "s@\$INSTALL_LOC@\$INSTALL_LOC_EXTERNAL/@g" \${x};
    done
}

install_env() {
  print "INSTALLING ENV"
  # This will break if the repo isn't cloned first
  set -e
  echo "Installing shallow virtual environment in \$PWD/.env..."
  python -m venv --without-pip --system-site-packages \$INSTALL_LOC.env
  source \$INSTALL_LOC.env/bin/activate
  unlink \$INSTALL_LOC.env/lib64  # HTCondor can't transfer symlink to directory and it appears optional
  cd \${INSTALL_LOC}
  # if [ ! -d "coffea" ]; then
  #   echo "Cloning coffea for editable install"
  #   git clone -b smqawa-wz2ltaunu http://github.com/NJManganelli/coffea.git
  # fi
  # cd coffea
  # \$INSTALL_LOC.env/bin/python -m pip install -e .
  # cd ..
  cd SMQawa
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  cd ..
  \$INSTALL_LOC.env/bin/python -m pip install --upgrade 'hist >= 2.10.0' 'boost_histogram >= 1.7.1' 'mplhep >= 1.0.0'
  if [ ! -d "DCTools" ]; then
    echo "DCTools should be cloned into the directory adjacent to SMQawa to enable combine card building and postfit plotting"
    echo "e.g. git clone -b main git@github.com:yhaddad/DCTools.git for main branch"
    echo "For current development branch, git clone -b VZ_main git@github.com:NJManganelli/DCTools.git"
  fi
  tar -xzf /afs/cern.ch/user/n/nmangane/public/vzmain_2024jecs_temp.tar.gz # TEMP PATCH UNTIL CORRECTIONLIB JECS ARE INTEGRATED
  echo "done."
}

install_kernel() {
  # work around issues copying CVMFS xattr when copying to tmpdir
  export TMPDIR=\$(mktemp -d -p .)
  \$INSTALL_LOC.env/bin/python -m ipykernel install --user --name smqawa --display-name "smqawa" --env PYTHONPATH \$PYTHONPATH:\$PWD --env PYTHONNOUSERSITE 1
  rm -rf \$TMPDIR && unset TMPDIR
}

install_all() {
  install_env
  install_kernel
  # source \$INSTALL_LOC.env/bin/activate
}

export JUPYTER_PATH=\$INSTALL_LOC.jupyter
export JUPYTER_RUNTIME_DIR=\$INSTALL_LOC.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=\$INSTALL_LOC.local/share/jupyter
export IPYTHONDIR=\$INSTALL_LOC.ipython
unset GREP_OPTIONS

[[ -d \$INSTALL_LOC.env ]] || install_all
source \$INSTALL_LOC.env/bin/activate
patch_venv_pths
alias pip="python -m pip"

EOF
    curl -L https://iterm2.com/shell_integration/zsh -o ~/.iterm2_shell_integration.zsh
    mv shell zsh-shell
    chmod u+x zsh-shell .zshrc
    echo "Wrote zsh-shell and .zshrc to current directory. Run ./zsh-shell to start the apptainer shell"
else
    cat <<EOF > .bashrc
if [ ! -d "SMQawa" ]; then
  echo "SMQawa must already be cloned, e.g. via 'git clone -b <branch> git@github.com:<githubusername>/SMQawa.git'"
  echo "the bootstrap.zsh script should be run from the parent folder of SMQawa to allow editable install of coffea and other packages alongside it."
  echo "clean the virtual env before re-attempting install."
fi
# Source the call_host script again inside the container
source SMQawa/call_host.sh

# To get dasgoclient
export PATH=\$PATH:/cvmfs/cms.cern.ch/common
export PYTHONPATH=\$INSTALL_LOC.env/bin

export XRDPARALLELEVTLOOP=16 #This might only work in development environments, but should increase the throughput...
# export INSTALL_LOC=\$PWD/ #this could potentially be VIRTUAL_ENV, but creating that prior to activation may cause unforeseen problems... note TRAILING slash
if [[ -z "\$INSTALL_LOC" ]]; then
  echo "INSTALL_LOC not set, check shell script or export the variable for where the virtual environment should be installed/found"
else
  echo "INSTALL_LOC=" \$INSTALL_LOC
fi

patch_venv_pths() {
    echo Patching VIRTUAL_ENV .pth files
    for x in \$(find \$VIRTUAL_ENV -name "*.pth"); do
        echo \${x} && sed -i "s@\$INSTALL_LOC@\$INSTALL_LOC_EXTERNAL/@g" \${x};
    done
}

install_env() {
  # This will break if the repo isn't cloned first
  set -e
  echo "Installing shallow virtual environment in \$INSTALL_LOC.env..."
  python -m venv --without-pip --system-site-packages \$INSTALL_LOC.env
  source \$INSTALL_LOC.env/bin/activate
  unlink \$INSTALL_LOC.env/lib64  # HTCondor can't transfer symlink to directory and it appears optional
  cd \${INSTALL_LOC}
  # if [ ! -d "coffea" ]; then
  #   echo "Cloning coffea for editable install"
  #   git clone -b smqawa-wz2ltaunu http://github.com/NJManganelli/coffea.git
  # fi
  # cd coffea
  # \$INSTALL_LOC.env/bin/python -m pip install -e .
  # cd ..
  cd SMQawa
  \$INSTALL_LOC.env/bin/python -m pip install -e .
  cd ..
  \$INSTALL_LOC.env/bin/python -m pip install --upgrade 'hist >= 2.10.0' 'boost_histogram >= 1.7.1' 'mplhep >= 1.0.0'
  if [ ! -d "DCTools" ]; then
    echo "DCTools should be cloned into the directory adjacent to SMQawa to enable combine card building and postfit plotting"
    echo "e.g. git clone -b main git@github.com:yhaddad/DCTools.git for main branch"
    echo "For current development branch, git clone -b VZ_main git@github.com:NJManganelli/DCTools.git"
  fi
  tar -xzf /afs/cern.ch/user/n/nmangane/public/vzmain_2024jecs_temp.tar.gz # TEMP PATCH UNTIL CORRECTIONLIB JECS ARE INTEGRATED
  echo "done."
}

install_kernel() {
  # work around issues copying CVMFS xattr when copying to tmpdir
  export TMPDIR=\$(mktemp -d -p .)
  \$INSTALL_LOC.env/bin/python -m ipykernel install --user --name smqawa --display-name "smqawa" --env PYTHONPATH \$PYTHONPATH:\$PWD --env PYTHONNOUSERSITE 1
  rm -rf \$TMPDIR && unset TMPDIR
}

install_all() {
  install_env
  install_kernel
  # source \$INSTALL_LOC.env/bin/activate
}

export JUPYTER_PATH=\$INSTALL_LOC.jupyter
export JUPYTER_RUNTIME_DIR=\$INSTALL_LOC.local/share/jupyter/runtime
export JUPYTER_DATA_DIR=\$INSTALL_LOC.local/share/jupyter
export IPYTHONDIR=\$INSTALL_LOC.ipython
unset GREP_OPTIONS

[[ -d .env ]] || install_all
source \$INSTALL_LOC.env/bin/activate
patch_venv_pths
alias pip="python -m pip"

EOF
    mv shell bash-shell
    chmod u+x bash-shell .bashrc
    echo "Wrote bash-shell and .bashrc to current directory. Run ./bash-shell to start the apptainer shell"
fi
