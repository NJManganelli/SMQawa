# Installation
git clone this project into a folder which will include several other repos, then execute the bootstrap file with arguments `<shell>` and `<location>`.
`<shell>` is one of `zsh` or `bash`, matching your default shell
`<location>` is one of `lpc` or `lxplus` and must match where you're working from to setup bind paths correctly

## One-time Setup
Perform initial git clone, and use bootstrap.zsh to create shell and (.zshrc or .bashrc) profiles.
The shell executable sets up environment variables and sources the singularity image (with a default value that can be overridden with a different version for quick testing)
The profiles implicitly create the python virtual environment when none is already present (`.env`), then install coffea and SMQawa in editable mode, and otherwise source the environment if `.env` exists (so for reinstallation of code, you may need to delete the `.env` to trigger a reinstall via the profile).

```bash
export INSTALL_LOC_EXTERNAL=$PWD/VZAnalysis
mkdir -p $INSTALL_LOC_EXTERNAL
cd $INSTALL_LOC_EXTERNAL
git clone -b VZ_main git@github.com:<githubusername>/DCTools.git
git clone -b VZ_main git@github.com:<githubusername>/SMQawa.git
zsh SMQawa/bootstrap.zsh <shell> <location>
```

## Every-time setup
Navigate to the installation location (where shell and .*rc files are created, parent folder of SMQawa) and run for your preferred shell
```bash
./bash-shell
```
```zsh
./zsh-shell
```

## Executing SMQawa code locally
```bash
cd SMQawa
python brewer-remote-inclusive.py --isMC=1 --era=2018 --infile=/store/mc/RunIISummer20UL16NanoAODAPVv9/DYJetsToLL_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/NANOAODSIM/106X_mcRun2_asymptotic_preVFP_v11-v1/130000/9FBF4AF1-D77E-4648-916B-097401C9544B.root --executor=IterativeExecutor --copyInput
```

## Merging histograms
```bash
python hist-merger.py --dir <dir with jobs_<tag>_<era>_*> -t <tag> --era <era> --outdir <out>
python hist-merger.py --eos [--dir /eos/user/u/user/WZtotau2lnu] -t <tag> --era <era> --sample 'WZ*'
```
The tag is matched exactly (`-t WZ` does not read `jobs_WZ_DD_*`) and names the output. To read more directories,
give a glob for the tag part with `--match`, e.g. `-t WZ --match '*WZ*'`; the output is still named after `-t`.
The output is `merged-histogram-Inc-<tag>-<era>-<suffix>.pkl.gz`, with `-f` setting the suffix (default `latest`).
`--ncores` sets the worker processes (default 4). Each holds a partial merge and a decompressed job file, so when
the largest job file is 10 MiB or more (EFT-size outputs) the merge runs in the main process unless `--ncores` is
given. Unreadable or unmergeable job files are an error naming the file (and the dataset and histogram that
disagree); empty ones are skipped and reported. A worker that dies, for example killed for memory, ends the merge
with an error. A dataset's `lumi` record (processed run and luminosity-block pairs) is merged as a union.
`--splitByKind` writes the Data, MC-backgrounds and MC-signal datasets (and EFT ones) to separate files
`...-<suffix>-<part>.pkl.gz`, using each dataset's `kind` (data, mc, signal, eft). For job files without `kind`,
`--kindsFrom CONFIG.yaml [...]` takes it from DCTools input configs, whose `groups:` list `processes:` with
`type: signal|background|data` (signal, mc, data; a dataset in both a data group and a background group, as in the
data-driven DY group, is data). A dataset's own `kind` must agree with the configs (EFT kinds come only from the job
files), a dataset without `kind` that no config lists is an error, and a kind taken from a config must give a negative
`sumw` exactly for data. The files of one merge are replaced together, after all of them were written, and a part with
no datasets never replaces an existing file. On a terminal the merger shows its progress per directory, file
and chunk and the memory of all its processes; `--noProgress` turns this off, and it is off in batch logs. Every run ends with a line giving the elapsed time and
the peak memory (RSS) of the main process and of the largest worker.
