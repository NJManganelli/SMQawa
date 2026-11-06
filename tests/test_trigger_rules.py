"""Unit test: common.trigger_rules picks the trigger rules of the dataset's own stream.

The stream is the leading token of the dataset name (digits stripped, so Muon0 -> Muon)
and must equal one of the era's stream names exactly. A substring match would send
MuonEG_Run2024C to the Run2024.Muon rules.
"""
import os

import awkward as ak
import numpy as np
import pytest
import yaml

from qawa.common import trigger_rules

DATA_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'src', 'qawa', 'data')

DATASETS = {
    '2016': {
        'DoubleMuon': ['DoubleMuon_Run2016F-UL2016_MiniAODv2_NanoAODv9-v1', 'DoubleMuon'],
        'SingleMuon': ['SingleMuon_Run2016H-UL2016_MiniAODv2_NanoAODv9-v1', 'SingleMuon'],
        'DoubleEG': ['DoubleEG_Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v1', 'DoubleEG'],
        'SingleElectron': ['SingleElectron_Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v1', 'SingleElectron'],
        'MuonEG': ['MuonEG_Run2016G-UL2016_MiniAODv2_NanoAODv9-v1', 'MuonEG'],
    },
    '2017': {
        'DoubleMuon': ['DoubleMuon_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1', 'DoubleMuon'],
        'SingleMuon': ['SingleMuon_Run2017F-UL2017_MiniAODv2_NanoAODv9-v1', 'SingleMuon'],
        'DoubleEG': ['DoubleEG_Run2017C-UL2017_MiniAODv2_NanoAODv9-v1', 'DoubleEG'],
        'SingleElectron': ['SingleElectron_Run2017E-UL2017_MiniAODv2_NanoAODv9-v1', 'SingleElectron'],
        'MuonEG': ['MuonEG_Run2017D-UL2017_MiniAODv2_NanoAODv9-v1', 'MuonEG'],
    },
    '2018': {
        'DoubleMuon': ['DoubleMuon_Run2018A-UL2018_MiniAODv2_NanoAODv9-v1', 'DoubleMuon'],
        'SingleMuon': ['SingleMuon_Run2018D-UL2018_MiniAODv2_NanoAODv9-v2', 'SingleMuon'],
        'EGamma': ['EGamma_Run2018B-UL2018_MiniAODv2_NanoAODv9-v1', 'EGamma'],
        'MuonEG': ['MuonEG_Run2018C-UL2018_MiniAODv2_NanoAODv9-v1', 'MuonEG'],
    },
    '2024': {
        'Muon': ['Muon0_Run2024C-MINIv6NANOv15-v1', 'Muon1_Run2024D-MINIv6NANOv15-v1', 'Muon0'],
        'EGamma': ['EGamma0_Run2024G-MINIv6NANOv15-v2', 'EGamma1_Run2024D-MINIv6NANOv15-v1', 'EGamma1'],
        'MuonEG': ['MuonEG_Run2024C-MINIv6NANOv15-v1', 'MuonEG_Run2024I-MINIv6NANOv15-v1', 'MuonEG'],
    },
}


class FakeEvent:
    def __init__(self, dataset, hlt):
        self.metadata = {'dataset': dataset}
        self.HLT = ak.zip({k: np.asarray(v, dtype=bool) for k, v in hlt.items()})

    def __len__(self):
        return len(self.HLT)


def load_rules(era):
    with open(os.path.join(DATA_PATH, f'{era}-trigger-rules.yaml')) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


@pytest.mark.parametrize('era,stream,dataset', [
    (era, stream, dataset)
    for era, streams in DATASETS.items()
    for stream, names in streams.items()
    for dataset in names
])
def test_dataset_matches_own_stream(era, stream, dataset):
    # one event per stream, firing only that stream's single pass path
    streams = list(DATASETS[era])
    rules = {f'Run{era}.{s}': {'pass': [f'HLT_{s}'], 'veto': []} for s in streams}
    hlt = {f'HLT_{s}': [i == j for j in range(len(streams))] for i, s in enumerate(streams)}
    passed = trigger_rules(FakeEvent(dataset, hlt), rules, era)
    assert [streams[j] for j in np.flatnonzero(passed)] == [stream]


@pytest.mark.parametrize('era', DATASETS)
def test_real_rules_cover_every_stream(era):
    rules = load_rules(era)
    assert {f'Run{era}.{s}' for s in DATASETS[era]} <= set(rules)


def test_muoneg_2024_gets_muoneg_rules():
    rules = load_rules('2024')['Run2024.MuonEG']
    npass, nveto = len(rules['pass']), len(rules['veto'])
    # events 0-3 each fire one pass path; events 4-23 fire every pass path and one veto path
    hlt = {p: [j == i or j >= npass for j in range(npass + nveto)] for i, p in enumerate(rules['pass'])}
    hlt.update({p: [j == npass + i for j in range(npass + nveto)] for i, p in enumerate(rules['veto'])})
    passed = trigger_rules(FakeEvent('MuonEG_Run2024C-MINIv6NANOv15-v1', hlt), load_rules('2024'), '2024')
    assert (npass, nveto) == (4, 20)
    assert passed.tolist() == [True] * npass + [False] * nveto


def test_unknown_stream_raises():
    with pytest.raises(KeyError):
        trigger_rules(FakeEvent('JetMET0_Run2024C-MINIv6NANOv15-v1', {'HLT_x': [False]}), load_rules('2024'), '2024')
