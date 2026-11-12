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

from qawa.common import is_reference_stream, lepton_triggers, reference_triggers, trigger_rules

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
    '2025': {
        'Muon': ['Muon0_Run2025C', 'Muon1_Run2025G-PromptReco-v1', 'Muon0'],
        'EGamma': ['EGamma0_Run2025D-PromptReco-v1', 'EGamma1_Run2025F', 'EGamma1'],
        'MuonEG': ['MuonEG_Run2025C-PromptReco-v1', 'MuonEG_Run2025G', 'MuonEG'],
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


REFERENCE_PATHS = {
    '2016': ['PFMET100_PFMHT100_IDTight', 'PFMET110_PFMHT110_IDTight', 'PFMET120_PFMHT120_IDTight',
             'PFMETNoMu100_PFMHTNoMu100_IDTight', 'PFMETNoMu110_PFMHTNoMu110_IDTight',
             'PFMETNoMu120_PFMHTNoMu120_IDTight', 'PFMET170_HBHECleaned'],
    '2017': [f'PFMET{t}_PFMHT{t}_IDTight' for t in (110, 120, 130, 140)]
            + ['PFMET120_PFMHT120_IDTight_PFHT60']
            + [f'PFMETNoMu{t}_PFMHTNoMu{t}_IDTight' for t in (110, 120, 130, 140)]
            + ['PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60']
            + [f'PFMETTypeOne{t}_PFMHT{t}_IDTight' for t in (110, 120, 130, 140)]
            + ['PFMETTypeOne120_PFMHT120_IDTight_PFHT60',
               'PFMET200_HBHE_BeamHaloCleaned', 'PFMETTypeOne200_HBHE_BeamHaloCleaned'],
    '2024': [f'PFMET{t}_PFMHT{t}_IDTight' for t in (120, 130, 140)]
            + ['PFMET120_PFMHT120_IDTight_PFHT60']
            + [f'PFMETNoMu{t}_PFMHTNoMu{t}_IDTight' for t in (120, 130, 140)]
            + ['PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60', 'PFMETTypeOne140_PFMHT140_IDTight',
               'PFMET200_BeamHaloCleaned', 'PFMETTypeOne200_BeamHaloCleaned'],
}
REFERENCE_PATHS['2018'] = REFERENCE_PATHS['2017']
REFERENCE_PATHS['2025'] = REFERENCE_PATHS['2024']
REFERENCE_STREAMS = {'2016': ['MET'], '2017': ['MET'], '2018': ['MET'], '2024': ['JetMET'], '2025': ['JetMET']}


@pytest.mark.parametrize('era', DATASETS)
def test_reference_key_per_era(era):
    reference = load_rules(era)[f'Run{era}.Reference']
    assert set(reference) == {'streams', 'pass'}
    assert reference['streams'] == REFERENCE_STREAMS[era]
    assert reference['pass'] == REFERENCE_PATHS[era]


@pytest.mark.parametrize('era,dataset,expected', [
    ('2024', 'JetMET0_Run2024C-MINIv6NANOv15-v1', True),
    ('2024', 'JetMET1_Run2024I-MINIv6NANOv15-v1', True),
    ('2025', 'JetMET0_Run2025C-PromptReco-v1', True),
    ('2025', 'JetMET1_Run2025G', True),
    ('2016', 'MET_Run2016F-UL2016_MiniAODv2_NanoAODv9-v1', True),
    ('2017', 'MET_Run2017B-UL2017_MiniAODv2_NanoAODv9-v1', True),
    ('2018', 'MET_Run2018D-UL2018_MiniAODv2_NanoAODv9-v2', True),
    ('2024', 'Muon0_Run2024C-MINIv6NANOv15-v1', False),
    ('2024', 'EGamma1_Run2024D-MINIv6NANOv15-v1', False),
    ('2024', 'MuonEG_Run2024C-MINIv6NANOv15-v1', False),
    ('2018', 'SingleMuon_Run2018D-UL2018_MiniAODv2_NanoAODv9-v2', False),
    ('2016', 'DoubleEG_Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v1', False),
])
def test_is_reference_stream(era, dataset, expected):
    assert is_reference_stream(dataset, load_rules(era), era) is expected


def test_reference_and_lepton_or():
    rules = {'Run2024.Reference': {'streams': ['JetMET'], 'pass': ['MET_a', 'MET_b', 'MET_missing']},
             'Run2024.MC': {'pass': ['Lep_a', 'Lep_b'], 'veto': []}}
    hlt = {'MET_a': [True, False, False, False], 'MET_b': [False, True, False, False],
           'Lep_a': [False, False, True, False], 'Lep_b': [True, False, False, False]}
    event = FakeEvent('JetMET0_Run2024C-MINIv6NANOv15-v1', hlt)
    assert reference_triggers(event, rules, '2024').tolist() == [True, True, False, False]
    assert lepton_triggers(event, rules, '2024').tolist() == [True, False, True, False]


@pytest.mark.parametrize('era', DATASETS)
def test_flashsim_paths_skip_stream_names(era):
    from qawa.flashsim_shim import _trigger_paths
    paths = set(_trigger_paths(era))
    assert not paths & set(REFERENCE_STREAMS[era])
    assert set(REFERENCE_PATHS[era]) <= paths
