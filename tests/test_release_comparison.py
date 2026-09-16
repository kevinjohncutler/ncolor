"""Published speed ratios must use compatible, correct benchmark results."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


def _sample(version, timing, valid=True):
    return dict(metadata=dict(version=version, revision=version, corpus={'image': 'same'},
                              cpu='test', platform='test', machine='test', python='test', dependencies={}, threads=4, repeats=15),
                results={'image/default': dict(median_ms=timing, validation={'valid': valid})})


def test_release_summary_excludes_invalid_baseline(tmp_path):
    report = runpy.run_path(str(Path(__file__).parents[1] / 'bench/release_comparison.py'))['report']
    for version, timings in [('current', [2, 3, 100]), ('2.2.0', [6, 9, 300]), ('1.5.3', [30, 40, 50])]:
        for index, timing in enumerate(timings):
            (tmp_path / f'round_{index}_{version}.json').write_text(
                json.dumps(_sample(version, timing, valid=version != '1.5.3')))
    output = tmp_path / 'summary.json'
    report(SimpleNamespace(directory=tmp_path, output=output))
    row = json.loads(output.read_text())['image/default']
    assert row['median_ms']['current'] == 3
    assert row['speedup_vs_current'] == {'2.2.0': 3}
    assert row['median_ms']['1.5.3'] == 40


@pytest.mark.parametrize('field,value', [('corpus', {'image': 'different'}), ('threads', 1),
                                         ('cpu', 'different'), ('platform', 'different'), ('machine', 'different'),
                                         ('revision', 'different')])
def test_release_summary_rejects_mixed_runs(tmp_path, field, value):
    report = runpy.run_path(str(Path(__file__).parents[1] / 'bench/release_comparison.py'))['report']
    first = _sample('current', 3)
    second = _sample('current', 4)
    second['metadata'][field] = value
    for index, data in enumerate([first, second]):
        (tmp_path / f'round_{index}.json').write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        report(SimpleNamespace(directory=tmp_path, output=tmp_path / 'summary.json'))


def test_release_cpu_model_reads_linux_hardware(monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'bench/release_comparison.py'))
    monkeypatch.setattr(namespace['platform'], 'system', lambda: 'Linux')
    monkeypatch.setattr(Path, 'read_text', lambda self: 'processor : 0\nmodel name : Example 64-Core CPU\n')
    assert namespace['cpu_model']() == 'Example 64-Core CPU'


def test_release_cpu_model_handles_unavailable_linux_metadata(monkeypatch):
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'bench/release_comparison.py'))
    monkeypatch.setattr(namespace['platform'], 'system', lambda: 'Linux')
    monkeypatch.setattr(namespace['platform'], 'processor', lambda: '')
    monkeypatch.setattr(namespace['platform'], 'machine', lambda: 'x86_64')
    def unavailable(self):
        raise OSError('unavailable')
    monkeypatch.setattr(Path, 'read_text', unavailable)
    assert namespace['cpu_model']() == 'x86_64'
