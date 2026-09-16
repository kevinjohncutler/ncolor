"""Failures and concurrent cache writes must preserve usable engine state."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

import ncolor
from ncolor import _backend
from ncolor._backend import _smt


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_thread_counts_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        ncolor.Engine(n_threads=value)


@pytest.mark.parametrize('value', [2**32 + 1, 1e100])
def test_thread_counts_do_not_wrap(value):
    with pytest.raises(OverflowError, match='capacity'):
        ncolor.Engine(n_threads=value)


def test_fractional_thread_count_without_cpu_count(monkeypatch):
    import os
    monkeypatch.setattr(os, 'cpu_count', lambda: None)
    assert ncolor.Engine(n_threads=.5).n_threads == 1


def test_extension_copy_publishes_only_complete_content(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source.so', tmp_path / 'cache' / 'target.so'
    content = b'complete binary contents'
    src.write_bytes(content)
    started, finish = threading.Event(), threading.Event()

    def slow_copy(source, temporary):
        temporary.write_bytes(b'partial')
        started.set()
        assert finish.wait(5)
        temporary.write_bytes(content)

    monkeypatch.setattr(_backend.shutil, 'copyfile', slow_copy)
    with ThreadPoolExecutor(1) as pool:
        task = pool.submit(_backend._copy_off_remote, src, dst)
        try:
            assert started.wait(5)
            assert not dst.exists()
        finally:
            finish.set()
        task.result(timeout=5)
    assert dst.read_bytes() == content
    assert list(dst.parent.iterdir()) == [dst]


def test_failed_extension_copy_preserves_previous_cache(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source.so', tmp_path / 'target.so'
    src.write_bytes(b'new binary')
    dst.write_bytes(b'previous binary')

    def failing_copy(source, temporary):
        temporary.write_bytes(b'partial')
        raise OSError('simulated copy failure')

    monkeypatch.setattr(_backend.shutil, 'copyfile', failing_copy)
    with pytest.raises(OSError, match='simulated'):
        _backend._copy_off_remote(src, dst)
    assert dst.read_bytes() == b'previous binary'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['source.so', 'target.so']


def test_concurrent_extension_copies_finish_cleanly(tmp_path):
    src, dst = tmp_path / 'source.so', tmp_path / 'cache' / 'target.so'
    content = bytes(range(256)) * 4096
    src.write_bytes(content)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda _: _backend._copy_off_remote(src, dst), range(12)))
    assert dst.read_bytes() == content
    assert list(dst.parent.iterdir()) == [dst]


def test_mount_detection_respects_path_components(tmp_path, monkeypatch):
    mount = tmp_path / 'share'
    monkeypatch.setattr(_backend.sys, 'platform', 'darwin')
    monkeypatch.setattr(_backend.subprocess, 'check_output',
                        lambda *args, **kwargs: f'server on {mount} (smbfs, rw)')
    assert _backend._on_remote_mount(mount / 'library.so')
    assert not _backend._on_remote_mount(tmp_path / 'share-local' / 'library.so')


@pytest.mark.parametrize('payload', ['null', '[]', '42', '"invalid"',
                                     '{"test": null}', '{"test": "oops"}',
                                     '{"test": 1.5}', '{"test": -3}'])
def test_invalid_calibration_cache_is_ignored(tmp_path, monkeypatch, payload):
    path = tmp_path / 'cache.json'
    path.write_text(payload)
    monkeypatch.setattr(_smt, 'CACHE_PATH', path)
    monkeypatch.setattr(_smt, '_cache_key', lambda: 'test')
    monkeypatch.setattr(_smt, '_physical_cores', lambda: 2)
    assert _smt._load_cache() == {}
    assert _smt.auto_threads() == 2


def test_failed_calibration_publication_keeps_valid_json(tmp_path, monkeypatch):
    path = tmp_path / 'cache.json'
    path.write_text('{"test": 2}')
    monkeypatch.setattr(_smt, 'CACHE_PATH', path)

    def fail(*args):
        raise OSError('simulated replace failure')

    monkeypatch.setattr(_smt.os, 'replace', fail)
    with pytest.raises(OSError, match='simulated'):
        _smt._save_cache({'test': 4})
    assert json.loads(path.read_text()) == {'test': 2}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('high,size', [(2**31 - 1, 8), (70000, 25000)])
@pytest.mark.parametrize('threads', [1, 4])
def test_sparse_adjacency_handles_signed_labels(high, size, threads):
    engine = ncolor.Engine(n_threads=threads)
    image = np.resize(np.array([-1, 1, high, 0], np.int32), (2, size))
    expected = {(-1, 1), (1, high)}
    for _ in range(3):
        assert {tuple(pair) for pair in engine.connect(image)} == expected
    engine.release_buffers()
    assert {tuple(pair) for pair in engine.connect(image)} == expected


def test_parallel_label_presence_matches_serial_under_contention():
    rng = np.random.default_rng(733)
    image = rng.choice(np.array([0, 2, 11, 100, 3000], np.int32), (1024, 1024))
    serial = ncolor.Engine(n_threads=1)
    parallel = ncolor.Engine(n_threads=4)
    expected = serial.format_labels(image)
    for _ in range(8):
        np.testing.assert_array_equal(parallel.format_labels(image), expected)


@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('shape', [(1, 5), (1, 1, 5)])
def test_negative_only_adjacency_keeps_source_ids(wrap, shape):
    image = np.array([-1, -2, 0, -3, -4], np.int32).reshape(shape)
    expected = {(-2, -1), (-4, -3)}
    if wrap:
        expected.add((-4, -1))
    assert {tuple(pair) for pair in ncolor.Engine(n_threads=2)._solver.connect(image, wrap=wrap)} == expected


@pytest.mark.parametrize('weight', [0, 1, -1])
def test_contact_filter_applies_at_unit_radius(weight):
    image = np.array([[1, 2]], np.int32)
    out, count = ncolor.label(image, expand=False, soft_conn=0, soft_radius=0,
        min_contact=2, weight_objective=weight, return_n=True)
    assert count == 1
    assert out[0, 0] == out[0, 1]


def test_explicit_empty_soft_edges_disable_automatic_preferences():
    engine = ncolor.Engine(n_threads=1)
    image = np.array([[1, 0, 2]], np.int32)
    out, count = engine.label(image, expand=False,
        soft_extra_edges=np.empty((0, 2), np.int32), return_n=True)
    assert count == 1
    assert out[0, 0] == out[0, 2]
    assert engine._solver.get_last_soft_pairs().size == 0


def test_filtered_hard_contacts_can_remain_soft_preferences():
    engine = ncolor.Engine(n_threads=1)
    engine.label(np.array([[1, 2]], np.int32), expand=False,
                 min_contact=2, soft_conn=1, soft_radius=1)
    assert engine._solver.get_last_soft_pairs().tolist() == [[1, 2]]


@pytest.mark.parametrize('failure', [PermissionError, FileExistsError])
@pytest.mark.parametrize('winner', [None, b'short', b'corrupt payload', b'complete binary'])
def test_extension_publication_race_requires_complete_winner(tmp_path, monkeypatch, failure, winner):
    src, dst = tmp_path / 'source.so', tmp_path / 'cache' / 'target.so'
    src.write_bytes(b'complete binary')
    dst.parent.mkdir()
    if winner is not None:
        dst.write_bytes(winner)
    def collide(*args):
        raise failure('concurrent publication')
    monkeypatch.setattr(_backend.os, 'replace', collide)
    monkeypatch.setattr(_backend.time, 'sleep', lambda _: None)
    if winner == b'complete binary':
        _backend._copy_off_remote(src, dst)
        assert dst.read_bytes() == src.read_bytes()
    else:
        with pytest.raises(failure, match='concurrent publication'):
            _backend._copy_off_remote(src, dst)
    assert sorted(p.name for p in dst.parent.iterdir()) == ([] if winner is None else ['target.so'])


@pytest.mark.parametrize('stale_target', [False, True])
def test_extension_publication_uses_bytes_when_metadata_lags(tmp_path, monkeypatch, stale_target):
    src, dst = tmp_path / 'source.so', tmp_path / 'target.so'
    src.write_bytes(b'complete binary')
    dst.write_bytes(src.read_bytes())
    real_stat = Path.stat
    def stale_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        is_stale = (path == dst) if stale_target else path.name.startswith('.ncolor-')
        if is_stale:
            values = list(result)
            values[6] = 0
            return os.stat_result(values)
        return result
    def collide(*args):
        raise FileExistsError('concurrent publication')
    monkeypatch.setattr(Path, 'stat', stale_stat)
    monkeypatch.setattr(_backend.os, 'replace', collide)
    _backend._copy_off_remote(src, dst)
    assert dst.read_bytes() == src.read_bytes()



def test_extension_publication_waits_for_delayed_visibility(tmp_path, monkeypatch):
    src, dst = tmp_path / 'source.so', tmp_path / 'target.so'
    src.write_bytes(b'complete binary')
    dst.write_bytes(b'')
    def collide(*args):
        raise FileExistsError('concurrent publication')
    delays = []
    def publish_after_delay(delay):
        delays.append(delay)
        dst.write_bytes(src.read_bytes())
    monkeypatch.setattr(_backend.os, 'replace', collide)
    monkeypatch.setattr(_backend.time, 'sleep', publish_after_delay)
    _backend._copy_off_remote(src, dst)
    assert delays == [0.05]
    assert dst.read_bytes() == src.read_bytes()
