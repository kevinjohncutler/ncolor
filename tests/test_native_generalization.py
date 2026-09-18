"""Check internal memory and numeric contracts against the shipped headers."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


def test_native_generalization(tmp_path):
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip('a C++17 compiler is needed for the native contract tests')
    if Path(compiler[0]).stem.lower() in ('cl', 'clang-cl'):
        pytest.skip('this optional harness uses the Unix compiler driver')
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / 'native_generalization'
    # The threadpool parks workers on WaitOnAddress. MSVC picks the import
    # library up from a pragma in the header; the GNU driver needs it named.
    libraries = ['-lsynchronization'] if os.name == 'nt' else []
    build = subprocess.run(
        compiler + ['-std=c++17', '-O2', '-pthread', '-I', str(root / 'cpp'),
                    str(root / 'tests/native_generalization.cpp'), '-o', str(executable)]
        + libraries,
        capture_output=True, text=True, timeout=120)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
