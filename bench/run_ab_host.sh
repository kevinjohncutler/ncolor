#!/usr/bin/env bash
# Ship both trees and the corpus to a host and run the paired A/B there.
#   bench/run_ab_host.sh <host> <base_tree> <reps>
set -euo pipefail
HOST=$1; BASE_SRC=$2; REPS=${3:-6}
SRC=$(cd "$(dirname "$0")/.." && pwd)
EX=(--exclude .git --exclude build --exclude '*.so' --exclude '*.pyd'
    --exclude figures --exclude bench_outputs --exclude '_tmp_*' --exclude '*.npz'
    --exclude '*.png' --exclude '*.gif' --exclude scratch --exclude __pycache__
    --exclude .pytest_cache --exclude .venv --exclude venv --exclude '*.egg-info'
    --exclude test_files/example2.tif)
ssh "$HOST" 'mkdir -p ~/ncolor_ab/base ~/ncolor_ab/cand ~/ncolor_ab/out'
rsync -a --delete "${EX[@]}" "$BASE_SRC/" "$HOST:ncolor_ab/base/"
rsync -a --delete "${EX[@]}" "$SRC/"      "$HOST:ncolor_ab/cand/"
rsync -a "$SRC/bench_outputs/corpus.npz"  "$HOST:ncolor_ab/corpus.npz"
ssh "$HOST" bash -s -- "$REPS" <<'EOF'
set -eo pipefail
REPS=$1
cd ~/ncolor_ab
PY=$HOME/.pyenv/shims/python
[ -x venv/bin/python ] || { "$PY" -m venv venv && venv/bin/python -m pip -q install -U pip; }
venv/bin/python -m pip -q install numpy pybind11 setuptools setuptools_scm wheel platformdirs
rm -rf out; mkdir -p out
PY=$PWD/venv/bin/python bash cand/bench/ab_bench.sh "$PWD/base" "$PWD/cand" "$PWD/corpus.npz" "$PWD/out" "$REPS"
EOF
mkdir -p "$SRC/bench_outputs/ab/$(basename "$HOST" | cut -d@ -f2)"
rsync -a "$HOST:ncolor_ab/out/" "$SRC/bench_outputs/ab/$(ssh "$HOST" hostname -s)/"
