#!/usr/bin/env bash
# Build the current tree on a host and dump label digests in both modes.
#   bench/run_repro.sh <host>
set -euo pipefail
HOST=$1
SRC=$(cd "$(dirname "$0")/.." && pwd)
rsync -a --delete --exclude .git --exclude build --exclude '*.so' --exclude '*.pyd' \
  --exclude figures --exclude bench_outputs --exclude '_tmp_*' --exclude '*.npz' \
  --exclude '*.png' --exclude '*.gif' --exclude scratch --exclude __pycache__ \
  --exclude .pytest_cache --exclude .venv --exclude venv --exclude '*.egg-info' \
  --exclude test_files/example2.tif "$SRC/" "$HOST:ncolor_det/"
rsync -a "$SRC/bench_outputs/corpus.npz" "$HOST:ncolor_det_corpus.npz"
ssh "$HOST" bash -s <<'EOF'
set -eo pipefail
cd ~/ncolor_det
PY=$HOME/.pyenv/shims/python
[ -x .venv/bin/python ] || { "$PY" -m venv .venv && .venv/bin/python -m pip -q install -U pip; }
.venv/bin/python -m pip -q install numpy pybind11 setuptools setuptools_scm wheel platformdirs
export SETUPTOOLS_SCM_PRETEND_VERSION=2.0.3.dev0 NCOLOR_NO_CALIBRATE=1
rm -rf build src/ncolor/_backend/_impl*.so
.venv/bin/python setup.py build_ext --inplace > build_det.log 2>&1 || { tail -25 build_det.log; exit 1; }
for mode in default det; do
  for t in 0 4; do
    if [ "$mode" = det ]; then export NCOLOR_DETERMINISTIC=1; else unset NCOLOR_DETERMINISTIC; fi
    PYTHONPATH=src .venv/bin/python bench/repro_check.py \
      --corpus ~/ncolor_det_corpus.npz --threads $t --out "repro_${mode}_t${t}.json"
  done
done
EOF
mkdir -p "$SRC/bench_outputs/repro"
rsync -a "$HOST:ncolor_det/repro_*.json" "$SRC/bench_outputs/repro/$(ssh "$HOST" hostname -s)__" 2>/dev/null || \
for f in default_t0 default_t4 det_t0 det_t4; do
  scp -q "$HOST:ncolor_det/repro_${f}.json" "$SRC/bench_outputs/repro/$(ssh "$HOST" hostname -s)__${f}.json"
done
