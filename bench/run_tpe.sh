#!/usr/bin/env bash
set -euo pipefail
HOST=$1; shift
SIZES=${*:-"512 1024"}
SRC=$(cd "$(dirname "$0")/.." && pwd)
rsync -a --delete --exclude .git --exclude build --exclude '*.so' --exclude '*.pyd' \
  --exclude figures --exclude bench_outputs --exclude '_tmp_*' --exclude '*.npz' \
  --exclude '*.png' --exclude '*.gif' --exclude scratch --exclude __pycache__ \
  --exclude .pytest_cache --exclude .venv --exclude venv --exclude '*.egg-info' \
  --exclude test_files/example2.tif "$SRC/" "$HOST:ncolor_bench/"
ssh "$HOST" bash -s -- $SIZES <<'EOF'
set -eo pipefail
SIZES="$*"
cd ~/ncolor_bench
PY=${PYBIN:-$HOME/.pyenv/shims/python}
[ -x .venv/bin/python ] || { "$PY" -m venv .venv && .venv/bin/python -m pip -q install -U pip; }
export SETUPTOOLS_SCM_PRETEND_VERSION=2.0.3.dev0 NCOLOR_NO_CALIBRATE=1
rm -rf build src/ncolor/_backend/_impl*.so
.venv/bin/python setup.py build_ext --inplace > build_tpe.log 2>&1 || { tail -30 build_tpe.log; exit 1; }
for n in $SIZES; do
  NCOLOR_SWEEP_SIZE=$n PYTHONPATH=src .venv/bin/python bench/threads_per_engine.py
done
EOF
