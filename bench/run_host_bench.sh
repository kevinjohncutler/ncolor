#!/usr/bin/env bash
# Build ncolor from the current tree on a remote host and benchmark it.
#
#   bench/run_host_bench.sh <host> <tag> [--quick] [ENV=VAL ...]
#
# The tree is rsynced to ~/ncolor_bench on the host (build artifacts and
# scratch excluded), built in place inside a private venv there, and
# bench/bench_hosts.py is run with PYTHONPATH=src. The JSON result is
# copied back to bench_outputs/hosts/ here. Extra ENV=VAL arguments are
# exported for the build (e.g. NCOLOR_MARCH_NATIVE=0 NCOLOR_MARCH=x86-64-v2
# NCOLOR_USE_CLANG_CL=1). PYBIN=/path/to/python overrides the interpreter
# used to create the venv (default: the host's pyenv shim).
set -euo pipefail

HOST=$1; TAG=$2; shift 2
# ssh flattens its arguments into one string, so an empty argument would
# vanish; pass a fixed token either way.
QUICK="full"
if [ "${1:-}" = "--quick" ]; then QUICK="quick"; shift; fi
# SRC_DIR overrides the tree to benchmark (e.g. a saved baseline snapshot).
SRC=${SRC_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

rsync -a --delete \
  --exclude .git --exclude build --exclude '*.so' --exclude '*.pyd' \
  --exclude figures --exclude bench_outputs --exclude '_tmp_*' \
  --exclude '*.npz' --exclude '*.png' --exclude '*.gif' --exclude scratch \
  --exclude __pycache__ --exclude .pytest_cache --exclude .venv --exclude venv \
  --exclude '*.egg-info' --exclude test_files/example2.tif \
  "$SRC/" "$HOST:ncolor_bench/"

# No ``set -u`` on the remote side: macOS ships bash 3.2, where an empty
# "$@" under -u is itself an error.
ssh "$HOST" bash -s -- "$TAG" "$QUICK" "$@" <<'EOF'
set -eo pipefail
TAG=$1; QUICK=$2; shift 2
if [ "$QUICK" = "quick" ]; then QUICK="--quick"; else QUICK=""; fi
cd ~/ncolor_bench
for kv in "$@"; do export "$kv"; done
PY=${PYBIN:-$HOME/.pyenv/shims/python}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/python -m pip -q install -U pip
fi
# Idempotent: a no-op once satisfied. platformdirs is a runtime dep of
# the package; the rest are build deps.
.venv/bin/python -m pip -q install numpy pybind11 setuptools setuptools_scm wheel platformdirs
export SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-2.0.3.dev0}
export NCOLOR_NO_CALIBRATE=1
rm -rf build src/ncolor/_backend/_impl*.so
CC_USED=$(.venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var('CC'))")
.venv/bin/python setup.py build_ext --inplace > build_${TAG}.log 2>&1 || { tail -40 build_${TAG}.log; exit 1; }
echo "[$(hostname)] built $TAG with CC=$CC_USED ${NCOLOR_MARCH_NATIVE:+MARCH_NATIVE=$NCOLOR_MARCH_NATIVE} ${NCOLOR_MARCH:+MARCH=$NCOLOR_MARCH}"
export NCOLOR_BENCH_COMPILER="$CC_USED"
PYTHONPATH=src .venv/bin/python bench/bench_hosts.py --tag "$TAG" --out bench_outputs/hosts $QUICK
EOF

mkdir -p "$SRC/bench_outputs/hosts"
rsync -a "$HOST:ncolor_bench/bench_outputs/hosts/" "$SRC/bench_outputs/hosts/"
