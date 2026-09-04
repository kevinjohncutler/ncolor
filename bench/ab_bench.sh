#!/usr/bin/env bash
# Paired A/B of two or more ncolor trees on one machine.
#
#   bench/ab_bench.sh <corpus.npz> <out_dir> <reps> name=dir [name=dir ...]
#
# Every tree is built, then measured in a rotating order, one repetition
# at a time, so thermal drift and background load land on all of them
# equally and no tree always follows the same one. PY overrides the
# interpreter.
set -euo pipefail
CORPUS=$1; OUT=$2; REPS=$3; shift 3
TREES=("$@")
PY=${PY:-$HOME/.pyenv/shims/python}
# One measurement script for every tree, taken from the last one given.
RUNNER="${TREES[-1]#*=}/bench/ab_run.py"
mkdir -p "$OUT"

# Always rebuild. Skipping when a .so is already there looks like a
# cheap win and is a trap: the rsync that refreshes a tree excludes
# *.so, so --delete cannot remove the binary left by whatever was in
# that directory before. The build would then be skipped and the
# measurement would silently be of the previous tree.
for spec in "${TREES[@]}"; do
  name=${spec%%=*}; d=${spec#*=}
  ( cd "$d"
    rm -rf build src/ncolor/_backend/_impl*.so
    SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-2.0.3.dev0} \
    NCOLOR_NO_CALIBRATE=1 "$PY" setup.py build_ext --inplace > "build_$name.log" 2>&1 \
      || { tail -30 "build_$name.log"; exit 1; } )
  echo "[$(hostname -s)] built $name"
done

run() {  # run <tree> <name> <mode> <outfile> [shuffle-seed]
  NCOLOR_NO_CALIBRATE=1 PYTHONPATH="$1/src" "$PY" "$RUNNER" \
    --corpus "$CORPUS" --build "$2" --mode "$3" --out "$4" --shuffle "${5:--1}"
}

for spec in "${TREES[@]}"; do
  name=${spec%%=*}; d=${spec#*=}
  run "$d" "$name" verify "$OUT/verify_$name.json"
done

n=${#TREES[@]}
for r in $(seq 1 "$REPS"); do
  # Rotate which tree goes first, so none of them is always warmed by
  # the same neighbor.
  for k in $(seq 0 $((n - 1))); do
    spec=${TREES[$(( (r + k) % n ))]}
    name=${spec%%=*}; d=${spec#*=}
    # Same case order for every tree within a repetition, a different
    # one each repetition.
    run "$d" "$name" serial     "$OUT/serial_${name}_r${r}.json" "$r"
    run "$d" "$name" graphs     "$OUT/graphs_${name}_r${r}.json" "$r"
    run "$d" "$name" concurrent "$OUT/conc_${name}_r${r}.json"   "$r"
  done
done
echo "[$(hostname -s)] A/B done: $REPS reps over $n trees in $OUT"
