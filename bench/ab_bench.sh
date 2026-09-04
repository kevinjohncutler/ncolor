#!/usr/bin/env bash
# Paired A/B of two ncolor trees on one machine.
#
#   bench/ab_bench.sh <base_dir> <cand_dir> <corpus.npz> <out_dir> <reps>
#
# Both trees are built, then measured in alternating order, one
# repetition at a time, so thermal drift and background load land on
# both sides equally. PY overrides the interpreter.
set -euo pipefail
BASE=$1; CAND=$2; CORPUS=$3; OUT=$4; REPS=${5:-6}
PY=${PY:-$HOME/.pyenv/shims/python}
RUNNER="$CAND/bench/ab_run.py"          # one measurement script for both
mkdir -p "$OUT"

build() {
  local d=$1 tag=$2
  [ -n "$(ls "$d"/src/ncolor/_backend/_impl*.so 2>/dev/null)" ] && return 0
  ( cd "$d"
    SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-2.0.3.dev0} \
    NCOLOR_NO_CALIBRATE=1 "$PY" setup.py build_ext --inplace > "build_$tag.log" 2>&1 \
      || { tail -30 "build_$tag.log"; exit 1; } )
  echo "[$(hostname -s)] built $tag"
}
build "$BASE" base
build "$CAND" cand

run() {  # run <tree> <label> <mode> <outfile> [shuffle-seed]
  NCOLOR_NO_CALIBRATE=1 PYTHONPATH="$1/src" "$PY" "$RUNNER" \
    --corpus "$CORPUS" --build "$2" --mode "$3" --out "$4" --shuffle "${5:--1}"
}

run "$BASE" base verify "$OUT/verify_base.json"
run "$CAND" cand verify "$OUT/verify_cand.json"

for r in $(seq 1 "$REPS"); do
  if [ $((r % 2)) -eq 1 ]; then order="base cand"; else order="cand base"; fi
  for who in $order; do
    if [ "$who" = base ]; then d=$BASE; else d=$CAND; fi
    # Same order for both builds within a repetition, a different one
    # each repetition, so no case keeps sitting behind the same
    # neighbor.
    run "$d" "$who" serial     "$OUT/serial_${who}_r${r}.json" "$r"
    run "$d" "$who" concurrent "$OUT/conc_${who}_r${r}.json"   "$r"
  done
done
echo "[$(hostname -s)] A/B done: $REPS reps in $OUT"
