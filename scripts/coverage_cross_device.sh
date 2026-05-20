#!/usr/bin/env bash
#
# Run the ncolor test suite on the local host plus any configured
# remote, then combine .coverage data into a single merged report.
# Useful for picking up OS-specific branches (linux _proc_/cpuinfo,
# darwin sysctl, windows wmic, ...) on machines that actually run
# those branches, instead of mocking them.
#
# Configuration. The script needs at least the project's local root
# and (optionally) zero or more remotes to also run on:
#
#   NCOLOR_COV_LOCAL_ROOT
#       Absolute path to the project on this machine. Defaults to the
#       parent of the directory containing this script.
#
#   NCOLOR_COV_REMOTE_ROOT
#       Absolute path to the project on each remote (must be the same
#       across remotes; they're typically a shared NAS mount). May
#       contain shell variables like ``$HOME`` — these are expanded by
#       the remote shell, not locally. Defaults to
#       ``$HOME/DataDrive/ncolor``, since that's a common layout for a
#       NAS-shared source tree.
#
#   NCOLOR_COV_REMOTES
#       Space-separated list of "label=user@host" pairs to also run on.
#       Example: NCOLOR_COV_REMOTES="big=me@desktop.lan tiny=me@nuc.lan"
#       Unset / empty → local-only.
#
# Alternatively, a config file at scripts/coverage_cross_device.conf
# (gitignored) is read line-by-line, one "label  user@host" per line.
# Lines starting with '#' are ignored. The config file takes priority
# over NCOLOR_COV_REMOTES.
#
# Usage:
#   bash scripts/coverage_cross_device.sh         # merged report
#   bash scripts/coverage_cross_device.sh --html  # also open htmlcov
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${NCOLOR_COV_LOCAL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# Literal '$HOME' — expanded by the remote shell, not here.
REMOTE_ROOT="${NCOLOR_COV_REMOTE_ROOT:-\$HOME/DataDrive/ncolor}"
COV_DIR="$LOCAL_ROOT/.coverage_combined"
PYENV='export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"'
PYTEST_ARGS="tests/ -q"

WANT_HTML=0
if [[ "${1:-}" == "--html" ]]; then WANT_HTML=1; fi

# Build the remote list. Config file (if present) wins over env var.
REMOTES=()
CONF="$SCRIPT_DIR/coverage_cross_device.conf"
if [[ -f "$CONF" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line## }"; line="${line%% }"
        [[ -z "$line" ]] && continue
        REMOTES+=("$line")
    done < "$CONF"
elif [[ -n "${NCOLOR_COV_REMOTES:-}" ]]; then
    # Env var format: "label=user@host label=user@host"
    for entry in $NCOLOR_COV_REMOTES; do
        label="${entry%%=*}"; target="${entry#*=}"
        REMOTES+=("$label $target")
    done
fi

rm -rf "$COV_DIR"
mkdir -p "$COV_DIR"

# Local host run.
echo "=== local ==="
cd "$LOCAL_ROOT"
python -m coverage run --data-file="$COV_DIR/.coverage.local" -m pytest $PYTEST_ARGS

# Remote runs (best-effort: unreachable hosts skipped, not fatal).
run_remote() {
    local label="$1" host="$2"
    echo ""
    echo "=== $label ($host) ==="
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" true 2>/dev/null; then
        echo "  ($host unreachable — skipped)"
        return
    fi
    # parallel=true in pyproject.toml suffixes the data file per-process,
    # so we combine on the remote first to collapse everything into one
    # canonical filename before scp'ing it back.
    local remote_dir=/tmp/ncolor_cov_$label
    ssh "$host" "$PYENV && cd $REMOTE_ROOT && \
        rm -rf $remote_dir && mkdir -p $remote_dir && \
        python -m coverage run --data-file=$remote_dir/.coverage \
            -m pytest $PYTEST_ARGS && \
        python -m coverage combine --data-file=$remote_dir/.coverage \
            $remote_dir/.coverage.*" || {
            echo "  ($label run failed — skipped)"; return
        }
    scp "$host:$remote_dir/.coverage" "$COV_DIR/.coverage.$label" >/dev/null
    ssh "$host" "rm -rf $remote_dir" >/dev/null 2>&1 || true
}

for entry in "${REMOTES[@]}"; do
    read -r label host <<< "$entry"
    run_remote "$label" "$host"
done

echo ""
echo "=== combining coverage ==="
cd "$LOCAL_ROOT"
python -m coverage combine --data-file="$COV_DIR/.coverage" "$COV_DIR"/.coverage.*
python -m coverage report --data-file="$COV_DIR/.coverage" --show-missing

if [[ $WANT_HTML -eq 1 ]]; then
    python -m coverage html --data-file="$COV_DIR/.coverage" -d "$COV_DIR/htmlcov"
    echo ""
    echo "HTML report: file://$COV_DIR/htmlcov/index.html"
fi
