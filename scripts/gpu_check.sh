#!/usr/bin/env bash
# GPU verification for diffqcp. Run from the repository root on a machine with an
# NVIDIA GPU (CUDA 12 driver) and `uv` installed:
#
#     bash scripts/gpu_check.sh
#
# Writes gpu_report.txt (send this back) and gpu_pytest.log (full test log).
# It only installs into the project's .venv and changes no tracked files.
set -uo pipefail

REPORT=gpu_report.txt
LOG=gpu_pytest.log
: > "$REPORT"

say() { echo "$*" | tee -a "$REPORT"; }

say "diffqcp GPU check — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "commit: $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then say "note: working tree has local changes"; fi

if ! command -v uv >/dev/null 2>&1; then
    say "ERROR: uv not found. Install it (https://docs.astral.sh/uv/) and re-run."
    exit 1
fi

say ""
say "== 0. Install (uv sync --extra gpu --group dev)"
if ! uv sync --extra gpu --group dev >"$LOG.sync" 2>&1; then
    say "ERROR: uv sync failed; last lines:"
    tail -n 30 "$LOG.sync" | tee -a "$REPORT"
    exit 1
fi
say "  ok"

say ""
say "== Test suite on GPU (pytest; ~15 min)"
uv run pytest -q -p no:cacheprovider -rfE --tb=short >"$LOG" 2>&1
PYTEST_STATUS=$?
# Summary line plus the failure section, if any.
grep -E "^(FAILED|ERROR) |passed|failed|error" "$LOG" | tail -n 40 | sed 's/^/  /' | tee -a "$REPORT"
say "  pytest exit status: $PYTEST_STATUS (full log: $LOG)"

say ""
uv run python scripts/gpu_checks.py "$REPORT"
CHECKS_STATUS=$?

echo
echo "Done. Send back $REPORT (and $LOG if any test failed)."
exit $(( PYTEST_STATUS != 0 || CHECKS_STATUS != 0 ))
