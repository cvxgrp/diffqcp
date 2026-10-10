#!/usr/bin/env bash
# GPU verification and benchmark for diffqcp. Run from the repository root on a
# machine with an NVIDIA GPU (CUDA 12 driver) and `uv` installed:
#
#     bash scripts/gpu_check.sh
#
# Everything is written to gpu_results/ and bundled as gpu_results.tar.gz (send
# that back). Set DIFFQCP_GPU_BENCH_CPU_BASELINE=1 to also run the CPU benchmark
# on the same machine (roughly doubles the benchmark time).
# It only installs into the project's .venv and changes no tracked files.
set -uo pipefail

OUT=gpu_results
mkdir -p "$OUT"
REPORT=$OUT/gpu_report.txt
LOG=$OUT/gpu_pytest.log
: > "$REPORT"

say() { echo "$*" | tee -a "$REPORT"; }

say "diffqcp GPU check — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "commit: $(git rev-parse --short HEAD 2>/dev/null) ($(git rev-parse --abbrev-ref HEAD 2>/dev/null))"
if [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]; then say "note: working tree has local changes"; fi
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >"$OUT/nvidia-smi.txt" 2>&1
lscpu >"$OUT/lscpu.txt" 2>&1 || true

if ! command -v uv >/dev/null 2>&1; then
    say "ERROR: uv not found. Install it (https://docs.astral.sh/uv/) and re-run."
    exit 1
fi

say ""
say "== 0. Install (uv sync --extra gpu --group dev, then the benchmark extras)"
if ! uv sync --extra gpu --group dev >"$OUT/install.log" 2>&1; then
    say "ERROR: uv sync failed; last lines:"
    tail -n 30 "$OUT/install.log" | tee -a "$REPORT"
    exit 1
fi
# Benchmark-only packages, not in the lockfile: Moreau (CPU + CUDA 12 builds) and mpmath.
if uv pip install "moreau==0.4.1" "mpmath>=1.3" >>"$OUT/install.log" 2>&1; then
    say "  moreau + mpmath: ok"
else
    say "  WARNING: could not install moreau/mpmath (Moreau rows and the conditioning sweep will fail)"
fi
if uv pip install "moreau-cuda12==0.4.1" >>"$OUT/install.log" 2>&1; then
    say "  moreau-cuda12: ok"
else
    say "  note: moreau-cuda12 not installed; Moreau will run on the CPU"
fi
say "  ok"

say ""
say "== Test suite on GPU (pytest; ~15 min)"
uv run --no-sync pytest -q -p no:cacheprovider -rfE --tb=short >"$LOG" 2>&1
PYTEST_STATUS=$?
# Summary line plus the failure section, if any.
grep -E "^(FAILED|ERROR) |passed|failed|error" "$LOG" | tail -n 40 | sed 's/^/  /' | tee -a "$REPORT"
say "  pytest exit status: $PYTEST_STATUS (full log: $LOG)"

say ""
# Solver agreement, learning curves and the derivative benchmark (~1-2 h).
uv run --no-sync python scripts/gpu_checks.py "$REPORT" "$OUT"
CHECKS_STATUS=$?

tar -czf gpu_results.tar.gz "$OUT"
echo
echo "Done. Send back gpu_results.tar.gz (everything in $OUT/)."
exit $(( PYTEST_STATUS != 0 || CHECKS_STATUS != 0 ))
