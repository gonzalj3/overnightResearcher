#!/usr/bin/env bash
# Run the nightly research pipeline end-to-end.
#
# Usage:
#   ./run.sh              # full run (~400 sources, ~2-3 hours)
#   ./run.sh --smoke      # quick smoke test (5 sources, ~2-3 min)
#   ./run.sh --max 20     # custom source limit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
OLLAMA_URL="http://localhost:11434"

# Parse args
MAX_TOTAL=""
SMOKE=false
for arg in "$@"; do
    case "$arg" in
        --smoke) SMOKE=true ;;
        --max)   shift_next=true ;;
        *)
            if [ "${shift_next:-}" = true ]; then
                MAX_TOTAL="$arg"
                shift_next=false
            fi
            ;;
    esac
done
if [ "$SMOKE" = true ] && [ -z "$MAX_TOTAL" ]; then
    MAX_TOTAL=5
fi

# --- Pre-flight checks ---

echo "=== Pre-flight checks ==="

# Check venv
if [ ! -d "$VENV" ]; then
    echo "FAIL: Python venv not found at $VENV"
    echo "  Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
echo "  venv: OK"

# Check Ollama is running, start if not
if ! curl -s --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "  Ollama not running — starting..."
    OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=5m ollama serve &
    OLLAMA_PID=$!
    echo "  Waiting for Ollama (pid $OLLAMA_PID)..."
    for i in $(seq 1 30); do
        if curl -s --max-time 2 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if ! curl -s --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        echo "FAIL: Ollama did not start within 30s"
        exit 1
    fi
    echo "  Ollama: started"
else
    echo "  Ollama: already running"
fi

# Check model is available
if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "qwen3.5:27b\|qwen3.5-27b"; then
    echo "FAIL: qwen3.5:27b model not found. Run: ollama pull qwen3.5:27b"
    exit 1
fi
echo "  Model qwen3.5:27b: OK"

# Disk space check
DISK_FREE_GB=$(df -g "$HOME" | tail -1 | awk '{print $4}')
if [ "$DISK_FREE_GB" -lt 1 ]; then
    echo "FAIL: Less than 1GB disk space free"
    exit 1
fi
echo "  Disk free: ${DISK_FREE_GB}GB"

echo "=== Pre-flight passed ==="
echo ""

# --- Run pipeline ---

if [ -n "$MAX_TOTAL" ]; then
    echo "Running pipeline with max_total=$MAX_TOTAL ..."
    PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" -c "
from research.orchestrate import run_nightly_research
path = run_nightly_research(sources_override={'max_total': $MAX_TOTAL})
print()
print(f'Report saved to: {path}')
"
else
    echo "Running full pipeline ..."
    PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" -m research.orchestrate
fi

echo ""
echo "=== Done ==="
echo "  Report:  ~/reports/$(date +%Y-%m-%d).md"
echo "  Log:     $SCRIPT_DIR/research/logs/$(date +%Y-%m-%d).log"
echo "  DB:      $SCRIPT_DIR/research/research.db"
