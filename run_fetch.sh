#!/usr/bin/env bash
# Run a lightweight fetch cycle (no synthesis/report).
# Designed to run every 3 hours via launchd.
#
# Usage:
#   ./run_fetch.sh              # full fetch cycle
#   ./run_fetch.sh --smoke      # quick smoke test (5 sources)

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

echo "=== Fetch cycle pre-flight ==="

# Check venv
if [ ! -d "$VENV" ]; then
    echo "FAIL: Python venv not found at $VENV"
    exit 1
fi
echo "  venv: OK"

# Check Ollama is running, start if not
if ! curl -s --max-time 5 "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "  Ollama not running — starting..."
    OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_FLASH_ATTENTION=1 ollama serve &
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
if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "qwen3:32b"; then
    echo "FAIL: qwen3:32b model not found. Run: ollama pull qwen3:32b"
    exit 1
fi
echo "  Model qwen3:32b: OK"

echo "=== Pre-flight passed ==="
echo ""

# --- Run fetch cycle ---

if [ -n "$MAX_TOTAL" ]; then
    echo "Running fetch cycle with max_total=$MAX_TOTAL ..."
    PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" -c "
from research.orchestrate import run_fetch_cycle
result = run_fetch_cycle(sources_override={'max_total': $MAX_TOTAL})
print()
print(f'Fetch cycle complete: {result}')
"
else
    echo "Running full fetch cycle ..."
    PYTHONPATH="$SCRIPT_DIR" "$VENV/bin/python" -m research.orchestrate --fetch-only
fi

echo ""
echo "=== Fetch cycle done ==="
echo "  DB:  $SCRIPT_DIR/research/research.db"
echo "  Log: $SCRIPT_DIR/research/logs/$(date +%Y-%m-%d).log"
