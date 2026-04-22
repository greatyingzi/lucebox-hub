#!/usr/bin/env bash
# DFlash llama-server launcher with process management
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../deps/llama.cpp/build" && pwd)"
BIN="$BUILD_DIR/bin/llama-server"

# Defaults
MODEL="${MODEL:-$HOME/llm-server/models/unsloth/Qwen3.5-27B-UD-Q5_K_XL.gguf}"
DRAFT="${DRAFT:-$SCRIPT_DIR/../models/draft/qwen3.5-27b-dflash-draft-Q8_0.gguf}"
PORT="${PORT:-8092}"
CTX="${CTX:-4096}"
GPULAYERS="${GPULAYERS:-99}"
HOST="${HOST:-0.0.0.0}"
DFLASH_CTX="${DFLASH_CTX:-}"

usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start     Start DFlash server (default)"
    echo "  baseline  Start server without DFlash"
    echo "  stop      Stop server"
    echo "  restart   Stop and start"
    echo "  status    Check if server is running"
    echo "  test      Quick test (requires running server)"
    echo "  bench     Run benchmark suite"
    echo ""
    echo "Options:"
    echo "  -p PORT     Port (default: 8092)"
    echo "  -m MODEL    Target model path"
    echo "  -d DRAFT    Draft model path"
    echo "  -c CTX      Context size (default: 4096)"
    echo "  -g LAYERS   GPU layers (default: 99)"
    echo "  -x CTX_MAX  DFlash draft context (default: model default)"
    echo "  -h          This help"
}

kill_server() {
    local pids
    pids=$(pgrep -f "llama-server.*--port $PORT" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing llama-server on port $PORT (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
        # Verify killed
        pids=$(pgrep -f "llama-server.*--port $PORT" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "WARNING: Process still alive, trying harder..."
            echo "$pids" | xargs kill -9 2>/dev/null || true
            sleep 2
        fi
        echo "Server stopped."
    else
        echo "No llama-server found on port $PORT."
    fi
}

wait_for_server() {
    local timeout="${1:-60}"
    local elapsed=0
    echo -n "Waiting for server on port $PORT"
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo " TIMEOUT"
    echo "Server did not start within ${timeout}s. Check logs:"
    echo "  $LOG_FILE"
    return 1
}

run_test() {
    local url="http://localhost:$PORT"
    if ! curl -sf "$url/health" > /dev/null 2>&1; then
        echo "Server not running on port $PORT"
        return 1
    fi

    echo "=== Quick Test ==="
    local tmpf=$(mktemp)
    curl -s "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Tell me a joke."}],"max_tokens":50,"temperature":0}' \
        -o "$tmpf" 2>/dev/null

    if [ ! -s "$tmpf" ]; then
        echo "ERROR: Empty response"
        rm -f "$tmpf"
        return 1
    fi

    python3 -c "
import sys, json
with open('$tmpf') as f:
    d = json.loads(f.read(), strict=False)
t = d['timings']
acc = t.get('draft_n_accepted', 0)
tot = t.get('draft_n', 0)
pct = f'{acc/tot*100:.0f}%' if tot > 0 else 'N/A'
print(f'Speed: {t[\"predicted_per_second\"]:.1f} tok/s')
print(f'Accepted: {acc}/{tot} = {pct}')
print(f'Tokens: {t[\"predicted_n\"]}')
" 2>&1
    rm -f "$tmpf"
}

run_bench() {
    local url="http://localhost:$PORT"
    if ! curl -sf "$url/health" > /dev/null 2>&1; then
        echo "Server not running on port $PORT"
        return 1
    fi

    echo "=== Benchmark Suite ==="
    echo "Prompt | Speed | Acceptance"
    echo "-------|-------|----------"
    for prompt in "Tell me a joke." "Write a haiku about rain." "Explain quantum computing" "What is the meaning of life?" "Count from 1 to 20"; do
        local tmpf=$(mktemp)
        curl -s "$url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":200,\"temperature\":0}" \
            -o "$tmpf" 2>/dev/null
        python3 -c "
import sys, json
with open('$tmpf') as f:
    d = json.loads(f.read(), strict=False)
t = d['timings']
acc = t.get('draft_n_accepted', 0)
tot = t.get('draft_n', 0)
pct = f'{acc/tot*100:.0f}%' if tot > 0 else 'N/A'
print(f'$prompt | {t[\"predicted_per_second\"]:.1f} tok/s | {acc}/{tot} = {pct}')
" 2>&1 || echo "$prompt | ERROR | ERROR"
        rm -f "$tmpf"
    done
}

# Parse arguments
COMMAND="start"
while [ $# -gt 0 ]; do
    case "$1" in
        start|baseline|stop|restart|status|test|bench) COMMAND="$1"; shift ;;
        -p) PORT="$2"; shift 2 ;;
        -m) MODEL="$2"; shift 2 ;;
        -d) DRAFT="$2"; shift 2 ;;
        -c) CTX="$2"; shift 2 ;;
        -g) GPULAYERS="$2"; shift 2 ;;
        -x) DFLASH_CTX="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

LOG_FILE="/tmp/llama-server-${PORT}.log"

case "$COMMAND" in
    stop)
        kill_server
        ;;
    status)
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server running on port $PORT"
            pgrep -fa "llama-server.*--port $PORT" 2>/dev/null || true
        else
            echo "Server NOT running on port $PORT"
        fi
        ;;
    test)
        run_test
        ;;
    bench)
        run_bench
        ;;
    start|baseline|restart)
        if [ "$COMMAND" = "restart" ]; then
            kill_server
        fi

        # Check if already running
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server already running on port $PORT. Use '$0 stop' first or '$0 restart'."
            exit 0
        fi

        # Kill any leftover on this port
        kill_server

        # Build command
        COMMON_ARGS=(
            -m "$MODEL"
            -ngl "$GPULAYERS"
            -c "$CTX"
            -np 1
            --port "$PORT"
            --host "$HOST"
            -ctk q8_0
            -ctv q8_0
        )

        if [ "$COMMAND" = "start" ] || [ "$COMMAND" = "restart" ]; then
            if [ ! -f "$DRAFT" ]; then
                echo "ERROR: Draft model not found: $DRAFT"
                exit 1
            fi
            echo "Starting DFlash server on port $PORT"
            echo "  Target: $MODEL"
            echo "  Draft:  $DRAFT"
            DFLASH_ARGS=()
            if [ -n "$DFLASH_CTX" ]; then
                DFLASH_ARGS+=(--dflash-ctx-max "$DFLASH_CTX")
                echo "  CtxMax: $DFLASH_CTX"
            fi
            nohup "$BIN" "${COMMON_ARGS[@]}" --dflash-draft "$DRAFT" "${DFLASH_ARGS[@]}" \
                </dev/null >"$LOG_FILE" 2>&1 &
        else
            echo "Starting baseline server (no DFlash) on port $PORT"
            echo "  Target: $MODEL"
            nohup "$BIN" "${COMMON_ARGS[@]}" \
                </dev/null >"$LOG_FILE" 2>&1 &
        fi

        PID=$!
        echo "  PID: $PID"
        echo "  Log: $LOG_FILE"
        disown

        if wait_for_server 90; then
            echo "Server ready!"
        else
            exit 1
        fi
        ;;
esac
