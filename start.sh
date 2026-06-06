#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  JJK Domain Expansion — Unified Launcher
#  Starts both Python backend + Vite frontend
#  Ctrl+C or closing the terminal kills BOTH
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8765
FRONTEND_PORT=5173
BACKEND_PID=""
FRONTEND_PID=""
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

kill_process_tree() {
    local pid="$1"
    local children
    children="$(pgrep -P "${pid}" 2>/dev/null || true)"
    for child in ${children}; do
        kill_process_tree "${child}"
    done
    kill "${pid}" 2>/dev/null || true
}

# Cleanup function — kills all child processes
cleanup() {
    echo ""
    echo "🔴 Shutting down all processes..."
    if [ -n "${BACKEND_PID}" ]; then
        kill_process_tree "${BACKEND_PID}"
        wait "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID}" ]; then
        kill_process_tree "${FRONTEND_PID}"
        wait "${FRONTEND_PID}" 2>/dev/null || true
    fi
    echo "✅ All processes stopped."
    exit 0
}

# If stale listeners exist on expected ports, clear them before launch.
clear_port_if_busy() {
    local port="$1"
    local pids
    pids="$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "${pids}" ]; then
        echo "⚠️  Port ${port} already in use. Stopping stale process(es): ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 1
    fi
}

# Trap INT (Ctrl+C) and TERM signals
trap cleanup INT TERM

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔮 JJK Domain Expansion — Starting Up"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

clear_port_if_busy "${BACKEND_PORT}"
clear_port_if_busy "${FRONTEND_PORT}"

# 1. Start Python backend (WebSocket + CV)
echo ""
echo "⚡ Starting Python backend (ws://localhost:${BACKEND_PORT})..."
cd "$PROJECT_DIR"
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "❌ Python virtualenv not found at ${PYTHON_BIN}"
    echo "   Create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi
export MPLCONFIGDIR="${PROJECT_DIR}/.mplconfig"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export PYTHONPYCACHEPREFIX="${PROJECT_DIR}/.pycache"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${PYTHONPYCACHEPREFIX}"
"${PYTHON_BIN}" app.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Give the backend a moment to initialize
sleep 2

# 2. Start Vite frontend dev server
echo ""
echo "🌐 Starting Vite frontend (http://localhost:${FRONTEND_PORT})..."
cd "$PROJECT_DIR/frontend"
./node_modules/.bin/vite &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Both servers running!"
echo "  🌐 Open: http://localhost:${FRONTEND_PORT}"
echo "  📡 API:  ws://localhost:${BACKEND_PORT}"
echo "  🛑 Press Ctrl+C to stop everything"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for either process to exit — then kill both
while kill -0 "${BACKEND_PID}" 2>/dev/null && kill -0 "${FRONTEND_PID}" 2>/dev/null; do
    sleep 1
done
cleanup
