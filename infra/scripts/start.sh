#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

#  Activate venv if present 
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

#  Load .env 
if [ -f ".env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
fi

APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_DEBUG="${APP_DEBUG:-false}"
UI_PORT="${UI_PORT:-7860}"
START_UI="${START_UI:-true}"

echo "==> [start] Starting FastAPI on ${APP_HOST}:${APP_PORT}  (debug=${APP_DEBUG})"

#  Start FastAPI (background if UI is also starting) 
if [ "${START_UI}" = "true" ]; then
    if [ "${APP_DEBUG}" = "true" ]; then
        uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload &
    else
        uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" &
    fi
    API_PID=$!

    echo "==> [start] Starting Gradio UI on 0.0.0.0:${UI_PORT}"
    python -m app.ui.app_ui &
    UI_PID=$!

    # Trap Ctrl-C and kill both processes
    trap 'echo ""; echo "Stopping..."; kill "$API_PID" "$UI_PID" 2>/dev/null; exit 0' INT TERM

    wait "$API_PID" "$UI_PID"
else
    if [ "${APP_DEBUG}" = "true" ]; then
        uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT" --reload
    else
        uvicorn app.main:app --host "$APP_HOST" --port "$APP_PORT"
    fi
fi