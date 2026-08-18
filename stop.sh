#!/usr/bin/env bash
#
# Stop the Docs Q&A server started by run.sh.
#
#   bash stop.sh             # stop the server on the default port (8000)
#   PORT=9000 bash stop.sh   # match a custom PORT used with run.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PID_FILE=".docsqa.pid"
PORT="${PORT:-8000}"
stopped=0

# 1) Preferred path: the PID recorded by run.sh.
if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo ">> Stopping Docs Q&A (PID $PID)"
    kill "$PID" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo ">> Still alive; forcing (SIGKILL)"
      kill -9 "$PID" 2>/dev/null || true
    fi
    stopped=1
  fi
  rm -f "$PID_FILE"
fi

# 2) Fallback: whatever is still listening on the port (e.g. PID file lost).
if [ "$stopped" = "0" ]; then
  if command -v lsof >/dev/null 2>&1; then
    PIDS="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
    if [ -n "$PIDS" ]; then
      echo ">> Stopping process(es) on port $PORT: $PIDS"
      # shellcheck disable=SC2086
      kill $PIDS 2>/dev/null || true
      stopped=1
    fi
  elif command -v fuser >/dev/null 2>&1; then
    if fuser -k "${PORT}/tcp" 2>/dev/null; then stopped=1; fi
  fi
fi

if [ "$stopped" = "1" ]; then
  echo ">> Stopped."
else
  echo ">> Nothing to stop (no live PID file, nothing on port $PORT)."
fi
