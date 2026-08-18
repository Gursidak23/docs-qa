#!/usr/bin/env bash
#
# One-command launcher for the Moonshot Docs Q&A assistant.
#
# Creates an isolated virtualenv (so its dependencies never clash with your
# system Python), installs the project, applies database migrations, and starts
# the FastAPI server + chat UI in the background so you can stop it later with
# stop.sh. Safe to re-run: the venv and install happen only once.
#
#   bash run.sh             # set up (first run) and serve on :8000
#   PORT=9000 bash run.sh   # serve on a different port
#   SKIP_MIGRATE=1 bash run.sh   # don't run alembic (DB already migrated)
#   bash stop.sh            # stop the background server
#
# Prerequisites the script can NOT install for you:
#   * Postgres 16 with the pgvector extension, reachable at DOCSQA_POSTGRES__DSN
#     (default postgresql+asyncpg://docsqa:docsqa@localhost:5432/docsqa).
#   * An LLM key in .env for answering - DOCSQA_LLM__GEMINI_API_KEY or
#     DOCSQA_LLM__GROQ_API_KEY (ingest + retrieval still work without one).
#
# Requires Python 3.11+ . Override the interpreter with:
#   PYTHON=python3.12 bash run.sh
#
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3.11}"
VENV_DIR=".venv"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PID_FILE=".docsqa.pid"
LOG_FILE="docsqa.log"

ensure_postgres() {
  local host="${PGHOST:-${DOCSQA_POSTGRES__HOST:-localhost}}"
  local port="${PGPORT:-${DOCSQA_POSTGRES__PORT:-5432}}"
  local attempts=0
  local pg_isready_bin=""
  local pg_ctl_bin=""
  local data_dir=""

  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready_bin="$(command -v pg_isready)"
  else
    for dir in \
      /opt/homebrew/opt/postgresql@16/bin \
      /opt/homebrew/opt/postgresql/bin \
      /usr/local/opt/postgresql@16/bin \
      /usr/local/opt/postgresql/bin \
      /Library/PostgreSQL/16/bin \
      /Applications/Postgres.app/Contents/Versions/latest/bin; do
      if [ -x "$dir/pg_isready" ]; then
        pg_isready_bin="$dir/pg_isready"
        break
      fi
    done
  fi

  if command -v pg_ctl >/dev/null 2>&1; then
    pg_ctl_bin="$(command -v pg_ctl)"
  else
    for dir in \
      /opt/homebrew/opt/postgresql@16/bin \
      /opt/homebrew/opt/postgresql/bin \
      /usr/local/opt/postgresql@16/bin \
      /usr/local/opt/postgresql/bin \
      /Library/PostgreSQL/16/bin \
      /Applications/Postgres.app/Contents/Versions/latest/bin; do
      if [ -x "$dir/pg_ctl" ]; then
        pg_ctl_bin="$dir/pg_ctl"
        break
      fi
    done
  fi

  if [ -n "$pg_isready_bin" ] && "$pg_isready_bin" -h "$host" -p "$port" >/dev/null 2>&1; then
    echo ">> PostgreSQL is ready at $host:$port"
    return 0
  fi

  if pgrep -x postgres >/dev/null 2>&1; then
    echo ">> PostgreSQL process detected"
    return 0
  fi

  while [ "$attempts" -lt 10 ]; do
    if [ -n "$pg_isready_bin" ] && "$pg_isready_bin" -h "$host" -p "$port" >/dev/null 2>&1; then
      echo ">> PostgreSQL is ready at $host:$port"
      return 0
    fi

    if [ "$attempts" -eq 0 ]; then
      echo ">> PostgreSQL not reachable at $host:$port; attempting to start it"
      if command -v brew >/dev/null 2>&1; then
        if brew services list 2>/dev/null | grep -q 'postgresql@16'; then
          brew services start postgresql@16 >/dev/null 2>&1 || true
        elif brew services list 2>/dev/null | grep -q 'postgresql'; then
          brew services start postgresql >/dev/null 2>&1 || true
        fi
      elif command -v service >/dev/null 2>&1; then
        service postgresql start >/dev/null 2>&1 || service postgresql@16 start >/dev/null 2>&1 || true
      fi

      if [ -n "$pg_ctl_bin" ]; then
        for candidate in \
          /opt/homebrew/var/postgresql@16 \
          /opt/homebrew/var/postgresql \
          /usr/local/var/postgresql@16 \
          /usr/local/var/postgres \
          "$HOME/Library/Application Support/Postgres/var-16"; do
          if [ -d "$candidate" ] && [ -f "$candidate/PG_VERSION" ]; then
            data_dir="$candidate"
            break
          fi
        done

        if [ -n "$data_dir" ]; then
          if [ -f "$data_dir/postmaster.pid" ]; then
            local stale_pid
            stale_pid="$(head -n 1 "$data_dir/postmaster.pid" 2>/dev/null || true)"
            if [ -n "$stale_pid" ] && ! kill -0 "$stale_pid" 2>/dev/null; then
              rm -f "$data_dir/postmaster.pid"
            fi
          fi
          "$pg_ctl_bin" -D "$data_dir" -l "$PWD/.postgres.log" start >/dev/null 2>&1 || true
        fi
      fi
    fi

    attempts=$((attempts + 1))
    sleep 1
  done

  if [ -n "$pg_isready_bin" ] && ! "$pg_isready_bin" -h "$host" -p "$port" >/dev/null 2>&1; then
    echo "!! PostgreSQL could not be started automatically; check your local Postgres installation."
    exit 1
  fi
}

# Ensure the database is up before deciding whether the app needs to be started.
ensure_postgres

# Don't start a second copy if one is already running.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo ">> Already running (PID $(cat "$PID_FILE")). Stop it first with: bash stop.sh"
  exit 0
fi

# Fall back to python3 if the requested interpreter isn't on PATH.
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

FRESH=0
if [ ! -d "$VENV_DIR" ]; then
  echo ">> Creating virtualenv ($("$PYTHON_BIN" --version 2>&1)) in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  FRESH=1
fi

# Resolve the venv interpreter (Linux/macOS vs Git-Bash on Windows).
if [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PY="$VENV_DIR/bin/python"
else
  VENV_PY="$VENV_DIR/Scripts/python.exe"
fi

# Install the project + dependencies only when needed (can take a few minutes
# the first time - it pulls the local embedding/reranker runtimes).
if [ "$FRESH" = "1" ] || ! "$VENV_PY" -c "import docsqa" >/dev/null 2>&1; then
  echo ">> Installing dependencies (first run only)"
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -e .
fi

# Give settings/keys a home on first run (never overwrites an existing .env).
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ">> Wrote .env from .env.example - add your Postgres DSN / LLM key there."
fi

# Apply migrations (enables pgvector + creates the schema). Needs a reachable
# Postgres; bypass with SKIP_MIGRATE=1 once the database is already migrated.
if [ "${SKIP_MIGRATE:-0}" != "1" ]; then
  echo ">> Applying database migrations (alembic upgrade head)"
  if ! "$VENV_PY" -m alembic upgrade head; then
    echo ""
    echo "!! Migrations failed - is Postgres running with pgvector, and do the"
    echo "   role/database in DOCSQA_POSTGRES__DSN exist? One-time setup:"
    echo "     psql -c \"CREATE USER docsqa WITH PASSWORD 'docsqa' SUPERUSER;\""
    echo "     psql -c \"CREATE DATABASE docsqa OWNER docsqa;\""
    echo "   Then re-run:  bash run.sh   (or SKIP_MIGRATE=1 bash run.sh to bypass)"
    exit 1
  fi
fi

# Launch the API + chat UI in the background; record the PID so stop.sh finds it.
echo ">> Starting Docs Q&A on http://localhost:$PORT/  (logs -> $LOG_FILE)"
nohup "$VENV_PY" -m uvicorn docsqa.api.app:app --host "$HOST" --port "$PORT" \
  >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# Confirm it actually came up (surface import/DSN errors instead of failing silently).
sleep 2
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo ">> Running (PID $(cat "$PID_FILE")). Stop it with: bash stop.sh"
else
  echo "!! Server exited immediately - check $LOG_FILE for the error."
  rm -f "$PID_FILE"
  exit 1
fi
