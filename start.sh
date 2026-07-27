#!/usr/bin/env bash
# One-command startup for usain-bot.
#
#   ./start.sh                 # real Garmin data (needs .env credentials)
#   ./start.sh --demo          # bundled sample data, no Garmin account or API key needed
#   ./start.sh --port 9000     # any extra flags pass through to `usain-bot serve`
#
# Handles: venv creation/activation, dependency install, first-run plan
# generation, and launching the UI in your browser. Safe to re-run.

set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
DEMO=0
SERVE_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --demo) DEMO=1 ;;
    *) SERVE_ARGS+=("$arg") ;;
  esac
done

if [ "$DEMO" -eq 1 ]; then
  SERVE_ARGS+=(--mock-fixture tests/fixtures/mock_activities.json)
fi

# --- venv ---------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv ($VENV_DIR)"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- dependencies -------------------------------------------------------
if ! python -c "import usain_bot" 2>/dev/null; then
  echo "==> Installing dependencies (first run only, may take a minute)"
  pip install --quiet --upgrade pip
  pip install --quiet -e ".[dev,chat]"
fi

# --- env ----------------------------------------------------------------
if [ ! -f .env ] && [ "$DEMO" -eq 0 ]; then
  echo "==> No .env found; creating one from .env.example"
  cp .env.example .env
  echo "    [!] Edit .env with your Garmin credentials, then re-run ./start.sh"
  echo "        (or run './start.sh --demo' right now to explore with sample data)"
  exit 1
fi

# --- first-run plan -----------------------------------------------------
DATA_DIR="${USAIN_BOT_DATA_DIR:-./data}"
DB_PATH="$DATA_DIR/usain_bot.db"
if [ ! -f "$DB_PATH" ]; then
  echo "==> First run: pulling Garmin history and generating your plan"
  if [ "$DEMO" -eq 1 ]; then
    usain-bot init --yes --mock-fixture tests/fixtures/mock_activities.json
  else
    echo "    (tip: run 'usain-bot backfill' afterwards for your full history)"
    usain-bot init --yes
  fi
fi

# --- launch -------------------------------------------------------------
echo "==> Starting usain-bot. Press Ctrl+C to stop."
exec usain-bot serve --open "${SERVE_ARGS[@]}"
