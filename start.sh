#!/usr/bin/env bash
set -o errexit

export PATH="$HOME/.local/bin:$PATH"
PORT="${PORT:-8000}"

echo "==> Starting Quant Trading Desk on port ${PORT}..."
if command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn web.api:app --host 0.0.0.0 --port "$PORT"
else
  exec python3 -m uvicorn web.api:app --host 0.0.0.0 --port "$PORT"
fi
