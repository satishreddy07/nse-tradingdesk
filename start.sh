#!/usr/bin/env bash
set -o errexit

export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"
PORT="${PORT:-8000}"

echo "==> Starting Quant Trading Desk on port ${PORT}..."
if [ -x "$PWD/.venv/bin/uvicorn" ]; then
  exec "$PWD/.venv/bin/uvicorn" web.api:app --host 0.0.0.0 --port "$PORT"
elif command -v uvicorn >/dev/null 2>&1; then
  exec uvicorn web.api:app --host 0.0.0.0 --port "$PORT"
else
  exec python3 -m uvicorn web.api:app --host 0.0.0.0 --port "$PORT"
fi
