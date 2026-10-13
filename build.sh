#!/usr/bin/env bash
# Exit on error
set -o errexit

export PATH="$PWD/.venv/bin:$HOME/.local/bin:$PATH"

echo "==> Installing Quant Trading Desk Python Dependencies..."
if pip install -r requirements.txt; then
  echo "==> Installed via default pip"
elif pip install --break-system-packages -r requirements.txt; then
  echo "==> Installed via pip --break-system-packages"
elif python3 -m pip install --break-system-packages -r requirements.txt; then
  echo "==> Installed via python3 -m pip --break-system-packages"
else
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
  echo "==> Installed inside .venv"
fi
echo "==> Build Completed Successfully!"
