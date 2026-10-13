#!/usr/bin/env bash
# Exit on error
set -o errexit

export PATH="$HOME/.local/bin:$PATH"

echo "==> Installing Quant Trading Desk Python Dependencies..."
pip install --upgrade pip || python3 -m pip install --upgrade pip || true
pip install -r requirements.txt || python3 -m pip install -r requirements.txt || pip install --user -r requirements.txt
echo "==> Build Completed Successfully!"
