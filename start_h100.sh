#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,13), "Use Python 3.13 for the verified environment"'
"$PYTHON_BIN" runner.py check --cnn-pool first --models LSTM,CNN-LSTM,MS-AgentNet,Transformer,CNN-Transformer --device cuda
exec "$PYTHON_BIN" -u start_h100.py --gpu-index "${GPU_INDEX:-0}"
