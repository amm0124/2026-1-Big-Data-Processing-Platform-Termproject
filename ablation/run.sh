#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== [1/3] Parsing XML data ==="
python3 parse_data.py

echo ""
echo "=== [2/3] Training BiLSTM ==="
python3 train.py

echo ""
echo "=== [3/3] Evaluating (Linear vs LSTM) ==="
python3 evaluate.py
