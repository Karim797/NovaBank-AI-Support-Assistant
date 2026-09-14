#!/usr/bin/env bash
# Cold start from a clean checkout: data -> model -> index -> evaluate -> serve.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m training.prepare_data
python -m training.train_intent
python -m training.build_index
python -m training.evaluate
python -m training.eval_retrieval
exec uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8000
