.PHONY: install data train index eval test lint serve ui docker smoke all

install:      ; pip install -r requirements-dev.txt
data:         ; python -m training.prepare_data
train:        ; python -m training.train_intent
index:        ; python -m training.build_index
eval:         ; python -m training.evaluate
test:         ; pytest
lint:         ; ruff check .
serve:        ; uvicorn app.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
ui:           ; streamlit run ui/streamlit_app.py
docker:       ; docker build -t novabank-assistant:local .
smoke:        ; bash scripts/smoke_test.sh
all: data train index eval test
