# Everything runs offline against synthetic data. No credentials, no cloud.
#
# `make demo` is the entry point: generate trips, build as-of features, train
# one ranker per availability tier, and print the comparison against every
# baseline and against the simulator's own accuracy ceiling.

PY := .venv/bin/python
PROFILE ?= demo

.PHONY: setup data pipeline demo tune serve dagster mlflow test lint types clean

setup:  ## Create the virtualenv and install exactly what uv.lock pins
	uv venv --python 3.11
	uv sync --frozen --extra dev
	.venv/bin/pre-commit install || true

data:  ## Generate offices, employees, trips and the truth set
	OVR_PROFILE=$(PROFILE) $(PY) -m visit_ranker.gen.run

pipeline:  ## Features, models, evaluation, calibration, charts
	OVR_PROFILE=$(PROFILE) $(PY) -m visit_ranker.reporting.results

tune:  ## Optuna search on the validation window, then the full pipeline
	OVR_PROFILE=$(PROFILE) $(PY) -m visit_ranker.reporting.results --tune

demo: data pipeline  ## The whole thing

serve:  ## FastAPI ranking service
	.venv/bin/uvicorn visit_ranker.serving.app:app --reload --port 8000

dagster:  ## Dagster UI for the asset graph
	.venv/bin/dagster dev -m visit_ranker.orchestration.definitions

mlflow:  ## MLflow UI for the model comparison
	.venv/bin/mlflow ui --backend-store-uri file:mlruns --port 5000

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

types:
	.venv/bin/mypy src

clean:  ## Remove generated data, outputs and tracking
	rm -rf data out mlruns models
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
