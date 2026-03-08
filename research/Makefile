.PHONY: install run-dashboard run-sweep run-pipeline run-scheduler generate-synthetic test validate validate-strict

install:
	pip install -r requirements.txt

run-dashboard:
	streamlit run dashboard/app.py

run-sweep:
	python -m experiments.sweep

run-pipeline:
	python main.py run-pipeline

run-scheduler:
	python pipeline/scheduler.py

generate-synthetic:
	python -m experiments.synthetic_generator

test:
	pytest -q

validate:
	python -m analysis.validate_run --root results/runs --all

validate-strict:
	python -m analysis.validate_run --run results/runs/sample_run --strict
