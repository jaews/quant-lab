.PHONY: install run-dashboard run-sweep run-pipeline run-scheduler generate-synthetic test validate validate-strict

install:
	cd research && pip install -r requirements.txt

run-dashboard:
	cd research && streamlit run dashboard/app.py

run-sweep:
	cd research && python -m experiments.sweep

run-pipeline:
	cd research && python main.py run-pipeline

run-scheduler:
	cd research && python pipeline/scheduler.py

generate-synthetic:
	cd research && python -m experiments.synthetic_generator

test:
	cd research && pytest -q

validate:
	cd research && python -m analysis.validate_run --root results/runs --all

validate-strict:
	cd research && python -m analysis.validate_run --run results/runs/sample_run --strict
