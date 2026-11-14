# Rebuilds every number, chart and table in this repo from the raw data.
#
#   make all        download the data, build the warehouse, run all seven phases  (~25 min)
#   make data       download + dbt build (models and data tests)                  (~1 min)
#   make test       lint + unit tests (+ warehouse tests if the warehouse exists)
#   make app        launch the coupon budget simulator
#   make replicate DUNNHUMBY=path/to/csvs   rerun the triple difference on the 2-year release

PYTHON ?= python
RUN = $(PYTHON) -W ignore -m coupon_incrementality
export DBT_PROFILES_DIR = .

.PHONY: all install data analysis figures report test lint app replicate clean

all: data analysis figures report

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

data:
	$(RUN).download
	cd dbt && dbt build

# Order matters: later phases read the result files of earlier ones.
analysis:
	$(RUN).naive
	$(RUN).dml
	$(RUN).triple_diff
	$(RUN).event_study
	$(RUN).dml_bc
	$(RUN).decomposition
	$(RUN).heterogeneity
	$(RUN).policy
	$(RUN).robustness

figures:
	$(RUN).figures

report:
	$(RUN).report

lint:
	ruff check src tests app
	ruff format --check src tests app

test: lint
	$(PYTHON) -W ignore -m pytest -q

app:
	streamlit run app/streamlit_app.py

replicate:
	$(RUN).replicate $(DUNNHUMBY)

clean:
	rm -rf data dbt/target dbt/logs reports/results reports/figures reports/RESULTS.md
