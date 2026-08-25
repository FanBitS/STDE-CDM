.PHONY: install test compile figures fica-figure

install:
	python -m pip install -e ".[dispatch,plot,test,notebook]"

test:
	python -m pytest -q

compile:
	python -m compileall -q src scripts tests

figures:
	python scripts/build_paper_locked_test_result.py
	python scripts/make_paper_visualizations.py
	python scripts/make_locked_test_dm_tests.py

fica-figure:
	python scripts/make_fica_backtest_figure.py
