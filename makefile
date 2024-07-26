get_coverage: |
	export PYTHONPATH=`pwd` && pytest -vv --cov=adaptivee --cov-report=term-missing
	rm .coverage