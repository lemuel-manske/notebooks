VENV ?= .venv

PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)

init:
	python3 -m venv $(VENV)

	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

serve:
	$(PYTHON) -m jupyter lab
