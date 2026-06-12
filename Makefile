init:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

serve:
	. .venv/bin/activate
	jupyter lab
