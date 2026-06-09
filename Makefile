init:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

activate:
	. .venv/bin/activate

serve:
	jupyter lab
