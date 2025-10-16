python = venv/bin/python
pip = venv/bin/pip

setup:
	python3 -m venv venv
	$(python) -m pip install --upgrade pip
	$(pip) install -r requirements.txt

run:
	$(python) main.py

mlflow:
	venv/bin/mlflow ui
  
clean:
	rm -rf src/__pycache__
	rm -rf __pycache__

remove:
	rm -rf venv