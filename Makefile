.PHONY: install train run docker-build

install:
	python -m venv venv && ./venv/bin/pip install -r projectaeromed/requirements.txt

train:
	python projectaeromed/train.py --data projectaeromed/data/sample_kepler.csv --out projectaeromed/models/best_model.joblib

run:
	streamlit run projectaeromed/app.py
