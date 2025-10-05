# ProjectAeroMed — Exoplanet ML Research Tool (Improved)

**Elevator pitch (2–3 lines):** ProjectAeroMed is a reproducible ML pipeline and Streamlit app that analyzes transit data to classify candidate exoplanet signals using NASA datasets (Kepler/TESS). The project emphasizes reproducibility, explainability (SHAP), and deployability for quick follow-up prioritization.

---

## What this repo contains (high level)
- `scripts/get_nasa_data.py` — script to download and preprocess NASA Exoplanet Archive / Kepler/TESS sample (usage shown below).
- `train.py` — training entrypoint with Stratified K-Fold cross-validation, model selection, and model metadata saving.
- `exoplanet_model.py` — model utilities: fit/predict/evaluate.
- `app.py` — Streamlit app with sidebar controls, model selection, metrics, and SHAP explainability placeholders.
- `requirements.txt` — pinned dependencies for reproducibility.
- `Dockerfile`, `Makefile`, `Procfile` — deployment helpers.
- `models/` — recommended location for saved models (created by training).
- `notebooks/` — optional EDA and results (add your own).

---

## Data used
This project is designed to use NASA public datasets (examples):
- NASA Exoplanet Archive (Kepler Objects of Interest). Recommended CSV export URL (example): `https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html`
- TESS candidate tables (via the Exoplanet Archive)

> **Important:** The `scripts/get_nasa_data.py` script will attempt to download a CSV from a given URL and perform safe preprocessing. If you are offline, you can place a CSV at `data/sample_kepler.csv` (this repo includes a small sample). Always cite dataset URLs and the date you accessed them in your submission.

---

## Quickstart (local)
1. Create a Python env and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. (Optional) Download NASA sample data:
```bash
python scripts/get_nasa_data.py --url "https://your-nasa-csv-url.csv" --out data/nasa_kepler.csv
```
3. Train a model (example):
```bash
python train.py --data data/sample_kepler.csv --out models/best_model.joblib --cv 5
```
This will save `models/best_model.joblib` and `models/model_metadata.json` with metrics and hyperparameters.
4. Run the app:
```bash
streamlit run app.py
```

---

## Reproducibility & notes
- Random seed is fixed in training to improve reproducibility.
- Model metadata (dataset checksum, hyperparameters, CV metrics mean/std) are saved alongside the model.
- For deployment, see `Dockerfile` and `Procfile` (Streamlit Cloud / Heroku).

---

## Limitations & next steps
- Add automated unit tests for preprocessing and edge cases.
- Integrate live NASA API queries and scheduled retraining for new candidate ingestion.
- Add human-in-the-loop review and priority queuing for follow-up telescope observations.

---

## Contact
Project by the ProjectAeroMed team — contact: projectaeromed@gmail.com
Generated/updated: 2025-10-05T09:37:53.519973 UTC
