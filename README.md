# Sentiment analysis in the text of reviews with Classical Machine Learning

This project implements an end-to-end sentiment analysis pipeline using TF-IDF and Logistic Regression.
It includes model training, experiment tracking with MLflow, and a FastAPI inference endpoint.

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements
# or
# make setup
```

### Download data
```bash
python -u dataset.py
```

### Run model training and evaluation
```bash
python -u main.py
# or
# make run
```

## FastAPI inference

Run local API:
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Results
Validation Macro F1-score with a strong class imbalance: 0.78

### Figures
<p float="left">
  <img src="https://github.com/danilkos00/sent-analysis/blob/main/figures/pr-curve_1.png?raw=true" width="400"/>
  <img src="https://github.com/danilkos00/sent-analysis/blob/main/figures/pr-curve_0.png?raw=true" width="400"/>
</p>

## Project Structure

```
sent-analysis/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── data_ingestion.py
│   ├── data_processing.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── model.pkl
├── app.py
├── dataset.py
├── main.py
├── config.yml
├── Makefile
├── dockerfile
├── .gitignore
├── requirements.txt
└── README.md
```
