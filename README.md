# Peakflo expense classifier

This is my solution for the Peakflo take-home task. The model predicts the `accountName` of an expense from the item name, description, vendor and amount.

LinearSVC on TF-IDF text features + vendor + amount gets 88.4% accuracy on a random 20% test set (the target was 85%).

## Setup

I used [uv](https://docs.astral.sh/uv/) to manage the packages.

```bash
uv sync
```

If you don't use uv, pip works too:

```bash
pip install -r requirements.txt
```

The data file `accounts-bills.json` is not in this repo. Put it in the root folder before running the notebooks.

## How to run

Run the notebooks in this order:

1. `data_analysis.ipynb`: looks at the data (imbalance, renamed accounts, vendors, amounts)
2. `feature_engineering.ipynb`: cleans the text and builds the features, saves `peakflo_features.parquet`
3. `model_training.ipynb`: trains and evaluates the models, saves the final model to `models/`

```bash
uv run jupyter notebook
```

## API

After training, the saved model can be used through a small FastAPI service.

```bash
uv run uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs. Example request:

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"vendorId": "lL1pcuEf3q6ufBVg2R75", "itemName": "0825 Zoom Workplace Pro Monthly", "itemTotalAmount": 245.5}'
```

It returns the predicted account and the top 3 candidates with scores. There is also `/predict/batch` for a list of bills and `/health` to check the model is loaded.

## Docker

```bash
docker build -t peakflo-classifier .
docker run -p 8000:8000 peakflo-classifier
```

The image only installs the API packages. The notebook tools are in a separate `dev` group.

## Files

```
data_analysis.ipynb        data analysis
feature_engineering.ipynb  feature engineering
model_training.ipynb       training, tuning, evaluation, saving the model
app/features.py            same text cleaning as the notebook, used by the API
app/main.py                FastAPI service
models/                    saved model + its details (accuracy, sklearn version)
Dockerfile                 container for the API
```
