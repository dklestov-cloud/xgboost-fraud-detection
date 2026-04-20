# Fraud Detection API

FastAPI backend for fraud detection using the **IEEE-CIS Fraud Detection** (Kaggle) dataset.
Model: **XGBoost** (replacing LightGBM from the notebook).

---

## Project Structure

```text
fraud_api/
├── app/
│   ├── main.py            # FastAPI application
│   ├── model.py           # model artifact loading
│   ├── preprocessing.py   # preprocessing pipeline
│   └── schemas.py         # Pydantic schemas
├── scripts/
│   └── train.py           # model training
├── models/                # created after training
│   ├── xgb_model.joblib
│   ├── label_encoders.joblib
│   ├── feature_columns.joblib
│   ├── cat_cols.joblib
│   ├── val_sample.csv     # 500 rows from validation set (with isFraud)
│   └── test_sample.csv    # 500 rows from test set (without isFraud, optional)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Step 1 — Train the Model

Place the data next to the project folder (or specify paths explicitly):

```bash
python scripts/train.py \
    --train-transaction /path/to/train_transaction.csv \
    --train-identity    /path/to/train_identity.csv \
    --test-transaction  /path/to/test_transaction.csv \   # optional
    --test-identity     /path/to/test_identity.csv \      # optional
    --output-dir        models/
```

After training, the `models/` directory will contain all artifacts and two CSV samples for API testing.

---

## Step 2 — Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## Endpoints

| Method | Path           | Description                                 |
| ------ | -------------- | ------------------------------------------- |
| GET    | `/health`      | Service and model status                    |
| POST   | `/predict`     | Prediction from a JSON list of transactions |
| POST   | `/predict/csv` | Prediction from an uploaded CSV             |
| GET    | `/sample`      | S                                           |
