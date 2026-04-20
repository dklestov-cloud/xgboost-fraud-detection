'''
scripts/make_submission.py — submission.csv generation for Kaggle.

Example:
    python scripts/make_submission.py \
        --test-transaction ieee-fraud-detection/test_transaction.csv \
        --test-identity ieee-fraud-detection/test_identity.csv \
        --model-dir models \
        --output submission.csv
'''

from __future__ import annotations

import argparse
import os

import pandas as pd

from app.model import load_artifacts, get_artifacts
from app.preprocessing import preprocess_for_inference


def load_and_merge(test_tx_path: str, test_id_path: str | None) -> pd.DataFrame:
    df_tx = pd.read_csv(test_tx_path)

    if test_id_path and os.path.exists(test_id_path):
        df_id = pd.read_csv(test_id_path)
        df = df_tx.merge(df_id, how='left', on='TransactionID')
    else:
        df = df_tx

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-transaction', required=True, help='Path to test_transaction.csv')
    parser.add_argument('--test-identity', default=None, help='Path to test_identity.csv')
    parser.add_argument('--model-dir', default='models', help='Directory with saved model artifacts')
    parser.add_argument('--output', default='submission.csv', help='Output CSV path')
    args = parser.parse_args()

    print(f'Loading artifacts from: {args.model_dir}')
    load_artifacts(args.model_dir)

    arts = get_artifacts()
    if not arts.loaded:
        raise RuntimeError('Model artifacts were not loaded')

    print('Loading test data...')
    raw_df = load_and_merge(args.test_transaction, args.test_identity)

    if 'TransactionID' not in raw_df.columns:
        raise ValueError('Test CSV must contain TransactionID column')

    print('Preprocessing...')
    X = preprocess_for_inference(
        raw_df,
        arts.cat_cols,
        arts.encoders,
        arts.feature_columns,
    )

    print('Predicting...')
    preds = arts.model.predict_proba(X)[:, 1]

    submission = pd.DataFrame(
        {
            'TransactionID': raw_df['TransactionID'].astype(int),
            'isFraud': preds,
        }
    )

    submission.to_csv(args.output, index=False)
    print(f'Saved submission to: {args.output}')
    print(submission.head())


if __name__ == '__main__':
    main()