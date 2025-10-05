# train.py - CLI training utility with StratifiedKFold CV and model metadata saving
import argparse, json, os
import pandas as pd, numpy as np
from exoplanet_model import load_data, preprocess, train_and_select, evaluate_model, save_model, dataset_checksum
from sklearn.model_selection import StratifiedKFold
import joblib, datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Input CSV file')
    parser.add_argument('--out', required=True, help='Output model path (joblib)')
    parser.add_argument('--model', default='random_forest', help='Model to train (random_forest|logistic)')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()

    df = load_data(args.data)
    X, y, features = preprocess(df)
    print(f'Data: {len(df)} rows, Features: {len(features)}')

    # Evaluate baseline with chosen model via cross-validated predictions
    print('Evaluating model with cross-validation...')
    clf = train_and_select(X, y, model_name=args.model)
    metrics, probs, preds = evaluate_model(clf, X, y)

    metadata = {
        'model_name': args.model,
        'trained_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'metrics': metrics,
        'features': features,
        'dataset_checksum': dataset_checksum(args.data),
        'n_samples': int(len(y)),
    }

    # Save final model and metadata
    save_model(clf, args.out, metadata)
    print('Saved model to', args.out)
    print('Metadata written. Summary:')
    print(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
