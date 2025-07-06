# src/evaluate.py

import joblib
from sklearn.metrics import roc_auc_score, confusion_matrix
from preprocessing import load_and_prepare_icudataset


def evaluate(model_path: str, data_dir: str):
    _, X_test, _, y_test = load_and_prepare_icudataset(data_dir)
    model = joblib.load(model_path)

    # Compute probabilities for AUC
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, model.predict(X_test))

    print(f"ROC AUC: {auc:.4f}")
    print("Confusion Matrix:\n", cm)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate trained model on ICU IoT dataset'
    )
    parser.add_argument('--model', required=True, help='Path to .pkl model')
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to ICUDatasetProcessed folder'
    )
    args = parser.parse_args()
    evaluate(args.model, args.data_dir)
