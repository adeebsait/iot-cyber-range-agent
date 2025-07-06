# src/train_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing import load_and_prepare_icudataset


def train(input_dir: str, model_out: str):
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_icudataset(input_dir)

    # Initialise and train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=['Benign', 'Attack']
    ))

    # Save the trained model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train RF model on ICU IoT dataset'
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to ICUDatasetProcessed folder'
    )
    parser.add_argument(
        '--out', default='models/rf_model.pkl',
        help='Where to save trained model'
    )
    args = parser.parse_args()
    train(args.data_dir, args.out)
