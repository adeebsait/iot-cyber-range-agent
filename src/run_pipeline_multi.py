import os
import time
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocessing import load_and_prepare_icudataset

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict and compute metrics for one model."""
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict & probabilities
    t1 = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    eval_time = time.time() - t1

    # Compute metrics
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    # Throughput & latency
    t2 = time.time()
    model.predict(X_test)
    pred_time = time.time() - t2
    throughput = len(X_test) / pred_time
    latency    = pred_time / len(X_test)

    return {
        'Training(s)': train_time,
        'Inference(s)': eval_time,
        'Accuracy(%)': acc*100,
        'Precision(%)': prec*100,
        'Recall(%)': rec*100,
        'F1(%)': f1*100,
        'ROC AUC(%)': auc*100,
        'FPR(%)': fpr*100,
        'FNR(%)': fnr*100,
        'Throughput(samps/s)': throughput,
        'Latency(ms/samp)': latency*1000
    }

def main(data_dir, out_dir='models'):
    # 1. Prepare data
    print("Step 1/2: Loading and preprocessing data…")
    X_train, X_test, y_train, y_test = load_and_prepare_icudataset(data_dir)

    # 2. Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'KNN (k=5)'  : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'AdaBoost'   : AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)'  : SVC(kernel='rbf', probability=True, random_state=42),
        'LogReg'     : LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    }

    os.makedirs(out_dir, exist_ok=True)

    # 3. Train & evaluate each model with progress bar
    results = []
    print("Step 2/2: Training & evaluating models…")
    for name, model in tqdm(models.items(), desc='Models', unit='model'):
        print(f"\n→ {name}")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        metrics['Model'] = name
        results.append(metrics)
        # Save the trained model
        joblib.dump(model, os.path.join(out_dir, f"{name.replace(' ','_')}.pkl"))
        print(f"✔ Saved {name} to disk")

    # 4. Compile results DataFrame
    df = pd.DataFrame(results).set_index('Model')

    # 5. Baseline comparisons
    baselines = {
        'Hussain et al. (2021 RF)': 0.995123,
        'Perera et al. (2024 RF)': 0.99550
    }
    for label, b_acc in baselines.items():
        col = f'Δ vs {label}(pp)'
        df[col] = df['Accuracy(%)']/100 - b_acc
        df[col] *= 100

    # 6. Display final table
    pd.set_option('display.precision', 2)
    print("\n=== MODEL COMPARISON ===")
    print(df)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-model pipeline with tqdm progress"
    )
    parser.add_argument(
        '--data-dir', required=True,
        help='Path to ICUDatasetProcessed folder'
    )
    parser.add_argument(
        '--out-dir', default='models',
        help='Directory to save trained models'
    )
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
