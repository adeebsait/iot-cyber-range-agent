import os
import time
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from preprocessing import load_and_prepare_icudataset

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict and compute metrics for one model."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = float("nan")
    eval_time = time.time() - t1

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    # throughput & latency
    t2 = time.time()
    model.predict(X_test)
    pred_time = time.time() - t2
    throughput = len(X_test) / pred_time
    latency    = pred_time / len(X_test)

    return {
        'Training(s)': train_time,
        'Inference(s)': eval_time,
        'Accuracy(%)': acc * 100,
        'Precision(%)': prec * 100,
        'Recall(%)': rec * 100,
        'F1(%)': f1 * 100,
        'ROC AUC(%)': auc * 100 if not pd.isna(auc) else None,
        'FPR(%)': fpr * 100,
        'FNR(%)': fnr * 100,
        'Throughput(samps/s)': throughput,
        'Latency(ms/samp)': latency * 1000
    }

def main(data_dir, out_dir='models'):
    # 1. Data prep
    print("Step 1/2: Loading and preprocessing data…")
    X_train, X_test, y_train, y_test = load_and_prepare_icudataset(data_dir)

    # 2. Define your real-world models
    models = {
        'Random Forest':        RandomForestClassifier(
                                    n_estimators=100, n_jobs=-1, random_state=42
                                ),
        'Gradient Boosting':    GradientBoostingClassifier(
                                    n_estimators=100, learning_rate=0.1, random_state=42
                                ),
        'KNN (k=5)':            KNeighborsClassifier(
                                    n_neighbors=5, n_jobs=-1
                                ),
        'AdaBoost':             AdaBoostClassifier(
                                    n_estimators=100, random_state=42
                                ),
        'Logistic Reg':         LogisticRegression(
                                    solver='saga', max_iter=1000, n_jobs=-1, random_state=42
                                ),
        'Decision Tree (weak)': DecisionTreeClassifier(
                                    max_depth=3, random_state=42
                                ),
        'MLP Classifier':       MLPClassifier(
                                    hidden_layer_sizes=(50,50),
                                    max_iter=500,
                                    random_state=42
                                )
    }

    os.makedirs(out_dir, exist_ok=True)

    # 3. Train & eval
    results = []
    print("Step 2/2: Training & evaluating models…")
    for name, model in tqdm(models.items(), desc='Models', unit='model'):
        print(f"\n→ {name}")
        metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        metrics['Model'] = name
        results.append(metrics)
        joblib.dump(model, os.path.join(out_dir, f"{name.replace(' ','_')}.pkl"))
        print(f"✔ Saved {name}")

    # 4. Compile results
    df = pd.DataFrame(results).set_index('Model')

    # 5. Baseline comparisons
    baselines = {
        'Hussain21_RF':      0.995123,
        'Perera24_RF':       0.995500,
        'BoT-IoT_RF':        0.997000,
        'RF-LSTM_Hybrid':    0.940000,
        'ProSRN-ICOM':       0.998500
    }
    for label, base_acc in baselines.items():
        df[f'Δ vs {label} (pp)'] = (df['Accuracy(%)'] / 100 - base_acc) * 100

    # 6. Display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width',     1000)
    pd.set_option('display.precision', 2)
    print("\n=== MODEL COMPARISON ===")
    print(df)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True,
                   help='Path to ICUDatasetProcessed')
    p.add_argument('--out-dir',  default='models',
                   help='Directory to save models')
    args = p.parse_args()
    main(args.data_dir, args.out_dir)
