#!/bin/bash
set -e

# Install Python and ML dependencies for LightGBM benchmark
apt-get update -y
apt-get install -y python3 python3-pip python3-venv

python3 -m pip install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy

# Create working directory and write benchmark script
mkdir -p /home/benchmark

cat > /home/benchmark/benchmark.py << 'PYEOF'
import time
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

print("=== LightGBM Credit Card Fraud Detection Benchmark ===")
print(f"LightGBM version: {lgb.__version__}")

# --- Generate synthetic dataset (similar structure to creditcardfraud) ---
t0 = time.time()
np.random.seed(42)
n_samples = 284807
n_features = 30

X = np.random.randn(n_samples, n_features).astype(np.float32)
# Imbalanced classes: ~0.17% fraud (similar to real dataset)
y = np.zeros(n_samples, dtype=np.int32)
fraud_idx = np.random.choice(n_samples, size=492, replace=False)
y[fraud_idx] = 1

df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, n_features)] + ["Amount"])
df["Class"] = y
load_time = time.time() - t0
print(f"Data generated: {n_samples} rows, {n_features} features — {load_time:.2f}s")
print(f"Fraud ratio: {y.mean()*100:.4f}%")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Class", axis=1), df["Class"], test_size=0.2, random_state=42, stratify=df["Class"]
)
dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_test,  label=y_test, reference=dtrain)

# --- Train ---
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
    "n_jobs": -1,
    "verbose": -1,
}

print("\nTraining LightGBM...")
t1 = time.time()
callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)]
model = lgb.train(params, dtrain, num_boost_round=300,
                  valid_sets=[dval], callbacks=callbacks)
train_time = time.time() - t1
print(f"Training done in {train_time:.2f}s | Best iteration: {model.best_iteration}")

# --- Evaluate ---
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int)

auc       = roc_auc_score(y_test, y_prob)
acc       = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

print(f"\n=== Results ===")
print(f"AUC-ROC   : {auc:.6f}")
print(f"Accuracy  : {acc:.6f}")
print(f"F1-Score  : {f1:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")

# --- Inference latency ---
single_row = X_test.iloc[:1]
t2 = time.time()
for _ in range(100):
    model.predict(single_row)
latency_1row = (time.time() - t2) / 100 * 1000

batch_1000 = X_test.iloc[:1000]
t3 = time.time()
model.predict(batch_1000)
throughput_1000 = (time.time() - t3) * 1000

print(f"\nInference latency (1 row, avg 100 runs) : {latency_1row:.3f} ms")
print(f"Inference latency (1000 rows)           : {throughput_1000:.3f} ms")

# --- Save results ---
results = {
    "load_data_time_s":   round(load_time, 4),
    "training_time_s":    round(train_time, 4),
    "best_iteration":     model.best_iteration,
    "auc_roc":            round(auc, 6),
    "accuracy":           round(acc, 6),
    "f1_score":           round(f1, 6),
    "precision":          round(precision, 6),
    "recall":             round(recall, 6),
    "inference_1row_ms":  round(latency_1row, 4),
    "inference_1000rows_ms": round(throughput_1000, 4),
}

with open("/home/benchmark/benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nbenchmark_result.json saved.")
print("=== Benchmark Complete ===")
PYEOF

chmod +x /home/benchmark/benchmark.py
echo "Setup complete. Run: python3 /home/benchmark/benchmark.py"
