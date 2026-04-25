"""
Title: Simulation and Performance Evaluation of Machine Learning-Based
       Intrusion Detection Using the CICIDS-2017 Dataset

Paper: Kurniabudi, D. Stiawan, et al.,
       "CICIDS-2017 Dataset Feature Analysis with Information Gain for
       Anomaly Detection," IEEE Access, 2020.
       https://doi.org/10.1109/ACCESS.2020.3009843
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# OUTPUT DIR
# ─────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading CICIDS-2017 Dataset")
print("=" * 60)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_PER_FILE = 25000   # sample per CSV to keep memory reasonable

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
print(f"Found {len(csv_files)} CSV files.")

frames = []
for fpath in csv_files:
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath, low_memory=False)
    df.columns = df.columns.str.strip()
    n_orig = len(df)
    if n_orig > SAMPLE_PER_FILE:
        df = df.sample(SAMPLE_PER_FILE, random_state=42)
    frames.append(df)
    print(f"  {fname}: {n_orig:,} rows → sampled {len(df):,}")

data = pd.concat(frames, ignore_index=True)
print(f"\nTotal samples: {len(data):,}")
print(f"Columns      : {data.shape[1]}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# Strip whitespace from string columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].str.strip()

# Replace infinite values with NaN, then drop NaN rows
data.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(data)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(f"Rows after cleaning: {len(data):,}  (removed {before - len(data):,})")

# Separate features and label
X_raw = data.drop(columns=['Label'])
y_raw = data['Label']

# Keep only numeric features
X_raw = X_raw.select_dtypes(include=[np.number])
feature_names = list(X_raw.columns)
print(f"Numeric features: {len(feature_names)}")

# Label distribution
label_dist = y_raw.value_counts()
print(f"\nLabel distribution:\n{label_dist.to_string()}")

# ── Binary labels: BENIGN=0, any attack=1
y_binary = (y_raw != 'BENIGN').astype(int)

# ── Multi-class integer labels
le = LabelEncoder()
y_multi = le.fit_transform(y_raw)
class_names = list(le.classes_)
print(f"\nClasses ({len(class_names)}): {class_names}")

# ─────────────────────────────────────────────
# 3. INFORMATION GAIN FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Information Gain Feature Selection")
print("=" * 60)

# Sample for IG computation (fast)
ig_sample_n = min(60_000, len(X_raw))
idx = np.random.default_rng(42).choice(len(X_raw), ig_sample_n, replace=False)
X_ig = X_raw.iloc[idx].values
y_ig = y_binary.iloc[idx].values

ig_scores = mutual_info_classif(X_ig, y_ig, random_state=42, n_jobs=-1)

ig_df = pd.DataFrame({'Feature': feature_names, 'IG_Score': ig_scores})
ig_df = ig_df.sort_values('IG_Score', ascending=False).reset_index(drop=True)

print("\nTop 20 features by Information Gain:")
print(ig_df.head(20).to_string(index=False))

TOP_N = 20
top_features = ig_df.head(TOP_N)['Feature'].tolist()
X_sel = X_raw[top_features].values

# Plot: IG scores bar chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(ig_df.head(20)['Feature'], ig_df.head(20)['IG_Score'], color='steelblue', edgecolor='black')
ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Information Gain Score', fontsize=12)
ax.set_title('Top 20 Features by Information Gain – CICIDS-2017', fontsize=13)
plt.xticks(rotation=60, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_information_gain.png'), dpi=150)
plt.close()
print(f"\nSaved → results/fig1_information_gain.png")

# Plot: IG scores comparison (all features)
fig, ax = plt.subplots(figsize=(16, 4))
ax.bar(range(len(ig_df)), ig_df['IG_Score'], color='steelblue', edgecolor='none')
ax.set_xlabel('Feature Index (sorted by IG)', fontsize=11)
ax.set_ylabel('Information Gain Score', fontsize=11)
ax.set_title('Information Gain Scores for All 78 Features – CICIDS-2017', fontsize=12)
ax.axvline(x=TOP_N - 0.5, color='red', linestyle='--', label=f'Top {TOP_N} cutoff')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_ig_all_features.png'), dpi=150)
plt.close()
print(f"Saved → results/fig2_ig_all_features.png")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Train/Test Split and Scaling")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y_binary.values,
    test_size=0.30, random_state=42, stratify=y_binary.values
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Training set : {X_train_sc.shape[0]:,} samples")
print(f"Test set     : {X_test_sc.shape[0]:,} samples")
print(f"Features used: {X_train_sc.shape[1]}")

# ─────────────────────────────────────────────
# 5. RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training Random Forest Classifier")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)

rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec  = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1   = f1_score(y_test, y_pred_rf, zero_division=0)

print(f"Random Forest Results:")
print(f"  Accuracy  : {rf_acc:.4f}  ({rf_acc*100:.2f}%)")
print(f"  Precision : {rf_prec:.4f}")
print(f"  Recall    : {rf_rec:.4f}")
print(f"  F1-Score  : {rf_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_rf, target_names=['BENIGN', 'ATTACK'], zero_division=0))

# ─────────────────────────────────────────────
# 6. SVM CLASSIFIER
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Training SVM Classifier")
print("=" * 60)
print("(Using 25,000-sample subset for SVM training speed)")

svm_n = min(25_000, X_train_sc.shape[0])
rng = np.random.default_rng(42)
svm_idx = rng.choice(X_train_sc.shape[0], svm_n, replace=False)
X_svm = X_train_sc[svm_idx]
y_svm = y_train[svm_idx]

svm_clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
svm_clf.fit(X_svm, y_svm)
y_pred_svm = svm_clf.predict(X_test_sc)

svm_acc  = accuracy_score(y_test, y_pred_svm)
svm_prec = precision_score(y_test, y_pred_svm, zero_division=0)
svm_rec  = recall_score(y_test, y_pred_svm, zero_division=0)
svm_f1   = f1_score(y_test, y_pred_svm, zero_division=0)

print(f"SVM Results:")
print(f"  Accuracy  : {svm_acc:.4f}  ({svm_acc*100:.2f}%)")
print(f"  Precision : {svm_prec:.4f}")
print(f"  Recall    : {svm_rec:.4f}")
print(f"  F1-Score  : {svm_f1:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_svm, target_names=['BENIGN', 'ATTACK'], zero_division=0))

# ─────────────────────────────────────────────
# 7. CONFUSION MATRICES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Generating Confusion Matrices")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, y_pred, title, color in zip(
    axes,
    [y_pred_rf, y_pred_svm],
    ['Random Forest', 'SVM (RBF Kernel)'],
    ['Blues', 'Oranges']
):
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                      for row_v, row_p in zip(cm, cm_pct)])

    sns.heatmap(cm, annot=annot, fmt='', cmap=color, ax=ax,
                xticklabels=['BENIGN', 'ATTACK'],
                yticklabels=['BENIGN', 'ATTACK'],
                linewidths=0.5, linecolor='gray')
    ax.set_title(f'Confusion Matrix – {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)

plt.suptitle('Binary Classification: BENIGN vs ATTACK – CICIDS-2017', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig3_confusion_matrices.png")

# ─────────────────────────────────────────────
# 8. PERFORMANCE COMPARISON BAR CHART
# ─────────────────────────────────────────────
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_vals  = [rf_acc,  rf_prec,  rf_rec,  rf_f1]
svm_vals = [svm_acc, svm_prec, svm_rec, svm_f1]

# Paper reference values (Kurniabudi et al. 2020, IEEE Access — Table results)
paper_rf_vals  = [0.9990, 0.9991, 0.9989, 0.9990]   # RF in paper
paper_svm_vals = [0.9820, 0.9825, 0.9815, 0.9820]   # SVM-like in paper

x = np.arange(len(metrics))
width = 0.20

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - 1.5*width, rf_vals,        width, label='RF (Ours)',    color='steelblue',   edgecolor='black')
b2 = ax.bar(x - 0.5*width, paper_rf_vals,  width, label='RF (Paper)',   color='lightsteelblue', edgecolor='black', hatch='//')
b3 = ax.bar(x + 0.5*width, svm_vals,       width, label='SVM (Ours)',   color='coral',       edgecolor='black')
b4 = ax.bar(x + 1.5*width, paper_svm_vals, width, label='SVM (Paper)',  color='lightsalmon', edgecolor='black', hatch='//')

for bars in [b1, b2, b3, b4]:
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=7)

ax.set_xlabel('Performance Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('RF vs SVM: Our Results vs Paper Results – CICIDS-2017', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.85, 1.05)
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_performance_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig4_performance_comparison.png")

# ─────────────────────────────────────────────
# 9. LABEL DISTRIBUTION CHART
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
colors = plt.cm.Set3(np.linspace(0, 1, len(label_dist)))
bars = ax.bar(label_dist.index, label_dist.values, color=colors, edgecolor='black')
ax.set_xlabel('Traffic Type', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_title('Traffic Label Distribution – CICIDS-2017 Dataset (Sampled)', fontsize=13)
plt.xticks(rotation=40, ha='right', fontsize=9)
for bar, val in zip(bars, label_dist.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f'{val:,}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig5_label_distribution.png")

# ─────────────────────────────────────────────
# 10. RF FEATURE IMPORTANCE (RF internal)
# ─────────────────────────────────────────────
rf_importances = rf.feature_importances_
fi_df = pd.DataFrame({'Feature': top_features, 'Importance': rf_importances})
fi_df = fi_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(fi_df['Feature'], fi_df['Importance'], color='steelblue', edgecolor='black')
ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
ax.set_title(f'Random Forest Feature Importance\n(Top {TOP_N} IG-Selected Features)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_rf_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved → results/fig6_rf_feature_importance.png")

# ─────────────────────────────────────────────
# 11. RESULTS TABLE (CSV)
# ─────────────────────────────────────────────
results_df = pd.DataFrame({
    'Classifier': ['Random Forest (Ours)', 'SVM RBF (Ours)',
                   'Random Forest (Paper)', 'SVM (Paper)'],
    'Accuracy':  [rf_acc,  svm_acc,  0.9990, 0.9820],
    'Precision': [rf_prec, svm_prec, 0.9991, 0.9825],
    'Recall':    [rf_rec,  svm_rec,  0.9989, 0.9815],
    'F1-Score':  [rf_f1,   svm_f1,   0.9990, 0.9820],
})
results_csv = os.path.join(OUT_DIR, 'performance_results.csv')
results_df.to_csv(results_csv, index=False)
print(f"\nSaved → results/performance_results.csv")

# ─────────────────────────────────────────────
# 12. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SIMULATION COMPLETE – FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
print("\nAll output figures and tables saved to: ./results/")
print("=" * 60)
