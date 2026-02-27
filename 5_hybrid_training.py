"""
UPDATED HYBRID TRAINING - HANDLES SEVERE IMBALANCE
Specifically designed for datasets with 96% DoS, 4% Benign
Combines with benchmark to create perfect balance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID TRAINING - BALANCED APPROACH")
print("Handles Severe Class Imbalance")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ===============================
# LOAD DATASETS
# ===============================
print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

print("\n📂 Loading real-time dataset...")
try:
    df_realtime = pd.read_csv("flow_dataset_final.csv")
    print(f"✅ Using flow_dataset_final.csv")
except FileNotFoundError:
    try:
        df_realtime = pd.read_csv("flow_dataset_improved.csv")
        print(f"✅ Using flow_dataset_improved.csv")
        # Drop extra columns if present
        if 'avg_packet_size' in df_realtime.columns:
            df_realtime = df_realtime.drop('avg_packet_size', axis=1)
        if 'window_size' in df_realtime.columns:
            df_realtime = df_realtime.drop('window_size', axis=1)
    except FileNotFoundError:
        df_realtime = pd.read_csv("flow_dataset.csv")
        print(f"✅ Using original flow_dataset.csv")
        if 'flow_id' in df_realtime.columns:
            df_realtime = df_realtime.drop('flow_id', axis=1)

print(f"   Flows: {len(df_realtime)}")
print(f"   Distribution:")
print(df_realtime['label'].value_counts())

print("\n📂 Loading benchmark dataset...")
df_benchmark = pd.read_csv("benchmark_6features.csv")
print(f"   Flows: {len(df_benchmark)}")
print(f"   Distribution:")
print(df_benchmark['label'].value_counts())

# ===============================
# ANALYZE IMBALANCE
# ===============================
print("\n" + "="*80)
print("ANALYZING CLASS IMBALANCE")
print("="*80)

rt_dos = (df_realtime['label']=='DoS').sum()
rt_benign = (df_realtime['label']=='Benign').sum()
rt_dos_pct = rt_dos / len(df_realtime) * 100

bm_dos = (df_benchmark['label']=='DoS').sum()
bm_benign = (df_benchmark['label']=='Benign').sum()
bm_dos_pct = bm_dos / len(df_benchmark) * 100

print(f"\n📊 Real-Time Dataset:")
print(f"   DoS:    {rt_dos} ({rt_dos_pct:.1f}%)")
print(f"   Benign: {rt_benign} ({100-rt_dos_pct:.1f}%)")
if rt_dos_pct > 90 or rt_dos_pct < 10:
    print(f"   ⚠️  SEVERE IMBALANCE!")

print(f"\n📊 Benchmark Dataset:")
print(f"   DoS:    {bm_dos} ({bm_dos_pct:.1f}%)")
print(f"   Benign: {bm_benign} ({100-bm_dos_pct:.1f}%)")

# ===============================
# CREATE BALANCED HYBRID DATASET
# ===============================
print("\n" + "="*80)
print("CREATING BALANCED HYBRID DATASET")
print("="*80)

print(f"""
🎯 Strategy for Severe Imbalance:
   1. Use ALL real-time data (preserve your patterns)
   2. Supplement with benchmark Benign samples
   3. Undersample benchmark DoS to match
   4. Target: 50-50 balance
""")

# Separate by class
rt_dos_all = df_realtime[df_realtime['label'] == 'DoS']
rt_benign_all = df_realtime[df_realtime['label'] == 'Benign']
bm_dos_all = df_benchmark[df_benchmark['label'] == 'DoS']
bm_benign_all = df_benchmark[df_benchmark['label'] == 'Benign']

print(f"\n📦 Available samples:")
print(f"   RT DoS:    {len(rt_dos_all)}")
print(f"   RT Benign: {len(rt_benign_all)}")
print(f"   BM DoS:    {len(bm_dos_all)}")
print(f"   BM Benign: {len(bm_benign_all)}")

# Strategy: Use ALL real-time + balance with benchmark
print(f"\n🔧 Building balanced dataset...")

# Use all real-time data
rt_sample = df_realtime.copy()
print(f"   Using ALL real-time: {len(rt_sample)} flows")

# Calculate how much benchmark we need
rt_total = len(rt_sample)
rt_dos_count = len(rt_dos_all)
rt_benign_count = len(rt_benign_all)

# We want 50-50 final balance
# Target total flows
target_total = 2500  # Reasonable size

# Calculate target for each class
target_per_class = target_total // 2

# How many more of each class do we need?
dos_needed = max(0, target_per_class - rt_dos_count)
benign_needed = max(0, target_per_class - rt_benign_count)

print(f"\n   Target per class: {target_per_class}")
print(f"   Need from benchmark:")
print(f"      DoS:    {dos_needed}")
print(f"      Benign: {benign_needed}")

# Sample from benchmark
if dos_needed > 0 and len(bm_dos_all) > 0:
    bm_dos_sample = bm_dos_all.sample(n=min(dos_needed, len(bm_dos_all)), 
                                      random_state=RANDOM_STATE)
else:
    bm_dos_sample = pd.DataFrame(columns=bm_dos_all.columns)

if benign_needed > 0 and len(bm_benign_all) > 0:
    bm_benign_sample = bm_benign_all.sample(n=min(benign_needed, len(bm_benign_all)), 
                                            random_state=RANDOM_STATE)
else:
    bm_benign_sample = pd.DataFrame(columns=bm_benign_all.columns)

print(f"   Actually sampled:")
print(f"      DoS:    {len(bm_dos_sample)}")
print(f"      Benign: {len(bm_benign_sample)}")

# Combine all
df_hybrid = pd.concat([
    rt_sample,
    bm_dos_sample,
    bm_benign_sample
], ignore_index=True)

# Shuffle
df_hybrid = df_hybrid.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"\n✅ Hybrid dataset created:")
print(f"   Total: {len(df_hybrid)} flows")
print(f"   Distribution:")
final_counts = df_hybrid['label'].value_counts()
print(final_counts)
final_dos_pct = (df_hybrid['label']=='DoS').sum() / len(df_hybrid) * 100
print(f"   DoS:    {(df_hybrid['label']=='DoS').sum()} ({final_dos_pct:.1f}%)")
print(f"   Benign: {(df_hybrid['label']=='Benign').sum()} ({100-final_dos_pct:.1f}%)")

# ===============================
# PREPARE FOR TRAINING  
# ===============================
print("\n" + "="*80)
print("PREPARING DATA FOR TRAINING")
print("="*80)

X = df_hybrid.drop('label', axis=1)
y = df_hybrid['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_encoded
)

print(f"\n📊 Train/Test Split:")
print(f"   Train: {len(X_train)}")
print(f"   Test:  {len(X_test)}")

# Check training distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n   Training distribution:")
for label, count in zip(unique, counts):
    pct = count/len(y_train)*100
    print(f"      {le.inverse_transform([label])[0]}: {count} ({pct:.1f}%)")

# Apply SMOTE to achieve perfect balance
print(f"\n🔄 Applying SMOTE for perfect 50-50 balance...")
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, min(counts)-1))
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

unique, counts = np.unique(y_train_balanced, return_counts=True)
print(f"   After SMOTE:")
for label, count in zip(unique, counts):
    pct = count/len(y_train_balanced)*100
    print(f"      {le.inverse_transform([label])[0]}: {count} ({pct:.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# ===============================
# TRAIN 3 MODELS (OPTIMIZED)
# ===============================
print("\n" + "="*80)
print("TRAINING 3 OPTIMIZED MODELS")
print("="*80)

models = {}

# Random Forest
print("\n--- Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train_balanced)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
auc_rf = roc_auc_score(y_test, y_proba_rf[:, 1])

print(f"Accuracy:  {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"F1-Score:  {f1_rf:.4f}")
print(f"ROC-AUC:   {auc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_rf = cross_val_score(rf, X_train_scaled, y_train_balanced, cv=cv)
print(f"CV: {cv_scores_rf.mean():.4f} (±{cv_scores_rf.std():.4f})")

models['Random Forest'] = {
    'model': rf, 'acc': acc_rf, 'f1': f1_rf, 'auc': auc_rf,
    'cv_mean': cv_scores_rf.mean(), 'cv_std': cv_scores_rf.std(),
    'pred': y_pred_rf, 'proba': y_proba_rf
}

# Gradient Boosting
print("\n--- Gradient Boosting ---")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)
gb.fit(X_train_scaled, y_train_balanced)
y_pred_gb = gb.predict(X_test_scaled)
y_proba_gb = gb.predict_proba(X_test_scaled)

acc_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb, average='weighted')
auc_gb = roc_auc_score(y_test, y_proba_gb[:, 1])

print(f"Accuracy:  {acc_gb:.4f} ({acc_gb*100:.2f}%)")
print(f"F1-Score:  {f1_gb:.4f}")
print(f"ROC-AUC:   {auc_gb:.4f}")
print(classification_report(y_test, y_pred_gb, target_names=le.classes_))

cv_scores_gb = cross_val_score(gb, X_train_scaled, y_train_balanced, cv=cv)
print(f"CV: {cv_scores_gb.mean():.4f} (±{cv_scores_gb.std():.4f})")

models['Gradient Boosting'] = {
    'model': gb, 'acc': acc_gb, 'f1': f1_gb, 'auc': auc_gb,
    'cv_mean': cv_scores_gb.mean(), 'cv_std': cv_scores_gb.std(),
    'pred': y_pred_gb, 'proba': y_proba_gb
}

# SVM
print("\n--- SVM ---")
svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=RANDOM_STATE
)
svm.fit(X_train_scaled, y_train_balanced)
y_pred_svm = svm.predict(X_test_scaled)
y_proba_svm = svm.predict_proba(X_test_scaled)

acc_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
auc_svm = roc_auc_score(y_test, y_proba_svm[:, 1])

print(f"Accuracy:  {acc_svm:.4f} ({acc_svm*100:.2f}%)")
print(f"F1-Score:  {f1_svm:.4f}")
print(f"ROC-AUC:   {auc_svm:.4f}")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

cv_scores_svm = cross_val_score(svm, X_train_scaled, y_train_balanced, cv=cv)
print(f"CV: {cv_scores_svm.mean():.4f} (±{cv_scores_svm.std():.4f})")

models['SVM'] = {
    'model': svm, 'acc': acc_svm, 'f1': f1_svm, 'auc': auc_svm,
    'cv_mean': cv_scores_svm.mean(), 'cv_std': cv_scores_svm.std(),
    'pred': y_pred_svm, 'proba': y_proba_svm
}

# ===============================
# SELECT BEST & SAVE
# ===============================
print("\n" + "="*80)
print("MODEL COMPARISON & SELECTION")
print("="*80)

comparison = []
for name, m in models.items():
    comparison.append({
        'Model': name,
        'Accuracy': f"{m['acc']:.4f}",
        'F1-Score': f"{m['f1']:.4f}",
        'ROC-AUC': f"{m['auc']:.4f}",
        'CV-Mean': f"{m['cv_mean']:.4f}",
        'CV-Std': f"{m['cv_std']:.4f}"
    })

df_comp = pd.DataFrame(comparison)
df_comp.to_csv('FINAL_model_comparison.csv', index=False)
print("\n" + df_comp.to_string(index=False))

# Best model
best_name = max(models.items(), key=lambda x: x[1]['f1'])[0]
best_model = models[best_name]['model']

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   F1-Score: {models[best_name]['f1']:.4f}")
print(f"   CV: {models[best_name]['cv_mean']:.4f} (±{models[best_name]['cv_std']:.4f})")

# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

for name, m in models.items():
    fname = f"FINAL_{name.lower().replace(' ', '_')}_model.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(m['model'], f)
    print(f"✅ {fname}")

with open('FINAL_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('FINAL_label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

feature_names = list(X.columns)
with open('FINAL_feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

with open('BEST_FINAL_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"✅ BEST_FINAL_model.pkl (deployment-ready)")

print("\n" + "="*80)
print("SUCCESS!")
print("="*80)
print(f"""
✅ Trained 3 models on balanced hybrid dataset
   Total flows: {len(df_hybrid)}
   Balance: {final_dos_pct:.1f}% DoS, {100-final_dos_pct:.1f}% Benign
   
🏆 Best: {best_name} ({models[best_name]['f1']*100:.1f}% F1)
   CV validated: {models[best_name]['cv_mean']:.3f} ±{models[best_name]['cv_std']:.3f}
   
🚀 Ready for real-time deployment!
   Use: BEST_FINAL_model.pkl
   With: FINAL_scaler.pkl, FINAL_label_encoder.pkl
""")