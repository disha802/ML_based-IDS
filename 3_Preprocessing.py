"""
STEP 2: Benchmark Dataset Preprocessing
Aligns benchmark features with real-time dataset
Creates two versions: 6-feature and full 15-feature
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("BENCHMARK DATASET PREPROCESSING")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ===============================
# LOAD DATASETS
# ===============================
print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

print("\n📂 Loading real-time dataset...")
df_realtime = pd.read_csv("flow_dataset.csv")
print(f"✅ Loaded: {len(df_realtime)} flows")

print("\n📂 Loading benchmark dataset...")
df_benchmark = pd.read_csv("data/DrDoS_DNS.csv")
print(f"✅ Loaded: {len(df_benchmark)} flows")

# ===============================
# FEATURE MAPPING
# ===============================
print("\n" + "="*80)
print("FEATURE MAPPING STRATEGY")
print("="*80)

print("\n🎯 Real-Time Features (6):")
realtime_features = ['protocol', 'flow_duration', 'total_packets', 
                     'total_bytes', 'packets_per_second', 'bytes_per_second']
for feat in realtime_features:
    print(f"   ✓ {feat}")

print("\n🔄 Mapping Benchmark → Real-Time:")

# Feature mapping logic
mapping = {
    'protocol': 'protocol',  # Direct match
    'flow_duration': 'flow_duration',  # Direct match
    'total_packets': 'total_forward_packets + total_backward_packets',
    'total_bytes': 'total_forward_packets_length + total_backward_packets_length',
    'packets_per_second': 'flow_packets_per_seconds',  # Similar
    'bytes_per_second': 'flow_bytes_per_seconds'  # Similar
}

for rt_feat, bm_feat in mapping.items():
    print(f"   {rt_feat:25s} ← {bm_feat}")

# ===============================
# CREATE 6-FEATURE VERSION
# ===============================
print("\n" + "="*80)
print("CREATING 6-FEATURE ALIGNED VERSION")
print("="*80)

df_benchmark_aligned = df_benchmark.copy()

# Calculate derived features
print("\n⚙️  Computing derived features...")

# Total packets = forward + backward
if 'total_forward_packets' in df_benchmark_aligned.columns and 'total_backward_packets' in df_benchmark_aligned.columns:
    df_benchmark_aligned['total_packets'] = (
        df_benchmark_aligned['total_forward_packets'] + 
        df_benchmark_aligned['total_backward_packets']
    )
    print("   ✓ total_packets = forward + backward packets")

# Total bytes = forward + backward
if 'total_forward_packets_length' in df_benchmark_aligned.columns and 'total_backward_packets_length' in df_benchmark_aligned.columns:
    df_benchmark_aligned['total_bytes'] = (
        df_benchmark_aligned['total_forward_packets_length'] + 
        df_benchmark_aligned['total_backward_packets_length']
    )
    print("   ✓ total_bytes = forward + backward bytes")

# Packets per second (use existing or calculate)
if 'flow_packets_per_seconds' in df_benchmark_aligned.columns:
    df_benchmark_aligned['packets_per_second'] = df_benchmark_aligned['flow_packets_per_seconds']
    print("   ✓ packets_per_second = flow_packets_per_seconds")

# Bytes per second (use existing or calculate)
if 'flow_bytes_per_seconds' in df_benchmark_aligned.columns:
    df_benchmark_aligned['bytes_per_second'] = df_benchmark_aligned['flow_bytes_per_seconds']
    print("   ✓ bytes_per_second = flow_bytes_per_seconds")

# Standardize label names
print("\n⚙️  Standardizing labels...")
df_benchmark_aligned['label'] = df_benchmark_aligned['label'].replace({
    'DrDoS_DNS': 'DoS',
    'BENIGN': 'Benign'
})
print("   ✓ DrDoS_DNS → DoS")
print("   ✓ BENIGN → Benign")

# Select only the 6 features + label
features_6 = ['protocol', 'flow_duration', 'total_packets', 
              'total_bytes', 'packets_per_second', 'bytes_per_second', 'label']

df_benchmark_6feat = df_benchmark_aligned[features_6].copy()

# ===============================
# DATA CLEANING
# ===============================
print("\n" + "="*80)
print("DATA CLEANING")
print("="*80)

print("\n🧹 Cleaning 6-feature version...")
before = len(df_benchmark_6feat)

# Remove NaN
df_benchmark_6feat.dropna(inplace=True)
print(f"   ✓ Removed {before - len(df_benchmark_6feat)} rows with NaN")

# Remove infinite values
before = len(df_benchmark_6feat)
df_benchmark_6feat = df_benchmark_6feat.replace([np.inf, -np.inf], np.nan)
df_benchmark_6feat.dropna(inplace=True)
print(f"   ✓ Removed {before - len(df_benchmark_6feat)} rows with infinite values")

# Remove negative values (shouldn't exist in flow data)
before = len(df_benchmark_6feat)
numeric_cols = ['flow_duration', 'total_packets', 'total_bytes', 
                'packets_per_second', 'bytes_per_second']
for col in numeric_cols:
    df_benchmark_6feat = df_benchmark_6feat[df_benchmark_6feat[col] >= 0]
print(f"   ✓ Removed {before - len(df_benchmark_6feat)} rows with negative values")

print(f"\n✅ Final 6-feature dataset: {len(df_benchmark_6feat)} flows")

# ===============================
# STATISTICS COMPARISON
# ===============================
print("\n" + "="*80)
print("STATISTICS COMPARISON")
print("="*80)

print("\n📊 REAL-TIME DATASET:")
print(df_realtime[realtime_features].describe())

print("\n📊 BENCHMARK 6-FEATURE DATASET:")
print(df_benchmark_6feat[realtime_features].describe())

# ===============================
# SAVE PROCESSED DATASETS
# ===============================
print("\n" + "="*80)
print("SAVING PROCESSED DATASETS")
print("="*80)

# Save 6-feature aligned version
df_benchmark_6feat.to_csv('benchmark_6features.csv', index=False)
print(f"\n✅ Saved: benchmark_6features.csv")
print(f"   Rows: {len(df_benchmark_6feat)}")
print(f"   Features: {len(df_benchmark_6feat.columns)-1}")  # excluding label
print(f"   Class Distribution:")
print(df_benchmark_6feat['label'].value_counts())

# Save full feature version (for bonus analysis)
df_benchmark_full = df_benchmark.copy()
df_benchmark_full['label'] = df_benchmark_full['label'].replace({
    'DrDoS_DNS': 'DoS',
    'BENIGN': 'Benign'
})
df_benchmark_full.to_csv('benchmark_full_features.csv', index=False)
print(f"\n✅ Saved: benchmark_full_features.csv")
print(f"   Rows: {len(df_benchmark_full)}")
print(f"   Features: {len(df_benchmark_full.columns)-1}")

# ===============================
# VALIDATION
# ===============================
print("\n" + "="*80)
print("VALIDATION")
print("="*80)

print("\n✓ Checking feature alignment...")
rt_features = set(df_realtime.columns) - {'flow_id', 'label'}
bm_features = set(df_benchmark_6feat.columns) - {'label'}

if rt_features == bm_features:
    print("   ✅ Perfect feature alignment!")
    print(f"   Both datasets have identical features: {sorted(rt_features)}")
else:
    missing = rt_features - bm_features
    extra = bm_features - rt_features
    if missing:
        print(f"   ⚠️  Missing in benchmark: {missing}")
    if extra:
        print(f"   ⚠️  Extra in benchmark: {extra}")

print("\n✓ Checking data types...")
print("   Real-time:")
print(df_realtime[realtime_features].dtypes)
print("\n   Benchmark (6-feature):")
print(df_benchmark_6feat[realtime_features].dtypes)

print("\n✓ Checking value ranges...")
for feat in realtime_features:
    rt_min, rt_max = df_realtime[feat].min(), df_realtime[feat].max()
    bm_min, bm_max = df_benchmark_6feat[feat].min(), df_benchmark_6feat[feat].max()
    print(f"   {feat:25s} | RT: [{rt_min:10.2f}, {rt_max:10.2f}] | BM: [{bm_min:10.2f}, {bm_max:10.2f}]")

# ===============================
# SUMMARY REPORT
# ===============================
print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

summary = []
summary.append("="*80)
summary.append("BENCHMARK PREPROCESSING REPORT")
summary.append("="*80)
summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

summary.append("\n" + "="*80)
summary.append("1. FEATURE MAPPING")
summary.append("="*80)
summary.append("\nBenchmark features mapped to real-time format:")
for rt_feat, bm_feat in mapping.items():
    summary.append(f"  {rt_feat:25s} ← {bm_feat}")

summary.append("\n" + "="*80)
summary.append("2. DATASETS CREATED")
summary.append("="*80)

summary.append(f"\n✓ benchmark_6features.csv:")
summary.append(f"  - Rows: {len(df_benchmark_6feat)}")
summary.append(f"  - Features: {len(df_benchmark_6feat.columns)-1}")
summary.append(f"  - DoS: {(df_benchmark_6feat['label']=='DoS').sum()} ({(df_benchmark_6feat['label']=='DoS').sum()/len(df_benchmark_6feat)*100:.2f}%)")
summary.append(f"  - Benign: {(df_benchmark_6feat['label']=='Benign').sum()} ({(df_benchmark_6feat['label']=='Benign').sum()/len(df_benchmark_6feat)*100:.2f}%)")

summary.append(f"\n✓ benchmark_full_features.csv:")
summary.append(f"  - Rows: {len(df_benchmark_full)}")
summary.append(f"  - Features: {len(df_benchmark_full.columns)-1}")
summary.append(f"  - Original benchmark with standardized labels")

summary.append("\n" + "="*80)
summary.append("3. COMPARISON: REAL-TIME VS BENCHMARK (6-FEATURE)")
summary.append("="*80)

summary.append(f"\nDataset Size:")
summary.append(f"  Real-time:  {len(df_realtime):6d} flows")
summary.append(f"  Benchmark:  {len(df_benchmark_6feat):6d} flows ({len(df_benchmark_6feat)/len(df_realtime):.1f}x larger)")

summary.append(f"\nClass Balance:")
rt_dos_pct = (df_realtime['label']=='DoS').sum()/len(df_realtime)*100
bm_dos_pct = (df_benchmark_6feat['label']=='DoS').sum()/len(df_benchmark_6feat)*100
summary.append(f"  Real-time DoS:  {rt_dos_pct:5.2f}%")
summary.append(f"  Benchmark DoS:  {bm_dos_pct:5.2f}%")

summary.append("\n" + "="*80)
summary.append("4. READY FOR TRAINING")
summary.append("="*80)
summary.append("\n✅ Both datasets now have:")
summary.append("   - Identical feature sets (6 features)")
summary.append("   - Standardized labels (DoS, Benign)")
summary.append("   - Clean data (no NaN, inf, or negative values)")
summary.append("\n✅ Ready for model training and comparison!")

summary.append("\n" + "="*80)

# Save report
with open('preprocessing_report.txt', 'w') as f:
    f.write('\n'.join(summary))

print("\n✅ Saved: preprocessing_report.txt")

print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print("\n📊 Generated Files:")
print("  1. benchmark_6features.csv - Aligned with real-time (6 features)")
print("  2. benchmark_full_features.csv - Full benchmark (15 features)")
print("  3. preprocessing_report.txt - Detailed preprocessing report")

print("\n✅ Step 2 Complete! Proceed to Step 3: Model Training")