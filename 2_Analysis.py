"""
STEP 1: Comprehensive Dataset Analysis
Analyzes both real-time and benchmark datasets
Compares features, distributions, and statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("="*80)
print("DATASET ANALYSIS & COMPARISON")
print("="*80)
print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ===============================
# LOAD DATASETS
# ===============================
print("\n" + "="*80)
print("LOADING DATASETS")
print("="*80)

print("\n📂 Loading real-time flow dataset...")
df_realtime = pd.read_csv("flow_dataset.csv")
print(f"✅ Loaded: {len(df_realtime)} flows")

print("\n📂 Loading benchmark dataset (DrDoS_DNS)...")
df_benchmark = pd.read_csv("data/DrDoS_DNS.csv")
print(f"✅ Loaded: {len(df_benchmark)} flows")

# ===============================
# DATASET OVERVIEW
# ===============================
print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)

print("\n📊 REAL-TIME DATASET")
print("-" * 40)
print(f"Total Flows:     {len(df_realtime)}")
print(f"Features:        {len(df_realtime.columns)}")
print(f"Feature Names:   {list(df_realtime.columns)}")
print(f"\nClass Distribution:")
print(df_realtime['label'].value_counts())
print(f"\nClass Percentages:")
print(df_realtime['label'].value_counts(normalize=True) * 100)

print("\n📊 BENCHMARK DATASET")
print("-" * 40)
print(f"Total Flows:     {len(df_benchmark)}")
print(f"Features:        {len(df_benchmark.columns)}")
print(f"Feature Names:   {list(df_benchmark.columns)}")
print(f"\nClass Distribution:")
print(df_benchmark['label'].value_counts())
print(f"\nClass Percentages:")
print(df_benchmark['label'].value_counts(normalize=True) * 100)

# ===============================
# FEATURE COMPARISON
# ===============================
print("\n" + "="*80)
print("FEATURE COMPARISON")
print("="*80)

realtime_features = set(df_realtime.columns) - {'flow_id', 'label'}
benchmark_features = set(df_benchmark.columns) - {'label'}

print("\n🔍 REAL-TIME FEATURES (6):")
for feat in sorted(realtime_features):
    print(f"   ✓ {feat}")

print("\n🔍 BENCHMARK FEATURES (15):")
for feat in sorted(benchmark_features):
    print(f"   ✓ {feat}")

# Find common features
common_features = realtime_features.intersection(benchmark_features)
print(f"\n🤝 COMMON FEATURES ({len(common_features)}):")
for feat in sorted(common_features):
    print(f"   ✓ {feat}")

unique_realtime = realtime_features - benchmark_features
unique_benchmark = benchmark_features - realtime_features

print(f"\n📌 UNIQUE TO REAL-TIME ({len(unique_realtime)}):")
for feat in sorted(unique_realtime):
    print(f"   ✓ {feat}")

print(f"\n📌 UNIQUE TO BENCHMARK ({len(unique_benchmark)}):")
for feat in sorted(unique_benchmark):
    print(f"   ✓ {feat}")

# ===============================
# STATISTICAL SUMMARY
# ===============================
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

print("\n📈 REAL-TIME DATASET STATISTICS")
print("-" * 80)
print(df_realtime[list(realtime_features)].describe())

print("\n📈 BENCHMARK DATASET STATISTICS (Common Features)")
print("-" * 80)
if common_features:
    print(df_benchmark[list(common_features)].describe())
else:
    print(df_benchmark[['protocol', 'flow_duration']].describe())

# ===============================
# VISUALIZATIONS
# ===============================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Class Distribution Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real-time
df_realtime['label'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Real-Time Dataset\nClass Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

# Benchmark
df_benchmark['label'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('Benchmark Dataset\nClass Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Class', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('01_class_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 01_class_distribution_comparison.png")
plt.close()

# 2. Feature Distribution - Real-time
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

features_to_plot = ['protocol', 'flow_duration', 'total_packets', 
                    'total_bytes', 'packets_per_second', 'bytes_per_second']

for idx, feature in enumerate(features_to_plot):
    if feature in df_realtime.columns:
        df_realtime[feature].hist(bins=30, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Value', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)

plt.suptitle('Real-Time Dataset - Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_realtime_feature_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 02_realtime_feature_distributions.png")
plt.close()

# 3. Protocol Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real-time protocols
protocol_counts_rt = df_realtime['protocol'].value_counts()
axes[0].bar(protocol_counts_rt.index, protocol_counts_rt.values, color='steelblue', edgecolor='black')
axes[0].set_title('Real-Time Dataset\nProtocol Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Protocol Number', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)

# Benchmark protocols
protocol_counts_bm = df_benchmark['protocol'].value_counts()
axes[1].bar(protocol_counts_bm.index, protocol_counts_bm.values, color='coral', edgecolor='black')
axes[1].set_title('Benchmark Dataset\nProtocol Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Protocol Number', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)

plt.tight_layout()
plt.savefig('03_protocol_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 03_protocol_comparison.png")
plt.close()

# 4. Box plots for common features
if len(common_features) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Real-time
    df_realtime.boxplot(column=list(common_features)[:4], ax=axes[0])
    axes[0].set_title('Real-Time Dataset\nFeature Distributions', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Benchmark
    df_benchmark.boxplot(column=list(common_features)[:4], ax=axes[1])
    axes[1].set_title('Benchmark Dataset\nFeature Distributions', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('04_feature_boxplots.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_feature_boxplots.png")
    plt.close()

# 5. Attack vs Benign comparison (Real-time)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    if feature in df_realtime.columns:
        df_realtime.boxplot(column=feature, by='label', ax=axes[idx])
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Class', fontsize=10)
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].get_figure().suptitle('')

plt.suptitle('Real-Time Dataset - Feature Comparison by Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_realtime_class_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 05_realtime_class_comparison.png")
plt.close()

# ===============================
# SAVE COMPARISON REPORT
# ===============================
print("\n" + "="*80)
print("SAVING COMPARISON REPORT")
print("="*80)

report = []
report.append("="*80)
report.append("DATASET ANALYSIS & COMPARISON REPORT")
report.append("="*80)
report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("\n" + "="*80)
report.append("1. DATASET OVERVIEW")
report.append("="*80)

report.append("\nREAL-TIME DATASET:")
report.append(f"  Total Flows:     {len(df_realtime)}")
report.append(f"  Features:        {len(realtime_features)}")
report.append(f"  DoS Flows:       {(df_realtime['label']=='DoS').sum()} ({(df_realtime['label']=='DoS').sum()/len(df_realtime)*100:.2f}%)")
report.append(f"  Benign Flows:    {(df_realtime['label']=='Benign').sum()} ({(df_realtime['label']=='Benign').sum()/len(df_realtime)*100:.2f}%)")

report.append("\nBENCHMARK DATASET:")
report.append(f"  Total Flows:     {len(df_benchmark)}")
report.append(f"  Features:        {len(benchmark_features)}")
report.append(f"  Attack Flows:    {(df_benchmark['label']=='DrDoS_DNS').sum()} ({(df_benchmark['label']=='DrDoS_DNS').sum()/len(df_benchmark)*100:.2f}%)")
report.append(f"  Benign Flows:    {(df_benchmark['label']=='BENIGN').sum()} ({(df_benchmark['label']=='BENIGN').sum()/len(df_benchmark)*100:.2f}%)")

report.append("\n" + "="*80)
report.append("2. FEATURE COMPARISON")
report.append("="*80)

report.append(f"\nCommon Features ({len(common_features)}):")
for feat in sorted(common_features):
    report.append(f"  ✓ {feat}")

report.append(f"\nUnique to Real-Time ({len(unique_realtime)}):")
for feat in sorted(unique_realtime):
    report.append(f"  ✓ {feat}")

report.append(f"\nUnique to Benchmark ({len(unique_benchmark)}):")
for feat in sorted(unique_benchmark):
    report.append(f"  ✓ {feat}")

report.append("\n" + "="*80)
report.append("3. KEY OBSERVATIONS")
report.append("="*80)

report.append("\n✓ Dataset Sizes:")
report.append(f"  - Benchmark dataset is {len(df_benchmark)/len(df_realtime):.1f}x larger")

report.append("\n✓ Class Balance:")
dos_pct_rt = (df_realtime['label']=='DoS').sum()/len(df_realtime)*100
dos_pct_bm = (df_benchmark['label']=='DrDoS_DNS').sum()/len(df_benchmark)*100
report.append(f"  - Real-time DoS: {dos_pct_rt:.2f}%")
report.append(f"  - Benchmark Attack: {dos_pct_bm:.2f}%")

report.append("\n✓ Feature Sets:")
report.append(f"  - Real-time uses {len(realtime_features)} core flow features")
report.append(f"  - Benchmark uses {len(benchmark_features)} features (including directional stats)")

report.append("\n" + "="*80)
report.append("4. RECOMMENDATIONS")
report.append("="*80)

report.append("\n✓ For Fair Comparison:")
report.append("  1. Reduce benchmark to 6 core features matching real-time")
report.append("  2. Train models on same feature space")
report.append("  3. Evaluate cross-dataset generalization")

report.append("\n✓ Feature Alignment Strategy:")
report.append("  - Map benchmark features to real-time equivalents")
report.append("  - Use: protocol, flow_duration, total_packets, total_bytes, pps, bps")

report.append("\n" + "="*80)

# Save report
with open('dataset_comparison_report.txt', 'w') as f:
    f.write('\n'.join(report))

print("\n✅ Saved: dataset_comparison_report.txt")

# ===============================
# SAVE CSV SUMMARIES
# ===============================
summary_data = {
    'Dataset': ['Real-Time', 'Benchmark'],
    'Total_Flows': [len(df_realtime), len(df_benchmark)],
    'Features': [len(realtime_features), len(benchmark_features)],
    'Attack_Flows': [(df_realtime['label']=='DoS').sum(), (df_benchmark['label']=='DrDoS_DNS').sum()],
    'Benign_Flows': [(df_realtime['label']=='Benign').sum(), (df_benchmark['label']=='BENIGN').sum()],
    'Attack_Percentage': [
        (df_realtime['label']=='DoS').sum()/len(df_realtime)*100,
        (df_benchmark['label']=='DrDoS_DNS').sum()/len(df_benchmark)*100
    ]
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('dataset_summary.csv', index=False)
print("✅ Saved: dataset_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\n📊 Generated Files:")
print("  1. 01_class_distribution_comparison.png")
print("  2. 02_realtime_feature_distributions.png")
print("  3. 03_protocol_comparison.png")
print("  4. 04_feature_boxplots.png")
print("  5. 05_realtime_class_comparison.png")
print("  6. dataset_comparison_report.txt")
print("  7. dataset_summary.csv")

print("\n✅ Step 1 Complete! Proceed to Step 2: Benchmark Preprocessing")