"""
FIXED FLOW GENERATION PIPELINE
Optimized for your specific dataset characteristics
Handles extreme class imbalance and maximizes flow generation
"""

import pandas as pd
import numpy as np

# ===============================
# CONFIG
# ===============================
INPUT_CSV = "cleaned_packets_5000.csv"
OUTPUT_PACKET_CSV = "labeled_packets_final.csv"
OUTPUT_FLOW_CSV = "flow_dataset_final.csv"

# More aggressive settings to generate MORE flows
FLOW_WINDOW = 0.5  # Use 0.5 second windows
BIDIRECTIONAL = True  # Treat A→B and B→A as same flow
MIN_PACKETS_PER_FLOW = 1  # Don't filter out single packets

print("="*80)
print("OPTIMIZED FLOW GENERATION PIPELINE")
print("="*80)

print("\n📂 Loading data...")
df = pd.read_csv(INPUT_CSV)
print(f"✅ Loaded: {len(df)} packets")

# ===============================
# STEP 1: IDENTIFY ATTACK WINDOW
# ===============================
print("\n" + "="*80)
print("STEP 1: IDENTIFYING ATTACK WINDOW")
print("="*80)

df["second"] = df["timestamp"].astype(int)

# Find packet rate per second
pps = df.groupby("second").size().sort_values(ascending=False)
print(f"\nTop 10 packet-rate seconds:")
print(pps.head(10))

# Calculate statistics for better threshold
mean_pps = pps.mean()
std_pps = pps.std()
threshold = mean_pps + 1.5 * std_pps  # 1.5 std above mean

print(f"\n📊 Packet rate statistics:")
print(f"   Mean PPS: {mean_pps:.2f}")
print(f"   Std PPS:  {std_pps:.2f}")
print(f"   Threshold: {threshold:.2f} (1.5σ above mean)")

# Identify attack seconds (above threshold)
attack_seconds = pps[pps > threshold].index.tolist()

if len(attack_seconds) > 0:
    ATTACK_START = min(attack_seconds)
    ATTACK_END = max(attack_seconds)
    print(f"\n📍 Attack window detected:")
    print(f"   Start: {ATTACK_START}")
    print(f"   End:   {ATTACK_END}")
    print(f"   Duration: {ATTACK_END - ATTACK_START} seconds")
    print(f"   Attack seconds: {len(attack_seconds)}")
else:
    print(f"\n⚠️  No clear attack spike detected, using top 5 seconds")
    attack_seconds = pps.head(5).index.tolist()
    ATTACK_START = min(attack_seconds)
    ATTACK_END = max(attack_seconds)

# ===============================
# STEP 2: LABEL PACKETS
# ===============================
print("\n" + "="*80)
print("STEP 2: LABELING PACKETS")
print("="*80)

df["label"] = "Benign"
df.loc[df["second"].isin(attack_seconds), "label"] = "DoS"

print(f"\n📊 Packet label distribution:")
print(df["label"].value_counts())
dos_pct = (df['label']=='DoS').sum()/len(df)*100
benign_pct = (df['label']=='Benign').sum()/len(df)*100
print(f"   DoS:    {(df['label']=='DoS').sum()} ({dos_pct:.2f}%)")
print(f"   Benign: {(df['label']=='Benign').sum()} ({benign_pct:.2f}%)")

# Save labeled packets
df.to_csv(OUTPUT_PACKET_CSV, index=False)
print(f"\n✅ Saved: {OUTPUT_PACKET_CSV}")

# ===============================
# STEP 3: GENERATE FLOWS (OPTIMIZED)
# ===============================
print("\n" + "="*80)
print("STEP 3: GENERATING FLOWS (OPTIMIZED)")
print("="*80)

print(f"""
🔧 Configuration:
   Window size: {FLOW_WINDOW}s
   Bidirectional: {BIDIRECTIONAL}
   Min packets: {MIN_PACKETS_PER_FLOW}
""")

# Sort by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

def create_flow_id(row, window_size=0.5, bidirectional=True):
    """Create flow ID"""
    time_window = int(row['timestamp'] / window_size)
    
    if bidirectional:
        # Sort IPs to make bidirectional
        ip1, ip2 = sorted([row['src_ip'], row['dst_ip']])
        return f"{ip1}_{ip2}_{row['protocol']}_{time_window}"
    else:
        # Keep direction
        return f"{row['src_ip']}_{row['dst_ip']}_{row['protocol']}_{time_window}"

# Create flow IDs
df['flow_id'] = df.apply(lambda row: create_flow_id(row, FLOW_WINDOW, BIDIRECTIONAL), axis=1)

print(f"📊 Unique flow IDs generated: {df['flow_id'].nunique()}")

# Aggregate packets into flows
print(f"\n🔄 Aggregating packets into flows...")
flows = df.groupby('flow_id').agg(
    protocol=('protocol', 'first'),
    src_ip=('src_ip', 'first'),
    dst_ip=('dst_ip', 'first'),
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    flow_duration=('timestamp', lambda x: x.max() - x.min()),
    total_packets=('timestamp', 'count'),
    total_bytes=('packet_length', 'sum'),
    label=('label', lambda x: x.value_counts().idxmax())
).reset_index()

# Calculate derived features
flows['packets_per_second'] = np.where(
    flows['flow_duration'] > 0,
    flows['total_packets'] / flows['flow_duration'],
    flows['total_packets']
)

flows['bytes_per_second'] = np.where(
    flows['flow_duration'] > 0,
    flows['total_bytes'] / flows['flow_duration'],
    flows['total_bytes']
)

print(f"✅ Generated: {len(flows)} flows")

# ===============================
# STEP 4: FILTER FLOWS (MINIMAL)
# ===============================
print("\n" + "="*80)
print("STEP 4: FILTERING FLOWS (KEEPING MORE DATA)")
print("="*80)

original_count = len(flows)

# Only remove if less than MIN_PACKETS_PER_FLOW
if MIN_PACKETS_PER_FLOW > 1:
    flows = flows[flows['total_packets'] >= MIN_PACKETS_PER_FLOW]
    print(f"✓ Removed {original_count - len(flows)} flows with <{MIN_PACKETS_PER_FLOW} packets")
else:
    print(f"✓ Keeping ALL flows (min_packets={MIN_PACKETS_PER_FLOW})")

print(f"\n✅ Final flow count: {len(flows)}")

# ===============================
# STEP 5: ANALYZE CLASS IMBALANCE
# ===============================
print("\n" + "="*80)
print("STEP 5: CLASS DISTRIBUTION ANALYSIS")
print("="*80)

print(f"\n📊 Flow label distribution:")
label_counts = flows['label'].value_counts()
print(label_counts)

dos_flows = (flows['label']=='DoS').sum()
benign_flows = (flows['label']=='Benign').sum()
dos_flow_pct = dos_flows/len(flows)*100
benign_flow_pct = benign_flows/len(flows)*100

print(f"\n   DoS:    {dos_flows} ({dos_flow_pct:.2f}%)")
print(f"   Benign: {benign_flows} ({benign_flow_pct:.2f}%)")

if benign_flow_pct < 10:
    print(f"\n⚠️  WARNING: Severe class imbalance detected!")
    print(f"   Only {benign_flow_pct:.1f}% Benign flows")
    print(f"   This will require SMOTE or special handling during training")

print(f"\n📈 Flow characteristics:")
print(f"   Avg packets per flow: {flows['total_packets'].mean():.2f}")
print(f"   Avg bytes per flow: {flows['total_bytes'].mean():.2f}")
print(f"   Avg flow duration: {flows['flow_duration'].mean():.4f}s")

# Protocol distribution
print(f"\n🌐 Protocol distribution:")
print(flows['protocol'].value_counts())

# ===============================
# STEP 6: SAVE FINAL DATASET
# ===============================
print("\n" + "="*80)
print("STEP 6: SAVING FINAL FLOW DATASET")
print("="*80)

# Create final feature set
df_flows_final = flows[[
    'protocol', 'flow_duration', 'total_packets', 'total_bytes',
    'packets_per_second', 'bytes_per_second', 'label'
]].copy()

# Ensure proper types
df_flows_final['protocol'] = df_flows_final['protocol'].astype(int)
df_flows_final['total_packets'] = df_flows_final['total_packets'].astype(int)
df_flows_final['total_bytes'] = df_flows_final['total_bytes'].astype(int)

# Sort by protocol and then by total_packets (for consistent ordering)
df_flows_final = df_flows_final.sort_values(['protocol', 'total_packets'], 
                                             ascending=[True, False]).reset_index(drop=True)

df_flows_final.to_csv(OUTPUT_FLOW_CSV, index=False)
print(f"\n✅ Saved: {OUTPUT_FLOW_CSV}")
print(f"   Rows: {len(df_flows_final)}")
print(f"   Features: {len(df_flows_final.columns) - 1}")

# ===============================
# RECOMMENDATIONS
# ===============================
print("\n" + "="*80)
print("RECOMMENDATIONS FOR TRAINING")
print("="*80)

print(f"""
📋 Your Dataset Characteristics:
   ✓ Total flows: {len(df_flows_final)}
   ✓ DoS flows:   {dos_flows} ({dos_flow_pct:.1f}%)
   ✓ Benign flows: {benign_flows} ({benign_flow_pct:.1f}%)
   ⚠️  SEVERE class imbalance

🎯 Recommended Training Approach:

1. COMBINE WITH BENCHMARK DATA:
   - Your data is heavily skewed toward DoS
   - Benchmark has more Benign samples
   - Hybrid approach will create balance

2. USE AGGRESSIVE SMOTE:
   - Oversample minority class (Benign) heavily
   - Target 50:50 ratio after SMOTE

3. USE CLASS WEIGHTS:
   - Set class_weight='balanced' in all models
   - Penalizes majority class mistakes

4. STRATIFIED SAMPLING:
   - Always use stratified split
   - Ensures both classes in train/test

5. EVALUATION METRICS:
   - Don't trust accuracy (will be ~97% by predicting all DoS)
   - Focus on: Precision, Recall, F1-Score for BOTH classes
   - Check confusion matrix carefully

Next Step: Run HYBRID_training.py to combine with benchmark data
""")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\n📁 Output files:")
print(f"   1. {OUTPUT_PACKET_CSV} - Labeled packets")
print(f"   2. {OUTPUT_FLOW_CSV} - Flow dataset ({len(df_flows_final)} flows)")
print(f"\n✅ Ready for hybrid training!")