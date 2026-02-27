import pyshark
import pandas as pd
import time

# ===============================
# CONFIG
# ===============================
INTERFACE = "7"          # Ethernet (your confirmed working interface)
TARGET_PACKETS = 10000
DEBUG_EVERY = 100

RAW_CSV = "raw_packets_5000.csv"
CLEAN_CSV = "cleaned_packets_5000.csv"

# ===============================
# CAPTURE PHASE
# ===============================
print("=== PACKET CAPTURE STARTED ===")
print(f"[INFO] Interface: {INTERFACE}")
print(f"[INFO] Target packets: {TARGET_PACKETS}")
print("Generate traffic now...\n")

capture = pyshark.LiveCapture(interface=INTERFACE)

packet_rows = []
seen_packets = 0
ip_packets = 0

try:
    for pkt in capture.sniff_continuously():
        seen_packets += 1

        if not hasattr(pkt, "ip"):
            continue

        try:
            row = {
                "timestamp": float(pkt.sniff_timestamp),
                "protocol": int(pkt.ip.proto),
                "src_ip": pkt.ip.src,
                "dst_ip": pkt.ip.dst,
                "packet_length": int(pkt.length)
            }
        except:
            continue

        packet_rows.append(row)
        ip_packets += 1

        if ip_packets % DEBUG_EVERY == 0:
            print(
                f"[DEBUG] IP #{ip_packets} | "
                f"Proto={row['protocol']} | "
                f"Len={row['packet_length']} | "
                f"{row['src_ip']} -> {row['dst_ip']}"
            )

        if ip_packets >= TARGET_PACKETS:
            break

except KeyboardInterrupt:
    print("\n[WARN] Capture interrupted manually")

finally:
    capture.close()

print(f"\n[INFO] Total packets seen: {seen_packets}")
print(f"[INFO] IP packets captured: {ip_packets}")

# ===============================
# SAVE RAW DATA
# ===============================
raw_df = pd.DataFrame(packet_rows)
raw_df.to_csv(RAW_CSV, index=False)
print(f"[INFO] Raw data saved to {RAW_CSV}")

# ===============================
# CLEANING / PREPROCESSING
# ===============================
print("\n=== PREPROCESSING STARTED ===")

clean_df = raw_df.copy()

# Drop NaNs
before = len(clean_df)
clean_df.dropna(inplace=True)
print(f"[CLEAN] Dropped {before - len(clean_df)} NaN rows")

# Remove zero-length packets
before = len(clean_df)
clean_df = clean_df[clean_df["packet_length"] > 0]
print(f"[CLEAN] Dropped {before - len(clean_df)} zero-length packets")

# Enforce types
clean_df["protocol"] = clean_df["protocol"].astype(int)
clean_df["packet_length"] = clean_df["packet_length"].astype(int)

# Sort by time
clean_df.sort_values(by="timestamp", inplace=True)

clean_df.to_csv(CLEAN_CSV, index=False)
print(f"[INFO] Cleaned data saved to {CLEAN_CSV}")

print("\n=== PIPELINE COMPLETE ===")
