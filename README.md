
Derived features include:
- Flow duration
- Packets per second
- Bytes per second
- Forward and backward packet counts
- Inter-arrival times (IAT)

---

### Step 6: Machine Learning Model Training

Two separate models are trained:

#### Model A: Offline IDS Model
- Trained only on benchmark dataset
- Uses extracted flow features
- Serves as a reference IDS

#### Model B: Real-Time IDS Model
- Trained only on real-time captured data
- Labeled manually as:
  - Benign
  - DoS

Both models use the **same ML algorithm** (e.g., Random Forest or Logistic Regression) for fair comparison.

---

### Step 7: Attack Simulation
A controlled ICMP flood is generated from the attacker machine:
- High-frequency ping requests
- Targeting victim machine
- Conducted in a closed lab environment

---

### Step 8: Deployment and Comparison
Both models are deployed on:
- Live captured traffic
- DoS attack traffic

Evaluation criteria:
- Detection accuracy
- False positives
- Sensitivity to DoS traffic
- Generalization to unseen traffic

---

## Technologies Used
- Python 3
- PyShark
- Wireshark / Tshark
- Pandas, NumPy
- Scikit-learn
- CSV-based data pipelines

---

## Outcome
This project demonstrates:
- The feasibility of building an ML-based IDS using real-time traffic
- Differences between offline-trained and real-time-trained IDS models
- Practical challenges in live IDS deployment

---

## Disclaimer
All attack simulations were performed in a **controlled academic environment** for educational purposes only.
