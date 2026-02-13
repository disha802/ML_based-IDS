# 🛡️ Real-Time Intrusion Detection System (IDS) using Machine Learning

## 📌 Project Overview

This project implements a flow-based Intrusion Detection System (IDS)
using supervised machine learning to detect Denial-of-Service (DoS)
attacks from real-time network traffic.

A controlled ICMP flood attack was simulated in a lab environment using
two machines:

-   Attacker Machine -- Generates ICMP flood traffic\
-   Victim Machine -- Captures traffic and runs the IDS pipeline

The captured packet-level data is processed, labeled, converted into
flows, and used for machine learning model training and evaluation.

------------------------------------------------------------------------

## 🎯 Objectives

-   Simulate a controlled ICMP Flood DoS attack
-   Capture real-time network traffic
-   Label packets using temporal attack correlation
-   Convert packets into flow-based features
-   Train and evaluate supervised ML models
-   Compare benchmark dataset model vs real-time trained model

------------------------------------------------------------------------

## ⚙️ System Architecture

Attacker PC\
⬇\
Victim PC (Packet Capture using PyShark)\
⬇\
Packet Cleaning\
⬇\
Time-Based Labeling\
⬇\
Flow Aggregation\
⬇\
Feature Extraction\
⬇\
Machine Learning Model

------------------------------------------------------------------------

## 🧪 Attack Simulation

-   Attack Type: ICMP Echo Request Flood (Network-layer DoS)
-   Method: High-frequency ping requests
-   Environment: Controlled local network
-   Duration: \~20--30 seconds

The attack created a measurable spike in packet rate, verified using
Wireshark I/O graphs.

------------------------------------------------------------------------

## 📡 Packet Capture

Live traffic was captured using:

-   Python
-   PyShark
-   Wireshark (TShark backend)

Captured packet features: - Timestamp - Protocol - Source IP -
Destination IP - Packet Length

------------------------------------------------------------------------

## 🧹 Data Preprocessing

-   Removed non-IP packets
-   Dropped missing values
-   Removed zero-length packets
-   Sorted by timestamp
-   Ensured correct numeric types

------------------------------------------------------------------------

## 🏷 Packet Labeling

Attack traffic was labeled using time-based correlation:

-   Packets during high packet-rate window → DoS
-   Remaining packets → Benign

This follows standard IDS dataset labeling methodology.

------------------------------------------------------------------------

## 🔁 Flow Generation

Packets were aggregated into 1-second network flows defined by:

(Source IP, Destination IP, Protocol, Time Window)

Extracted flow-level features: - Flow duration - Total packets - Total
bytes - Packets per second - Bytes per second - Flow label (majority
vote)

The final dataset is flow-based and suitable for IDS model training.

------------------------------------------------------------------------

## 📊 Output Files

-   cleaned_packets_5000.csv -- Clean packet-level data
-   labeled_packets.csv -- Packet-level labeled dataset
-   flow_dataset.csv -- Final flow-based dataset ready for ML training

------------------------------------------------------------------------

## 🚀 Future Work

-   Train benchmark-dataset IDS model
-   Train real-time dataset IDS model
-   Compare detection performance
-   Deploy real-time streaming IDS pipeline

------------------------------------------------------------------------

## 👩‍💻 Author

Developed as part of an academic project on Machine Learning for Cyber
Security.
