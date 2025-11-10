#!/usr/bin/env python3
"""
Self-Healing System Simulation for Jetson Nano
Author: Sam Topping
Description:
Simulates I²C bus health metrics (latency, error rate) across multiple
TCA9548A channels and uses an Isolation Forest model to detect anomalies.
When an anomaly is found, the script "resets" that channel logically.
"""

import numpy as np
import pandas as pd
import time, os
from sklearn.ensemble import IsolationForest
from joblib import dump, load

# --- Simulation Parameters ---
CHANNELS = range(8)     # 8 virtual I²C channels
SAMPLE_COUNT = 20       # samples to build baseline
POLL_INTERVAL = 2       # seconds between scans

# --- Simulated Probe Function ---
def probe_device(channel):
    """
    Simulate device response time and error occurrence.
    Normal channels: ~5 ms latency, ~10% error chance.
    Occasionally injects a large delay to emulate a fault.
    """
    latency = np.random.normal(5, 1)             # mean 5 ms
    error = 1 if np.random.rand() < 0.1 else 0   # 10% error
    if np.random.rand() < 0.05:                  # 5% chance of big fault
        latency += np.random.uniform(20, 60)
        error = 1
    return latency, error

# --- Data Collection ---
def collect_samples():
    rows = []
    for ch in CHANNELS:
        latency, err = probe_device(ch)
        rows.append(dict(channel=ch, latency=latency, errors=err))
    return pd.DataFrame(rows)

# --- Model Training ---
def train_model(df):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(df[['latency', 'errors']])
    dump(model, "iforest.pkl")

# --- Anomaly Detection ---
def detect_anomaly(model, df):
    preds = model.predict(df[['latency', 'errors']])
    for i, ch in enumerate(CHANNELS):
        if preds[i] == -1:
            print(f"[!] Channel {ch} abnormal → resetting (simulated)")
            # In real system: send I²C reset or GPIO pulse here
            os.system("sleep 0.1")

# --- Main Loop ---
if __name__ == "__main__":
    print("=== Self-Healing Simulation Start ===")
    print("Collecting baseline data...")
    base = pd.concat([collect_samples() for _ in range(SAMPLE_COUNT)], ignore_index=True)
    train_model(base)
    model = load("iforest.pkl")
    print("Model trained. Entering live monitoring loop.\n")

    while True:
        df = collect_samples()
        detect_anomaly(model, df)
        print(df)
        time.sleep(POLL_INTERVAL)
