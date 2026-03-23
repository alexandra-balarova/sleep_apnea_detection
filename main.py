import wfdb
import matplotlib.pyplot as plt
import numpy as np

record = wfdb.rdrecord('apnea-ecg-database-1.0.0/a01')
annotation = wfdb.rdann('apnea-ecg-database-1.0.0/a01', 'apn')

sampling_rate = record.fs
labels = annotation.symbol
labels_numeric = [1 if l == 'A' else 0 for l in labels]

ecg = record.p_signal[:, 0]
fs = record.fs

time = np.arange(len(ecg)) / fs

# Plot first 5 minutes
minutes = 1
samples = minutes * 10 * fs

plt.figure(figsize=(14, 5))
plt.plot(time[:samples], ecg[:samples], label="ECG")

# Mark apnea regions
for i, label in enumerate(annotation.symbol[:minutes]):
    if label == 'A':  # apnea
        start = i * 60
        plt.axvspan(start, start + 60, color='red', alpha=0.2)

plt.title("ECG Signal with Apnea Regions (Red)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

import numpy as np

samples_per_minute = 60 * sampling_rate

segments = []
for i in range(len(labels_numeric)):
    start = i * samples_per_minute
    end = start + samples_per_minute
    segment = ecg[start:end]
    segments.append(segment)

import pandas as pd

features = []

for segment in segments:
    features.append({
        "mean": np.mean(segment),
        "std": np.std(segment),
        "max": np.max(segment),
        "min": np.min(segment)
    })

df = pd.DataFrame(features)
df["label"] = labels_numeric

df.to_csv("apnea_features.csv", index=False)
