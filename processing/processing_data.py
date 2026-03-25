# ================================
# IMPORT REQUIRED LIBRARIES
# ================================

import os                          # For file and directory handling
import numpy as np                # Numerical operations
import matplotlib.pyplot as plt   # Plotting
from joblib import PrintTime
from scipy.signal import butter, filtfilt  # Signal filtering
import wfdb                       # Reading .dat + .hea ECG files
from biosppy.signals import ecg   # QRS detection (R-peak detection)


# ================================
# SIGNAL PREPROCESSING FUNCTION
# ================================

def preprocess_signal(signal, sampling_rate):
    """
    Applies filtering and downsampling to ECG signal.

    Parameters:
        signal (np.array): ECG signal (shape: channels x samples)
        sampling_rate (int): Original sampling rate (e.g., 500 Hz)

    Returns:
        downsampled_signal (np.array): Processed signal
        new_sampling_rate (int): Updated sampling rate after downsampling
    """

    # ---- Step 1: Low-pass filter ----
    # We remove high-frequency noise (e.g., muscle artifacts)
    # Cutoff = 40 Hz (typical for ECG)
    # Normalize cutoff by Nyquist frequency (fs / 2)

    if signal is None or signal.size == 0:
        raise ValueError("Empty signal")

    if np.isnan(signal).any():
        raise ValueError("Signal contains NaNs")



    low = 0.5 / (sampling_rate / 2)
    high = 40 / (sampling_rate / 2)
    print("sampling rate: ", sampling_rate)
    print("high", high)
    print("low", low)

    # Safety clamp
    high = min(high, 0.99)

    if not (0 < low < high < 1):
        raise ValueError(f"Invalid normalized frequencies: low={low}, high={high}")


    # Design a 4th-order Butterworth low-pass filter
    b, a = butter(4, [0.5, 30], btype='band', fs=sampling_rate)

    # Apply filter forward and backward (zero phase distortion)
    print("Signal shape before filtering:", signal.shape)
    filtered_signal = filtfilt(b, a, signal, axis=-1)

    # ---- Step 2: Downsampling ----
    # Reduce data size and computational load
    # Keep every 10th sample
    downsample_factor = 2
    downsampled_signal = filtered_signal[:, ::downsample_factor]

    # Update sampling rate accordingly
    new_sampling_rate = sampling_rate / downsample_factor

    return downsampled_signal, new_sampling_rate


# ================================
# ECG PLOTTING FUNCTION
# ================================

def plot_ecg(signal, sampling_rate, title="ECG Signal"):
    """
    Plots a single ECG lead.

    Parameters:
        signal (np.array): 1D ECG signal
        sampling_rate (float): Sampling frequency in Hz
        title (str): Plot title
    """

    # Create time axis (seconds)
    time = np.arange(0, len(signal)) / sampling_rate

    # Plot signal
    plt.figure(figsize=(12, 6))
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


# ================================
# QRS DETECTION FUNCTION
# ================================

def detect_qrs_complex(signal, sampling_rate):
    """
    Detects R-peaks (QRS complexes) in ECG signal.

    Parameters:
        signal (np.array): 1D ECG signal
        sampling_rate (float): Sampling frequency

    Returns:
        qrs_indices (np.array): Indices of detected R-peaks
    """

    # biosppy ECG processing returns a dictionary
    # 'rpeaks' contains indices of detected QRS complexes
    output = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
    qrs_indices = output['rpeaks']

    return qrs_indices


# ================================
# RR INTERVAL CALCULATION
# ================================

def calculate_rr_intervals(qrs_indices, sampling_rate):
    """
    Computes RR intervals (time between heartbeats).

    Parameters:
        qrs_indices (np.array): R-peak indices
        sampling_rate (float): Sampling frequency

    Returns:
        rr_intervals (np.array): RR intervals in seconds
    """

    # Difference between consecutive R-peaks
    rr_intervals = np.diff(qrs_indices) / sampling_rate

    return rr_intervals


# ================================
# LOAD WFDB (.dat) ECG FILE
# ================================

def load_dat_ecg(file_path):
    """
    Loads ECG signal from WFDB .dat file.

    Important:
        Requires both .dat and .hea files.

    Parameters:
        file_path (str): Path to .dat file

    Returns:
        signal (np.array): ECG signal (channels x samples)
        sampling_rate (int): Sampling frequency
    """

    # Remove .dat extension (wfdb expects base filename)
    record_name = file_path.replace('.dat', '')

    # Read record (automatically loads .hea + .dat)
    record = wfdb.rdrecord(record_name)

    # p_signal → shape: (samples, channels)
    # Transpose → (channels, samples)
    signal = record.p_signal.T

    # Sampling frequency
    sampling_rate = record.fs

    return signal, sampling_rate


# ================================
# MAIN PROCESSING LOOP
# ================================

def process_all_ecg_files(root_dir):
    """
    Iterates through directory and processes all .dat ECG files.

    Parameters:
        root_dir (str): Root directory containing ECG records
    """

    # Walk through directory tree
    for root, dirs, files in os.walk(root_dir):
        for file in files:

            # Only process .dat files
            if file.endswith(".dat") and not file.endswith("r.dat"):

                file_path = os.path.join(root, file)

                try:
                    print(f"\nProcessing file: {file}")

                    # ---- Step 1: Load ECG ----
                    signal, fs = load_dat_ecg(file_path)
                    print(f"Sampling rate (fs): {fs}")

                    # ---- Step 2: Preprocess ----
                    processed_signal, new_fs = preprocess_signal(signal, fs)

                    # ---- Step 3: Select one lead ----
                    # Use first channel (lead I typically)
                    lead_signal = processed_signal[0]

                    # ---- Step 4: Plot ----
                    segment = lead_signal[:5000]  # first 50 seconds
                    plot_ecg(segment, new_fs, title=file)

                    # ---- Step 5: Detect QRS ----
                    qrs_indices = detect_qrs_complex(lead_signal, new_fs)

                    # ---- Step 6: Compute RR intervals ----
                    rr_intervals = calculate_rr_intervals(qrs_indices, new_fs)

                    # ---- Step 7: Print insights ----
                    print(f"Number of QRS complexes: {len(qrs_indices)}")
                    print(f"Average RR interval: {np.mean(rr_intervals):.3f} s")
                    print(f"Minimum RR interval: {np.min(rr_intervals):.3f} s")
                    print(f"Maximum RR interval: {np.max(rr_intervals):.3f} s")
                    print("--------------------------------------------------")

                except Exception as e:
                    print(f"Error processing file {file}: {e}")


# ================================
# ENTRY POINT
# ================================

# Root directory containing WFDB records
data_root_dir = "C:/Users/julia/Documents/GitHub/sleep_apnea_detection/apnea-ecg-database-1.0.0"

# Run processing
process_all_ecg_files(data_root_dir)