import datetime
import os                          # For file and directory handling
import numpy as np                # Numerical operations
import matplotlib.pyplot as plt   # Plotting
from scipy.signal import butter, filtfilt  # Signal filtering
import wfdb                       # Reading .dat + .hea ECG files
from biosppy.signals import ecg   # QRS detection (R-peak detection)

def preprocess_signal(signal, sampling_rate):
    """
    Applies filtering and downsampling to ECG signal.

    Parameters:
        signal (np.array): ECG signal (shape: channels x samples)
        sampling_rate (int): Original sampling rate (100Hz)
    Returns:
        downsampled_signal (np.array): Processed signal
        new_sampling_rate (int): Updated sampling rate after downsampling
    """

    # removing noise w a filter
    #dont include nulls
    if signal is None or signal.size == 0:
        raise ValueError("Empty signal")
    if np.isnan(signal).any():
        raise ValueError("Signal contains NaNs")

    # Normalize cutoff by Nyquist frequency (fs / 2)
    nyquist = sampling_rate / 2

    low_cutoff = max(0.5, sampling_rate / 100)  # At least 0.5 Hz or higher
    high_cutoff = min(30, nyquist * 0.95)  # Max 30 Hz or just below Nyquist

    print(f"Using cutoffs: {low_cutoff} - {high_cutoff} Hz")

    b, a = butter(4, [low_cutoff, high_cutoff], btype='band', fs=sampling_rate)
    filtered_signal = filtfilt(b, a, signal, axis=-1)
    print("Filter successful ", b, a)
    print("Filtered signal: ", filtered_signal)
    #works
    return filtered_signal

def downsampling(filtered_signal, sampling_rate):
    downsample_factor = 2
    downsampled_signal = filtered_signal[:, ::downsample_factor]
    print("Downsampling success: ", downsampled_signal)
    # works
    new_sampling_rate = sampling_rate / downsample_factor
    # problem
    print("New sampling rate: ", new_sampling_rate)

    return downsampled_signal, new_sampling_rate

#plotting
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

#R peak detection

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
    print("im in qrs")

    #PROBLEM OCCURS HERE
    output = ecg.ecg(
        signal=signal,
        sampling_rate=sampling_rate,
        show=False
    )
    print("output of ecg.ecg: ", output)
    qrs_indices = output['rpeaks']
    print("qrs indices: ", qrs_indices)
    return qrs_indices


#rr intervals

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

#load data
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

    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Signal shape: {signal.shape}")
    print(f"Data type: {signal.dtype}")
    print(f"Contains NaN: {np.isnan(signal).any()}")
    print(f"Contains Inf: {np.isinf(signal).any()}")
    print(f"Min value: {np.min(signal):.6f}")
    print(f"Max value: {np.max(signal):.6f}")
    print(f"Mean value: {np.mean(signal):.6f}")
    print(f"Std deviation: {np.std(signal):.6f}")

    # Check for unusual values
    if np.max(np.abs(signal)) > 1e6:
        print("WARNING: Extremely large values detected!")

    if np.std(signal) == 0:
        print("WARNING: Zero standard deviation - signal is constant!")

    # Check first few samples
    print(f"First 10 samples: {signal[0, :10]}")

    return signal, sampling_rate

# MAIN PROCESSING LOOP

def process_all_ecg_files(root_dir):
    """
    Iterates through directory and processes all .dat ECG files.

    Parameters:
        root_dir (str): Root directory containing ECG records
    """
    start_time = datetime.datetime.now()

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
                    processed_signal = preprocess_signal(signal, fs)


                    print("Processed signal and new fs done: ", processed_signal)
                    #select one channel from filtered signal
                    lead_signal = processed_signal[0]
                    print("lead signal: ", lead_signal)

                    #PROBLEM OCCURED SOMEWHERE HERE while downsampling ran right after filtering
                    # decided to downsample AFTER RR peak detection
                    # Detect QRS
                    qrs_indices = detect_qrs_complex(lead_signal, fs)
                    print("qrs done")
                    # Compute RR intervals
                    rr_intervals = calculate_rr_intervals(qrs_indices, fs)
                    print("rr_intervals done")

                    #downsample after qrs and select the same channel as lead
                    downsampled_signal, new_fs = downsampling(processed_signal, fs)
                    lead_signal_downsampled = downsampled_signal[0]
                    print("lead signal: ", lead_signal_downsampled)

                    # Plot
                    segment = lead_signal_downsampled[:5000]  # first 50 seconds
                    plot_ecg(segment, new_fs, title=file)

                    print(f"Number of QRS complexes: {len(qrs_indices)}")
                    print(f"Average RR interval: {np.mean(rr_intervals):.3f} s")
                    print(f"Minimum RR interval: {np.min(rr_intervals):.3f} s")
                    print(f"Maximum RR interval: {np.max(rr_intervals):.3f} s")
                    print("--------------------------------------------------")


                except Exception as e:
                    print(f"Error processing file {file}: {e}")

        stopwatch = start_time - datetime.datetime.now()
        print("TOTAL EXECUTION TIME: ",stopwatch)


# ================================
# ENTRY POINT
# ================================

# Root directory containing WFDB records
data_root_dir = "C:/Users/ZelenePC/Desktop/sleep_apnea_detection/apnea-ecg-database-1.0.0"

# Run processing
process_all_ecg_files(data_root_dir)