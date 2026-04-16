import datetime
import os                          # For file and directory handling
import time
import numpy as np                # Numerical operations
from numpy.ma.core import sqrt
from scipy.signal import butter, filtfilt  # Signal filtering
import wfdb                       # Reading .dat + .hea ECG files
from biosppy.signals import ecg   # QRS detection (R-peak detection)
import pandas
from multiprocessing import Pool, cpu_count

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
    #works
    return filtered_signal

def downsampling(filtered_signal, sampling_rate):
    downsample_factor = 2
    downsampled_signal = filtered_signal[:, ::downsample_factor]
    # works
    new_sampling_rate = sampling_rate / downsample_factor

    return downsampled_signal, new_sampling_rate

def resample_to_seconds(timestamps, values, duration):
    time_grid = np.arange(0, int(duration))
    resampled = np.interp(time_grid, timestamps, values)
    return time_grid, resampled

def detect_qrs_complex(signal, sampling_rate):
    # biosppy ECG processing returns a dictionary
    # 'rpeaks' contains indices of detected QRS complexes
    output = ecg.ecg(
        signal=signal,
        sampling_rate=sampling_rate,
        show = False
    )

    return output['rpeaks'], output['heart_rate'], output['heart_rate_ts']

#rr intervals

def calculate_rr_intervals(qrs_indices, sampling_rate):
    """
        rr_intervals (np.array): RR intervals in seconds
    """
    # Difference between consecutive R-peaks
    rr_intervals = np.diff(qrs_indices) / sampling_rate

    return rr_intervals

def get_features(rr_intervals):
    rr_mean = np.mean(rr_intervals)
    rr_sd = np.std(rr_intervals)
    differences = []
    for i in range(len(rr_intervals)-1):
        current = rr_intervals[i]
        next = rr_intervals[i+1]
        differences.append((next - current)**2)
    summed = sum(differences)
    rmssd = sqrt((1 / (len(rr_intervals) - 1)) * summed)
    nn50 = 0
    for rr in rr_intervals:
        if rr > 0.50:
            nn50+=1
    pnn50 = nn50/len(rr_intervals)

    return rr_mean, rr_sd, rmssd, nn50, pnn50

#load data
def load_dat_ecg(file_path):
    # Remove .dat extension (wfdb expects base filename)
    record_name = file_path.replace('.dat', '')

    # Read record (automatically loads .hea + .dat)
    record = wfdb.rdrecord(record_name)

    # p_signal → shape: (samples, channels)
    # Transpose → (channels, samples)
    signal = record.p_signal.T

    # Sampling frequency
    sampling_rate = record.fs

    # Check for unusual values
    if np.max(np.abs(signal)) > 1e6:
        print("WARNING: Extremely large values detected!")

    if np.std(signal) == 0:
        print("WARNING: Zero standard deviation - signal is constant!")

    # Check first few samples
    return signal, sampling_rate

def process_segment(signal, fs):

    qrs_indices, heart_rate, heart_rate_ts = detect_qrs_complex(signal, fs)

    rr_intervals = calculate_rr_intervals(qrs_indices, fs)

    rr_mean, rr_sd, rmssd, nn50, pnn50 = get_features(rr_intervals)

    return rr_mean, rr_sd, rmssd, nn50, pnn50

# MAIN PROCESSING LOOP
def process_file(file_path):
    try:
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        signal, fs = load_dat_ecg(file_path)
        processed_signal = preprocess_signal(signal, fs)

        # select one channel from filtered signal
        lead_signal = processed_signal[0]

        #get a sample every minute
        samples_per_min = int(fs * 59.99)  # one minute
        length_of_signal = len(lead_signal)
        num_segments = length_of_signal // samples_per_min

        records = []

        for i in range(num_segments):
            start = i * samples_per_min
            end = start + samples_per_min
            segment = lead_signal[start:end]

            rr_mean, rr_sd, rmssd, nn50, pnn50 = process_segment(segment,fs)
            records.append({ "title": f"{os.path.basename(file_path)}_{i}",
                            "rr_mean": rr_mean,
                            "rr_sd":rr_sd,
                            "rmssd": rmssd,
                            "nn50": nn50,
                            "pnn50": pnn50})

        return records


    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")

def process_all_ecg_files(root_dir):
    start_time = time.time()
    # Walk through directory tree
    file_paths = []
    i =0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".dat") and not file.endswith("r.dat") and i < 10:
                i += 1
                file_paths.append(os.path.join(root, file))

    with Pool(cpu_count()) as p:
        recordings = p.map(process_file, file_paths)

    flat_records = [item for sublist in recordings if sublist for item in sublist]
    df = pandas.DataFrame(flat_records)
    df.to_csv("ecg_data.csv", index=False)
    stopwatch = time.time()- start_time
    print("TOTAL EXECUTION TIME: ",stopwatch)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()  # optional but recommended on Windows

    # Root directory containing WFDB records
    data_root_dir = "C:/Users/ZelenePC/Desktop/sleep_apnea_detection/apnea-ecg-database-1.0.0"

    # Run processing
    process_all_ecg_files(data_root_dir)

