import datetime
import os                          # For file and directory handling
import time
import numpy as np                # Numerical operations
from numpy.ma.core import sqrt
from scipy.signal import butter, filtfilt, welch  # Signal filtering, welch for frequency analysis
from scipy.interpolate import interp1d #for sampling of RR intervals
from scipy.integrate import trapezoid #for vlf,lf, hf
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

    return output['rpeaks']
#rr intervals

def calculate_rr_intervals(qrs_indices, sampling_rate):
    # Difference between consecutive R-peaks
    rr_intervals = np.diff(qrs_indices) / sampling_rate

    return rr_intervals

def get_frequency_features(rr_intervals):

    # Too few RR values
    if len(rr_intervals) < 4:
        return None, None, None, None, None, None

    # 1. Creating the timeline for RR intervals

    rr_times = np.cumsum(rr_intervals)

    #  Starts at 0 seconds
    rr_times = rr_times - rr_times[0]

    # 2. Prepare for interpolation

    interpolation_function = interp1d(
        rr_times,
        rr_intervals,
        kind='cubic',
        fill_value="extrapolate"
    )

    # 3. Generate a uniform time axis
    #    (4 Hz  as in the article)


    fs_resample = 4

    resampled_times = np.arange(0,rr_times[-1], 1 / fs_resample )

    # 4. Interpolate the RR signal

    interpolated_rr = interpolation_function(resampled_times)

    # 5. Welch Power Spectral Density

    freqs, psd = welch(interpolated_rr,fs=fs_resample, nperseg=min(256, len(interpolated_rr)))

    # 6. Define frequency ranges

    vlf_band = (freqs >= 0.00) & (freqs < 0.04)
    lf_band  = (freqs >= 0.04) & (freqs < 0.15)
    hf_band  = (freqs >= 0.15) & (freqs < 0.40)


    # 7. Calculate Power

    vlf = trapezoid(psd[vlf_band], freqs[vlf_band])
    lf  = trapezoid(psd[lf_band], freqs[lf_band])
    hf  = trapezoid(psd[hf_band], freqs[hf_band])

    # 8. LF/HF Ratio

    lf_hf = lf / hf if hf > 0 else 0

    # 9. normalised values

    total_power = lf + hf

    lf_norm = (lf / total_power) * 100 if total_power > 0 else 0
    hf_norm = (hf / total_power) * 100 if total_power > 0 else 0

    return vlf, lf, hf, lf_hf, lf_norm, hf_norm

def get_features(rr_intervals):
    if len(rr_intervals) > 2:
        rr_mean = np.mean(rr_intervals)
        rr_sd = np.std(rr_intervals)
        differences = []
        for i in range(len(rr_intervals)-1):
            current = rr_intervals[i]
            next = rr_intervals[i+1]
            differences.append((next - current)**2)
        summed = sum(differences)
        rmssd = sqrt((1 / (len(rr_intervals) - 1)) * summed)
        diffs = np.abs(np.diff(rr_intervals))
        nn50 = np.sum(diffs > 0.05)
        pnn50 = nn50 / len(diffs) if len(diffs) > 0 else 0
        return rr_mean, rr_sd, rmssd, nn50, pnn50, 60/rr_mean

    else:
        return None, None, None, None, None, None

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

    # Check first few samples
    return signal, sampling_rate

def process_segment(signal, fs):
    try:
        qrs_indices = detect_qrs_complex(signal, fs)

        if len(qrs_indices) < 2:
            return None, None, None, None, None, None

        rr_intervals = calculate_rr_intervals(qrs_indices, fs)

        time_features = get_features(rr_intervals)

        freq_features = get_frequency_features(rr_intervals)

        return (*time_features, *freq_features)

    except:
        return None, None, None, None, None, None

# MAIN PROCESSING LOOP
def process_file(file_path):
    try:
        print(f"\nProcessing file: {os.path.basename(file_path)[:3]}")
        annotation = wfdb.rdann(file_path.replace(".dat", ""), 'apn')
        labels = annotation.symbol
        labels_numeric = [1 if l == 'A' else 0 for l in labels]

        signal, fs = load_dat_ecg(file_path)
        processed_signal = preprocess_signal(signal, fs)

        # select one channel from filtered signal
        lead_signal = processed_signal[0]

        #get a sample every minute
        samples_per_min = int(fs * 60)  # one minute
        length_of_signal = len(lead_signal)
        num_segments = length_of_signal // samples_per_min

        records = []

        for i in range(num_segments):
            start = i * samples_per_min
            end = start + samples_per_min
            segment = lead_signal[start:end]

            rr_mean, rr_sd, rmssd, nn50, pnn50, bpm, \
                vlf, lf, hf, lf_hf, lf_norm, hf_norm = process_segment(segment, fs)
            records.append({ "title": f"{os.path.basename(file_path)}_{i}",
                            "rr_mean": rr_mean,
                            "rr_sd":rr_sd,
                            "rmssd": rmssd,
                            "nn50": nn50,
                            "pnn50": pnn50,
                             "bpm": bpm,
                             "vlf": vlf,
                             "lf": lf,
                             "hf": hf,
                             "lf_hf": lf_hf,
                             "lf_norm": lf_norm,
                             "hf_norm": hf_norm,
                             "label": None,
                             "patient":os.path.basename(file_path)[:3]})
        min_len = min(len(records), len(labels_numeric))
        for i in range(min_len):
            records[i]["label"] = labels_numeric[i]

        return records


    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return []

def process_all_ecg_files(root_dir):
    start_time = time.time()
    # Walk through directory tree
    file_paths = []
    i =0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".dat") and not file.endswith("r.dat") and i < 35:
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
    #data_root_dir = "C:/Users/ZelenePC/Desktop/sleep_apnea_detection/apnea-ecg-database-1.0.0"
    data_root_dir = "C:/Users/julia/Documents/GitHub/sleep_apnea_detection/apnea-ecg-database-1.0.0/TESTING"

    # Run processing
    process_all_ecg_files(data_root_dir)
