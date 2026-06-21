import h5py
import numpy as np
import os
import argparse
import sys
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# --- 1. ALGORITHMS & CONFIGURATION (EMBEDDED) ---
# All the necessary functions and configurations are now part of this script.

def find_peaks_scipy(waveform: np.array, config: dict) -> list:
    """'Teacher' algorithm using SciPy."""
    indices, _ = find_peaks(
        waveform, height=config['threshold'],
        prominence=config.get('prominence', config['threshold'] / 2),
        distance=config['distance_samples']
    )
    return [{'index': idx} for idx in indices]

def find_peaks_statemachine(waveform: np.array, config: dict) -> list:
    """High-performance state machine algorithm."""
    threshold = config['threshold']
    distance_samples = config['distance_samples']
    found_peaks, state, lockout_counter = [], 'SEARCHING', 0
    current_max_val, current_max_idx = -np.inf, -1
    def process_peak():
        nonlocal found_peaks, state, lockout_counter
        if current_max_idx != -1: found_peaks.append({'index': current_max_idx})
        state = 'LOCKOUT'; lockout_counter = distance_samples
    for i in range(len(waveform)):
        if state == 'LOCKOUT':
            lockout_counter -= 1
            if lockout_counter <= 0: state = 'SEARCHING'
            continue
        if state == 'SEARCHING':
            if waveform[i] > threshold:
                state = 'PEAK_DETECTED'; current_max_val = waveform[i]; current_max_idx = i
        elif state == 'PEAK_DETECTED':
            if waveform[i] > current_max_val:
                current_max_val = waveform[i]; current_max_idx = i
            if waveform[i] <= threshold: process_peak()
    if state == 'PEAK_DETECTED': process_peak()
    return found_peaks

def wrapper_find_range_simple_threshold(waveform: np.array, config: dict) -> list:
    """Wrapper for the simple threshold range finder."""
    threshold = config['threshold']
    above_threshold_indices = np.where(waveform > threshold)[0]
    if len(above_threshold_indices) < 2: return []
    start_idx, end_idx = above_threshold_indices[0], above_threshold_indices[-1]
    peak_region = waveform[start_idx:end_idx+1]
    if len(peak_region) == 0: return []
    peak_index = start_idx + np.argmax(peak_region)
    return [{'index': peak_index}]

def wrapper_find_range_peak_relative(waveform: np.array, config: dict) -> list:
    """Wrapper for the peak-relative range finder."""
    relative_height_fraction = config.get('relative_height_fraction', 0.1)
    peak_index = np.argmax(waveform)
    peak_amplitude = waveform[peak_index]
    if peak_amplitude <= config['threshold']: return []
    relative_threshold = relative_height_fraction * peak_amplitude
    start_idx = peak_index
    while start_idx > 0 and waveform[start_idx] > relative_threshold: start_idx -= 1
    end_idx = peak_index
    while end_idx < len(waveform) - 1 and waveform[end_idx] > relative_threshold: end_idx += 1
    return [{'index': peak_index}]

def wrapper_find_range_derivative(waveform: np.array, config: dict) -> list:
    """Wrapper for the derivative-based range finder."""
    baseline_samples = config.get('baseline_samples', 500)
    deriv_sigma_threshold = config.get('deriv_sigma_threshold', 3.0)
    derivative = np.diff(waveform)
    if len(derivative) < baseline_samples: return []
    deriv_noise_std = np.std(derivative[:baseline_samples])
    if deriv_noise_std == 0: return []
    deriv_threshold = deriv_sigma_threshold * deriv_noise_std
    start_candidates = np.where(derivative > deriv_threshold)[0]
    if len(start_candidates) == 0: return []
    start_idx = start_candidates[0]
    peak_index_relative = np.argmax(waveform[start_idx:])
    peak_index_absolute = start_idx + peak_index_relative
    end_candidates = np.where(derivative < -deriv_threshold)[0]
    valid_end_candidates = end_candidates[end_candidates > peak_index_absolute]
    if len(valid_end_candidates) == 0: return []
    end_idx = valid_end_candidates[0] + 1
    return [{'index': peak_index_absolute}]

# --- Central Configuration Dictionary ---
METHODS_TO_TEST = {
    '1. State Machine': {'function': find_peaks_statemachine, 'config': {}},
    '2. Simple Threshold': {'function': wrapper_find_range_simple_threshold, 'config': {}},
    '3. Peak Relative': {'function': wrapper_find_range_peak_relative, 'config': {'relative_height_fraction': 0.1}},
    '4. Derivative': {'function': wrapper_find_range_derivative, 'config': {'deriv_sigma_threshold': 3.0, 'baseline_samples': 500}}
}


# --- 2. DATA LOADER ---
class DataConverter():
    """Handles reading waveform data from HDF5 files."""
    def __init__(self, path: str):
        self.path = path
    def get_full_dataset_handle(self):
        f = h5py.File(self.path, 'r')
        return f, f['waveforms']
    def get_timestep(self) -> float:
        with h5py.File(self.path, 'r') as f:
            return float(f.attrs['x_increment_seconds'])

# --- 3. FALSE POSITIVE FINDER AND PLOTTER ---
class FalsePositiveFinder:
    """Finds and plots waveforms where a method incorrectly found a peak."""
    def __init__(self, data_converter: DataConverter, config: dict):
        self.dc = data_converter
        self.config = config

    def _find_false_positives(self, true_peaks: list, found_peaks: list) -> list:
        """Compares peak lists and returns the found peaks that were incorrect."""
        false_positives = []
        tolerance = self.config['match_tolerance_samples']
        true_indices = [p['index'] for p in true_peaks]
        for found_peak in found_peaks:
            is_match = False
            for true_idx in true_indices:
                if abs(found_peak['index'] - true_idx) <= tolerance:
                    is_match = True
                    break
            if not is_match:
                false_positives.append(found_peak)
        return false_positives

    def find_and_plot(self, method_to_test: dict, max_to_find: int):
        """
        Iterates through waveforms, finds false positives for the specified method,
        and generates a plot for each one.
        """
        f, dset = self.dc.get_full_dataset_handle()
        num_waveforms = dset.shape[0]
        found_count = 0

        print(f"\nSearching for up to {max_to_find} false positives for method: '{method_to_test['name']}'...")

        for i in range(num_waveforms):
            if found_count >= max_to_find: break
            raw_waveform = dset[i, :]
            noise_region = raw_waveform[:self.config['baseline_samples']]
            baseline = np.mean(noise_region)
            std_dev = np.std(noise_region)
            waveform = -(raw_waveform - baseline)
            smoothed_waveform = savgol_filter(waveform, window_length=11, polyorder=3)
            
            teacher_config = {
                'threshold': self.config['threshold_sigma'] * std_dev,
                'prominence': self.config['teacher_prominence_sigma'] * std_dev,
                'distance_samples': self.config['distance_samples']
            }
            true_peaks = find_peaks_scipy(smoothed_waveform, teacher_config)

            student_fn = method_to_test['function']
            student_config = method_to_test['config'].copy()
            student_config['threshold'] = self.config['threshold_sigma'] * std_dev
            student_config['distance_samples'] = self.config['distance_samples']
            found_peaks_by_student = student_fn(smoothed_waveform, student_config)

            if not found_peaks_by_student: continue

            false_positives = self._find_false_positives(true_peaks, found_peaks_by_student)

            if false_positives:
                for fp_peak in false_positives:
                    if found_count >= max_to_find: break
                    found_count += 1
                    print(f"  ({found_count}/{max_to_find}) Found false positive in waveform index {i}. Plotting...")
                    self.plot_failure(
                        raw_waveform=waveform, smoothed_waveform=smoothed_waveform,
                        true_peaks=true_peaks, fp_peak=fp_peak, threshold=teacher_config['threshold'],
                        waveform_info={'index': i, 'file': os.path.basename(self.dc.path), 'method_name': method_to_test['name']}
                    )
        f.close()
        if found_count == 0:
            print("\nSearch complete. No false positives found for this method.")
        else:
            print(f"\nSearch complete. Displayed {found_count} plots.")

    def plot_failure(self, raw_waveform, smoothed_waveform, true_peaks, fp_peak, threshold, waveform_info):
        """Generates a detailed plot of a single false positive event."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(raw_waveform, label='Waveform (Baseline Corrected)', color='lightblue', zorder=1)
        ax.plot(smoothed_waveform, label='Smoothed Waveform (Analyzed)', color='navy', zorder=2)
        fp_idx = fp_peak['index']
        fp_amp = smoothed_waveform[fp_idx]
        ax.axvline(x=fp_idx, color='red', linestyle='--', label=f'FALSE POSITIVE Peak', zorder=4)
        ax.plot(fp_idx, fp_amp, 'rx', markersize=12, zorder=5)
        for true_peak in true_peaks:
            true_idx, true_amp = true_peak['index'], smoothed_waveform[true_peak['index']]
            ax.plot(true_idx, true_amp, 'go', markersize=10, label='Correct Peak (found by Teacher)', zorder=5)
        ax.axhline(y=threshold, color='green', linestyle=':', label=f"Detection Threshold ({self.config['threshold_sigma']}σ)", zorder=3)
        title = (f"False Positive for Method: '{waveform_info['method_name']}'\n"
                 f"Waveform Index: {waveform_info['index']} in {waveform_info['file']}")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Amplitude (Inverted & Baseline Corrected)', fontsize=12)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        zoom_window = 250
        ax.set_xlim(max(0, fp_idx - zoom_window), min(len(raw_waveform), fp_idx + zoom_window))
        plt.tight_layout()
        plt.show()

# --- 4. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and visualize False Positive peak detections for a specific algorithm."
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Path to the ORIGINAL HDF5 data file you want to analyze."
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: The file '{args.input_file}' was not found.")
        sys.exit(1)

    ANALYSIS_CONFIG = {
        'baseline_samples': 3000, 'threshold_sigma': 5.0,
        'teacher_prominence_sigma': 3.0, 'distance_ns': 500, 'match_tolerance_ns': 40,
    }
    
    METHODS = {str(i+1): {'name': name, **info} for i, (name, info) in enumerate(METHODS_TO_TEST.items())}

    print("\n--- False Positive Analysis ---")
    print("Which method would you like to investigate?")
    for key, val in METHODS.items():
        print(f"  {key}: {val['name']}")
    
    choice = input(f"Enter your choice (1-{len(METHODS)}): ")
    if choice not in METHODS:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    selected_method = METHODS[choice]

    try:
        max_plots = int(input("How many examples would you like to find and plot? (e.g., 5): "))
    except ValueError:
        print("Invalid number. Exiting.")
        sys.exit(1)

    try:
        converter = DataConverter(args.input_file)
        timestep = converter.get_timestep()
        ANALYSIS_CONFIG['distance_samples'] = int(ANALYSIS_CONFIG['distance_ns'] * 1e-9 / timestep)
        ANALYSIS_CONFIG['match_tolerance_samples'] = int(ANALYSIS_CONFIG['match_tolerance_ns'] * 1e-9 / timestep)
    except Exception as e:
        print(f"Error setting up analysis configuration: {e}"); sys.exit(1)

    finder = FalsePositiveFinder(converter, ANALYSIS_CONFIG)
    finder.find_and_plot(selected_method, max_plots)