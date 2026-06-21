# peak_comparison_pipeline_v2.py
import h5py
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter

# --- 1. DATA LOADER (UNCHANGED) ---
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

# --- 2. PEAK FINDING & RANGE ALGORITHMS ---

# --- "Teacher" Algorithm (Ground Truth Generator) ---
def find_peaks_scipy(waveform: np.array, config: dict) -> list:
    """Uses SciPy find_peaks with robust settings to act as the ground truth."""
    indices, _ = find_peaks(
        waveform, # Expects pre-smoothed waveform
        height=config['threshold'],
        prominence=config.get('prominence', config['threshold'] / 2),
        distance=config['distance_samples']
    )
    return [{'index': idx} for idx in indices]

# --- "Student" Algorithms (Contenders) ---

# --- FIXED STATE MACHINE ---
def find_peaks_statemachine(waveform: np.array, config: dict) -> list:
    """
    FIXED: High-performance and logically correct state machine algorithm.
    """
    threshold = config['threshold']
    distance_samples = config['distance_samples']
    
    found_peaks = []
    state = 'SEARCHING'
    lockout_counter = 0
    
    # --- Performance Fix: Avoid slicing by tracking max manually ---
    potential_peak_start_index = 0
    current_max_val = -np.inf
    current_max_idx = -1

    # Function to process a found peak to avoid code duplication
    def process_peak():
        nonlocal found_peaks, state, lockout_counter
        if current_max_idx != -1:
            found_peaks.append({'index': current_max_idx})
        state = 'LOCKOUT'
        lockout_counter = distance_samples

    for i in range(len(waveform)):
        if state == 'LOCKOUT':
            lockout_counter -= 1
            if lockout_counter <= 0: state = 'SEARCHING'
            continue

        if state == 'SEARCHING':
            if waveform[i] > threshold:
                state = 'PEAK_DETECTED'
                # Reset trackers for the new peak
                current_max_val = waveform[i]
                current_max_idx = i
        
        elif state == 'PEAK_DETECTED':
            if waveform[i] > current_max_val:
                # Update the peak maximum
                current_max_val = waveform[i]
                current_max_idx = i
            
            if waveform[i] <= threshold:
                # Peak has ended, process it
                process_peak()

    # --- Logical Fix: Process any peak that was active at the end of the waveform ---
    if state == 'PEAK_DETECTED':
        process_peak()
                
    return found_peaks

# --- Wrappers for your Range-Finding Functions (Largely Unchanged) ---
def wrapper_find_range_simple_threshold(waveform: np.array, config: dict) -> list:
    threshold = config['threshold']
    above_threshold_indices = np.where(waveform > threshold)[0]
    if len(above_threshold_indices) < 2: return []
    
    start_idx, end_idx = above_threshold_indices[0], above_threshold_indices[-1]
    
    peak_region = waveform[start_idx:end_idx+1]
    if len(peak_region) == 0: return []
    peak_index = start_idx + np.argmax(peak_region)
    return [{'index': peak_index, 'start': start_idx, 'end': end_idx}]

def wrapper_find_range_peak_relative(waveform: np.array, config: dict) -> list:
    relative_height_fraction = config.get('relative_height_fraction', 0.1)
    peak_index = np.argmax(waveform)
    peak_amplitude = waveform[peak_index]
    if peak_amplitude <= config['threshold']: return []
    
    relative_threshold = relative_height_fraction * peak_amplitude
    start_idx = peak_index
    while start_idx > 0 and waveform[start_idx] > relative_threshold: start_idx -= 1
    end_idx = peak_index
    while end_idx < len(waveform) - 1 and waveform[end_idx] > relative_threshold: end_idx += 1
    
    return [{'index': peak_index, 'start': start_idx, 'end': end_idx}]

def wrapper_find_range_derivative(waveform: np.array, config: dict) -> list:
    baseline_samples = config.get('baseline_samples', 500)
    deriv_sigma_threshold = config.get('deriv_sigma_threshold', 3.0)
    
    derivative = np.diff(waveform)
    # Use a small portion of the start for derivative noise calculation
    deriv_noise_std = np.std(derivative[:baseline_samples])
    if deriv_noise_std == 0: return [] # Avoid division by zero if baseline is flat
    deriv_threshold = deriv_sigma_threshold * deriv_noise_std
    
    start_candidates = np.where(derivative > deriv_threshold)[0]
    if len(start_candidates) == 0: return []
    start_idx = start_candidates[0]
    
    # Find the absolute peak location *after* the trigger start
    peak_index_relative = np.argmax(waveform[start_idx:])
    peak_index_absolute = start_idx + peak_index_relative
    
    # Find end based on first negative derivative dip *after* the absolute peak
    end_candidates = np.where(derivative < -deriv_threshold)[0]
    valid_end_candidates = end_candidates[end_candidates > peak_index_absolute]
    if len(valid_end_candidates) == 0: return []
    end_idx = valid_end_candidates[0] + 1
    
    return [{'index': peak_index_absolute, 'start': start_idx, 'end': end_idx}]

# --- 3. ANALYSIS AND EVALUATION SUITE (WITH FIXES) ---
class PeakAnalysisSuite:
    def __init__(self, data_converter: DataConverter):
        self.dc = data_converter

    def _compare_peak_lists(self, true_peaks: list, found_peaks: list, tolerance_samples: int):
        tp = 0
        fp_found_indices = [p['index'] for p in found_peaks]
        
        for true_peak in true_peaks:
            true_idx = true_peak['index']
            match_found = False
            for i, found_idx in enumerate(fp_found_indices):
                if abs(true_idx - found_idx) <= tolerance_samples:
                    tp += 1
                    fp_found_indices.pop(i)
                    match_found = True
                    break
        fp = len(fp_found_indices)
        fn = len(true_peaks) - tp
        return tp, fp, fn

    def run_full_comparison(self, num_waveforms: int, methods_to_test: dict, config: dict):
        f, dset = self.dc.get_full_dataset_handle()
        num_waveforms_to_process = min(num_waveforms, dset.shape[0])

        results = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'time_us': 0.0, 'charges': []} for name in methods_to_test.keys()}

        for i in tqdm(range(num_waveforms_to_process), desc="Comparing Methods"):
            raw_waveform = dset[i, :]
            noise_region = raw_waveform[:config['baseline_samples']]
            baseline = np.mean(noise_region)
            std_dev = np.std(noise_region)
            waveform = -(raw_waveform - baseline)
            
            # --- HARMONIZATION FIX: Smooth the waveform ONCE for ALL methods ---
            # This ensures a fair, level playing field for comparison.
            smoothed_waveform = savgol_filter(waveform, window_length=11, polyorder=3)

            teacher_config = {
                'threshold': config['threshold_sigma'] * std_dev,
                'prominence': config['teacher_prominence_sigma'] * std_dev,
                'distance_samples': config['distance_samples']
            }
            true_peaks = find_peaks_scipy(smoothed_waveform, teacher_config)

            for name, method_info in methods_to_test.items():
                method_config = method_info['config'].copy()
                method_config['threshold'] = config['threshold_sigma'] * std_dev
                method_config['distance_samples'] = config['distance_samples']

                start_time = time.perf_counter()
                # Give every method the SAME smoothed waveform
                found_peaks = method_info['function'](smoothed_waveform, method_config)
                end_time = time.perf_counter()
                results[name]['time_us'] += (end_time - start_time) * 1e6

                tp, fp, fn = self._compare_peak_lists(true_peaks, found_peaks, config['match_tolerance_samples'])
                results[name]['tp'] += tp
                results[name]['fp'] += fp
                results[name]['fn'] += fn

                if len(true_peaks) == 1 and len(found_peaks) == 1 and tp == 1:
                    if 'start' in found_peaks[0] and 'end' in found_peaks[0]:
                        start, end = found_peaks[0]['start'], found_peaks[0]['end']
                        # Integrate on the original, non-inverted waveform for correct sign
                        charge = np.sum(waveform[start:end+1])
                        results[name]['charges'].append(charge)
        f.close()

        final_report = {}
        for name, data in results.items():
            tp, fp, fn = data['tp'], data['fp'], data['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            charges = np.array(data['charges'])
            mean_charge = np.mean(charges) if len(charges) > 0 else 0
            std_charge = np.std(charges) if len(charges) > 1 else 0
            charge_cv = std_charge / mean_charge if mean_charge != 0 else 0
            final_report[name] = {
                "precision": precision, "recall": recall, "f1_score": f1_score,
                "avg_time_us_per_waveform": data['time_us'] / num_waveforms_to_process,
                "mean_charge": mean_charge, "charge_cv": charge_cv,
                "total_true_positives": tp, "total_false_positives": fp, "total_false_negatives": fn
            }
        return final_report

# --- 4. SAVING AND PIPELINE EXECUTION (UNCHANGED) ---
def save_analysis_to_h5(h5_file, group_name: str, analysis_results: dict):
    print(f"  Saving results to group: '{group_name}'")
    main_group = h5_file.create_group(group_name)
    main_group.attrs['analysis_timestamp_utc'] = datetime.utcnow().isoformat()
    for method_name, data in analysis_results.items():
        clean_method_name = method_name.replace('. ', '_').replace(' ', '_')
        method_group = main_group.create_group(clean_method_name)
        for key, value in data.items(): method_group.attrs[key] = value

def run_analysis_pipeline(input_files: list, output_file_path: str, methods_to_test: dict, config: dict):
    print(f"Starting analysis pipeline. Output will be saved to '{output_file_path}'")
    with h5py.File(output_file_path, 'w') as f_out:
        f_out.attrs['creation_time_utc'] = datetime.utcnow().isoformat()
        f_out.attrs['analysis_type'] = 'Peak Finding Method Comparison'
        for file_path in tqdm(input_files, desc="Processing Files"):
            print(f"\nProcessing: {file_path}")
            if not os.path.exists(file_path):
                print(f"  - Warning: File not found. Skipping."); continue
            converter = DataConverter(file_path)
            analyzer = PeakAnalysisSuite(converter)
            results = analyzer.run_full_comparison(
                num_waveforms=config['num_waveforms_to_test'],
                methods_to_test=methods_to_test, config=config
            )
            group_name = os.path.basename(file_path).replace('.h5', '')
            save_analysis_to_h5(f_out, group_name, results)
            print(f"\n--- Results for {os.path.basename(file_path)} ---")
            for name, data in results.items():
                print(f"  Method: {name}")
                print(f"    F1-Score: {data['f1_score']:.4f} | Avg Time: {data['avg_time_us_per_waveform']:.2f} µs | Charge CV: {data['charge_cv']:.4f}")
            print("--------------------")
    print("\n\nPipeline finished successfully.")

# --- 5. MAIN EXECUTION BLOCK (UNCHANGED) ---
if __name__ == "__main__":
    import sys # Import sys to allow exiting the script

    # --- General Analysis Configuration (Unchanged) ---
    ANALYSIS_CONFIG = {
        'num_waveforms_to_test': 2000, 'baseline_samples': 3000,
        'threshold_sigma': 5.0, 'teacher_prominence_sigma': 3.0,
        'distance_ns': 500, 'match_tolerance_ns': 40,
    }

    # --- ROBUST PATH HANDLING ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- Define Input Files RELATIVE to the script's location ---
    # !!! CHECK THIS LIST CAREFULLY FOR TYPOS !!!
    INPUT_FILE_LIST_RELATIVE = [
        './pmt_data/voltage_test/CONVERTED_BAD_DATA_1400V.h5',
        './pmt_data/voltage_test/CONVERTED_BAD_DATA_1600V.h5',
        './pmt_data/voltage_test/CONVERTED_BAD_DATA_1970V.h5'
    ]
    OUTPUT_H5_FILE_RELATIVE = '../analysis/peak_finding_bakeoff_results_v2.h5'

    # --- Create absolute paths that will always work ---
    INPUT_FILE_LIST = [os.path.normpath(os.path.join(SCRIPT_DIR, p)) for p in INPUT_FILE_LIST_RELATIVE]
    OUTPUT_H5_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, OUTPUT_H5_FILE_RELATIVE))
    
    os.makedirs(os.path.dirname(OUTPUT_H5_FILE), exist_ok=True)
    
    # --- NEW: PRE-FLIGHT CHECK FOR ALL INPUT FILES ---
    print("--- Validating input file paths ---")
    all_files_found = True
    for file_path in INPUT_FILE_LIST:
        if not os.path.exists(file_path):
            print(f"❌ ERROR: The following input file does not exist:")
            print(f"   '{file_path}'")
            all_files_found = False
    
    if not all_files_found:
        print("\nOne or more input files were not found. Please check the paths in the script.")
        print("Pipeline execution halted.")
        sys.exit(1) # Stop the script
    else:
        print("✅ All input files found successfully.")
        
    # --- Define the Methods to Test (Unchanged) ---
    METHODS_TO_TEST = {
        '1. State Machine': {'function': find_peaks_statemachine, 'config': {}},
        '2. Simple Threshold': {'function': wrapper_find_range_simple_threshold, 'config': {}},
        '3. Peak Relative': {'function': wrapper_find_range_peak_relative, 'config': {'relative_height_fraction': 0.1}},
        '4. Derivative': {'function': wrapper_find_range_derivative, 'config': {'deriv_sigma_threshold': 3.0, 'baseline_samples': 500}}
    }

    # --- Create a dummy file for testing (Unchanged, but less critical now) ---
    # ... (This section can remain as is) ...

    # --- Convert time-based config to sample-based (Unchanged) ---
    try:
        existing_file = INPUT_FILE_LIST[0]
        temp_conv = DataConverter(existing_file)
        timestep = temp_conv.get_timestep()
        ANALYSIS_CONFIG['distance_samples'] = int(ANALYSIS_CONFIG['distance_ns'] * 1e-9 / timestep)
        ANALYSIS_CONFIG['match_tolerance_samples'] = int(ANALYSIS_CONFIG['match_tolerance_ns'] * 1e-9 / timestep)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: Could not read metadata from '{INPUT_FILE_LIST[0]}' to configure the pipeline. {e}"); exit()

    # --- Run the Pipeline ---
    run_analysis_pipeline(INPUT_FILE_LIST, OUTPUT_H5_FILE, METHODS_TO_TEST, ANALYSIS_CONFIG)