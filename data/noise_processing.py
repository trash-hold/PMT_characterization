import h5py
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# Assume the DataConverter class is in the same file or imported
class DataConverter():
    def __init__(self, path: str):
        self.path = path
    
    def get_single_data(self, index) -> np.array:
        with h5py.File(self.path, 'r') as f:
            return np.array(f['waveforms'][index])
        
    def get_timestep(self) -> float:
        with h5py.File(self.path, 'r') as f:
            return float(f.attrs['x_increment_seconds'])
    
    def get_full_dataset_handle(self):
        """Returns the h5py file and dataset objects for efficient reading."""
        f = h5py.File(self.path, 'r')
        return f, f['waveforms']


class DataAnalyzer():
    def __init__(self, dataConverter: DataConverter):
        self.dc = dataConverter

    # --- HEADLESS ANALYSIS ROUTINES ---

    def calculate_file_baseline(self, num_waveforms: int, baseline_samples: int) -> float:
        """Calculates a single, robust baseline value for an entire file."""
        baseline_medians = []
        f, dset = self.dc.get_full_dataset_handle()
        
        # Limit num_waveforms if the file is smaller
        if num_waveforms > dset.shape[0]:
            num_waveforms = dset.shape[0]
            
        for i in range(num_waveforms):
            waveform = dset[i]
            baseline_medians.append(np.median(waveform[:baseline_samples]))
        
        f.close()
        return np.mean(baseline_medians)

    def calculate_average_fft(self, baseline_correction_val: float) -> (np.array, np.array):
        """Calculates the average FFT spectrum for all waveforms in a file."""
        f, dset = self.dc.get_full_dataset_handle()
        num_waveforms, N = dset.shape
        T = self.dc.get_timestep()
        
        freq_axis = np.fft.fftfreq(N, T)[:N//2]
        sum_fft_mag = np.zeros_like(freq_axis)

        for i in range(num_waveforms):
            waveform = dset[i] - baseline_correction_val # Apply correction
            fft_vals = np.fft.fft(waveform)
            fft_mag = 2.0/N * np.abs(fft_vals[0:N//2])
            sum_fft_mag += fft_mag
            
        f.close()
        return freq_axis, sum_fft_mag / num_waveforms

    def calculate_noise_histogram(self, baseline_correction_val: float, num_waveforms: int, baseline_samples: int, num_bins: int) -> (np.array, np.array):
        """Calculates the noise distribution histogram data for a file."""
        f, dset = self.dc.get_full_dataset_handle()
        
        # Limit num_waveforms if the file is smaller
        if num_waveforms > dset.shape[0]:
            num_waveforms = dset.shape[0]
            
        noise_pool = []
        for i in range(num_waveforms):
            waveform = dset[i]
            # Use the pre-calculated baseline for consistency
            corrected_baseline = waveform[:baseline_samples] - baseline_correction_val
            noise_pool.append(corrected_baseline)
        
        f.close()
        all_noise_points = np.concatenate(noise_pool)
        counts, bin_edges = np.histogram(all_noise_points, bins=num_bins)
        return counts, bin_edges

    def calculate_convergence_metrics(self, metrics: list, max_samples: int, increment: int, baseline_end_fraction: float) -> dict:
        """Calculates convergence stats for a list of functions in a single pass. (Unchanged from before)"""
        f, dset = self.dc.get_full_dataset_handle()
        num_waveforms = dset.shape[0]
        total_points = dset.shape[1]
        baseline_end_index = int(total_points * baseline_end_fraction)
        if max_samples > baseline_end_index:
            max_samples = baseline_end_index

        split_sizes = list(range(increment, max_samples + increment, increment))
        all_results = [{size: [] for size in split_sizes} for _ in metrics]

        for i in range(num_waveforms):
            baseline_data = dset[i, :baseline_end_index]
            for size in split_sizes:
                data_slice = baseline_data[:size]
                for func_index, func in enumerate(metrics):
                    result_value = func(data_slice)
                    all_results[func_index][size].append(result_value)
        
        f.close()
        final_output = {}
        for func_index, func in enumerate(metrics):
            raw_results = all_results[func_index]
            func_name = func.__name__
            final_output[func_name] = {
                'splits': split_sizes,
                'means': [np.mean(raw_results[size]) for size in split_sizes],
                'stds': [np.std(raw_results[size]) for size in split_sizes]
            }
        return final_output

# --- HELPER FUNCTION FOR SAVING ---

def save_results_to_h5(h5_file, group_name: str, original_data_path: str, analysis_results: dict):
    """Saves original data and analysis results to a new HDF5 file."""
    print(f"  Saving results to group: '{group_name}'")
    
    # Create the main group for this file
    main_group = h5_file.create_group(group_name)
    
    # Copy the original waveform data
    with h5py.File(original_data_path, 'r') as f_in:
        main_group.copy(f_in['waveforms'], 'waveforms')
        # Copy original attributes as well
        for key, val in f_in.attrs.items():
            main_group.attrs[key] = val

    # Create a subgroup for all the analysis results
    analysis_group = main_group.create_group('analysis_results')
    
    # 1. Save baseline correction value
    analysis_group.create_dataset('baseline_correction_value', data=analysis_results['baseline'])
    
    # 2. Save average FFT
    fft_group = analysis_group.create_group('average_fft')
    fft_group.create_dataset('frequency_hz', data=analysis_results['avg_fft'][0])
    fft_group.create_dataset('magnitude', data=analysis_results['avg_fft'][1])
    
    # 3. Save noise distribution histogram
    hist_group = analysis_group.create_group('noise_distribution')
    hist_group.create_dataset('counts', data=analysis_results['noise_hist'][0])
    hist_group.create_dataset('bin_edges', data=analysis_results['noise_hist'][1])

    # 4. Save convergence metrics (nested structure)
    conv_group = analysis_group.create_group('convergence')
    for metric_name, data in analysis_results['convergence'].items():
        metric_group = conv_group.create_group(metric_name)
        metric_group.create_dataset('splits', data=data['splits'])
        metric_group.create_dataset('means', data=data['means'])
        metric_group.create_dataset('stds', data=data['stds'])

# --- MAIN PIPELINE FUNCTION ---

def process_files_pipeline(input_files: list, output_file_path: str):
    """
    Runs a full analysis pipeline on a list of HDF5 files and saves the
    results to a new, consolidated HDF5 file.
    """
    print(f"Starting processing pipeline. Output will be saved to '{output_file_path}'")
    
    # Open the output file once in write mode to create it
    with h5py.File(output_file_path, 'w') as f_out:
        
        for file_path in tqdm(input_files, desc="Processing Files"):
            print(f"\nProcessing: {file_path}")
            
            # --- Setup ---
            conv = DataConverter(file_path)
            anal = DataAnalyzer(conv)
            
            # --- Analysis Parameters ---
            # These can be configured as needed
            OPTIMAL_BASELINE_SAMPLES = 500
            
            # --- Run All Analysis Stages ---
            # 1. Calculate baseline correction
            baseline_val = anal.calculate_file_baseline(
                num_waveforms=500, 
                baseline_samples=OPTIMAL_BASELINE_SAMPLES
            )
            
            # 2. Calculate average FFT
            avg_fft_data = anal.calculate_average_fft(baseline_val)
            
            # 3. Calculate convergence metrics
            convergence_data = anal.calculate_convergence_metrics(
                metrics=[np.mean, np.median, np.std], 
                max_samples=4000, 
                increment=100, # Larger increment for faster processing
                baseline_end_fraction=0.4
            )
            
            # 4. Calculate noise distribution histogram
            noise_hist_data = anal.calculate_noise_histogram(
                baseline_correction_val=baseline_val,
                num_waveforms=500,
                baseline_samples=OPTIMAL_BASELINE_SAMPLES,
                num_bins=150
            )
            
            # --- Collate results and save ---
            all_analysis_data = {
                'baseline': baseline_val,
                'avg_fft': avg_fft_data,
                'convergence': convergence_data,
                'noise_hist': noise_hist_data
            }
            
            # Use the base filename as the group name in the output file
            group_name = os.path.basename(file_path).replace('.h5', '')
            save_results_to_h5(f_out, group_name, file_path, all_analysis_data)
            
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    # --- Define your list of input files ---
    # For this example, we'll just use the same file multiple times to simulate a list.
    # In a real scenario, these would be different file paths.
    INPUT_FILE_LIST = [
        './pmt_data/muon_test/MUON_TEST_V1_2025-08-22_11-02-09_1350.h5',
        './pmt_data/muon_test/MUON_TEST_V1_2025-08-22_09-59-05_1400.h5',
        './pmt_data/muon_test/MUON_TEST_V1_2025-08-22_13-29-35_1450.h5',
    ]
    
    OUTPUT_H5_FILE = 'analysis/pmt_data_22.08.25.h5'
    
    # --- Run the pipeline ---
    process_files_pipeline(INPUT_FILE_LIST, OUTPUT_H5_FILE)