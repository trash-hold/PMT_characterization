# [X] Reading data from file
# [X] FFT analysis:
#   [X] FFT
#   [X] Filtering + Checking amplitude
#   [ ]
# [ ] Trying simple active threshold method -- read N entry samples and try to establish some treshold
#   [ ] Finding optimal amount of samples -- check at multiple Vin
#   [ ] Comparing multiple methods 

# Core goals:
# a) Getting muon distribution -- for more optimized timeout definition
# b) Figuring out active threshold 

import h5py
import sys
from enum import Enum, auto
import dask.array as da
import numpy as np
import matplotlib.pyplot  as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import norm
from tqdm import tqdm
from scipy.optimize import curve_fit


class FileFormat(Enum):
    H5 = auto()

class DataConverter():
    def __init__(self, path: str, format: FileFormat):
        self.path = path
        self.format = format
        self.chunks = 500

    def set_chunks(self, chunks: int) -> None:
        self.chunks = chunks

    def info(self) -> None:
        try:
            if self.format == FileFormat.H5:
                with h5py.File(self.path, 'r') as f:
                    print(f"--- Inspecting File: {self.path} ---")

                    # Check for the main dataset to count records
                    if 'waveforms' in f:
                        # The number of records is the size of the first dimension
                        num_records = f['waveforms'].shape[0]
                        print(f"\n[SUCCESS] Found {num_records} records (waveforms) written to the file.")
                    else:
                        print("\n[ERROR] 'waveforms' dataset not found in the file.")
                        return

                    # --- Optional: Print other useful information ---
                    print("\nFile Metadata (Attributes):")
                    if not f.attrs:
                        print("  No metadata found.")
                    else:
                        for key, value in f.attrs.items():
                            print(f"  - {key}: {value}")

                    print("\nDatasets in file (name, shape, data type):")
                    for name, dset in f.items():
                        print(f"  - {name}: {dset.shape}, {dset.dtype}")
                    
                    print("\n--- Inspection Complete ---")

        except Exception as e: 
            raise FileNotFoundError()
        
    def get_all_data(self) -> da:
        try:
            if self.format == FileFormat.H5:
                with h5py.File(self.path, 'r') as f:
                    return da.from_array(f['waveforms'], chunks=(self.chunks, -1))
                
            else:
                return None

        except Exception as e:
            raise ValueError()
    
    def get_single_data(self, index) -> np.array:
        try:
            if self.format == FileFormat.H5:
                with h5py.File(self.path, 'r') as f:
                    return np.array(f['waveforms'][index])
            else: 
                return None
        except Exception as e:
            raise ValueError()
        
    def get_timestep(self) -> float:
        try:
            if self.format == FileFormat.H5:
                with h5py.File(self.path, 'r') as f:
                    return float(f.attrs['x_increment_seconds'])
            else: 
                return None
        except Exception as e:
            raise ValueError()


class FilterType(Enum):
    BUTTER = auto()


class DataAnalyzer():
    def __init__(self, dataConverter: DataConverter):
        self.dc = dataConverter

    def fft_nofilter(self, index: int, dt: float) -> np.array:
        '''
        Returns two np.arrays first is a series of frequencies and the second is amplitudes
        '''
        # First get data 
        data: np.array = self.dc.get_single_data(index)

        # Data needed for FFT correct plott
        N = data.shape[0]           # Sample size
        T = dt # Time increment 
        freq_axis = np.fft.fftfreq(N, T)[:N//2]

        # FFT
        data_fft = np.fft.fft(data)
        data_fft_magnitude = 2.0/N * np.abs(data_fft[0:N//2])

        return freq_axis, data_fft_magnitude
    
    def filter_data(self, data: np.array, dt: float, cutoff: float, order: int, type: FilterType) -> np.array:
        # Data needed for FFT correct plott
        N = data.shape[0]           # Sample size
        T = dt                      # Time increment 
        freq_axis = np.fft.fftfreq(N, T)[:N//2]

        # Filter definition
        nyq = 0.5/T
        normlised_cutoff = cutoff/nyq

        b = None
        a = None
        
        if type == FilterType.BUTTER:
            b, a = butter(order, normlised_cutoff, btype='lowpass')
        else:
            return None
        
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def fft_filter(self, index: int, dt: float, cutoff: float, order: int, type: FilterType) -> np.array:
        '''
        Utilizes scipy filters, returns two np.arrays first is a series of frequencies and the second is amplitudes
        '''
        # First get data 
        data: np.array = self.dc.get_single_data(index)

        # Data needed for FFT correct plott
        N = data.shape[0]           # Sample size
        T = dt                      # Time increment 
        freq_axis = np.fft.fftfreq(N, T)[:N//2]
        
        filtered_data = self.filter_data(data, dt, cutoff, order, type)

        # FFT
        data_fft = np.fft.fft(filtered_data)
        data_fft_magnitude = 2.0/N * np.abs(data_fft[0:N//2])
        
        return freq_axis, data_fft_magnitude 

    def analyze_noise_convergence(self, max_samples: int = 4000, increment: int = 10, baseline_end_fraction: float = 0.4):
        """
        Analyzes how the median of the baseline converges as more samples are taken.

        Args:
            max_samples (int): The maximum number of baseline samples to check.
            increment (int): The step size for increasing the number of samples.
            baseline_end_fraction (float): The fraction of the waveform to consider as baseline (e.g., 0.4 for the first 40%).
        """
        print("Starting noise convergence analysis. This may take a while...")
        
        # Get the total number of waveforms in the file
        with h5py.File(self.dc.path, 'r') as f:
            num_waveforms = f['waveforms'].shape[0]
            total_points = f['waveforms'].shape[1]

        baseline_end_index = int(total_points * baseline_end_fraction)
        if max_samples > baseline_end_index:
            print(f"Warning: max_samples ({max_samples}) is larger than the baseline region ({baseline_end_index}). Adjusting.")
            max_samples = baseline_end_index

        split_sizes = list(range(increment, max_samples + increment, increment))
        
        # Dictionary to store results: {split_size: [median1, median2, ...]}
        results = {size: [] for size in split_sizes}

        # a) Loop through every waveform
        for i in tqdm(range(num_waveforms), desc="Processing Waveforms"):
            waveform = self.dc.get_single_data(i)
            baseline_data = waveform[:baseline_end_index]

            # For each waveform, calculate the median for various split sizes
            for size in split_sizes:
                median_val = np.median(baseline_data[:size])
                results[size].append(median_val)
        
        # b) Calculate the mean and standard deviation for each split size
        mean_of_medians = []
        std_of_medians = []
        
        for size in split_sizes:
            medians_for_this_size = np.array(results[size])
            mean_of_medians.append(np.mean(medians_for_this_size))
            std_of_medians.append(np.std(medians_for_this_size))

        # --- Plotting the results ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Mean of the medians (should be flat)
        ax1.set_title('Stability of the Baseline Median Measurement')
        ax1.plot(split_sizes, mean_of_medians, '.-', label='Mean of Medians')
        ax1.set_ylabel('Mean Baseline Value')
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Standard Deviation of the medians (the key plot!)
        ax2.set_title('Convergence of the Baseline Median Measurement')
        ax2.plot(split_sizes, std_of_medians, '.-', label='Std Dev of Medians')
        ax2.set_xlabel('Number of Samples in Baseline Calculation')
        ax2.set_ylabel('Standard Deviation (Uncertainty)')
        ax2.set_yscale('log') # Log scale is best to see the convergence
        ax2.grid(True, which="both")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Analysis complete.")
        return split_sizes, mean_of_medians, std_of_medians
    
    def plot_baseline_correction_effect(self, waveform_index: int, baseline_samples: int):
        """
        Demonstrates the effect of baseline correction on a single waveform.

        Args:
            waveform_index (int): The index of the waveform to plot from the file.
            baseline_samples (int): The number of samples from the start of the
                                    waveform to use for calculating the baseline median.
                                    This should be determined from the convergence analysis.
        """
        print(f"\n--- Demonstrating Baseline Correction on Waveform #{waveform_index} ---")
        print(f"Using {baseline_samples} samples to calculate the baseline median.")

        # --- 1. Get the data ---
        waveform = self.dc.get_single_data(waveform_index)
        if waveform is None:
            print("Could not retrieve waveform.")
            return

        T = self.dc.get_timestep()
        time_axis = np.arange(len(waveform)) * T

        # --- 2. Calculate the baseline offset ---
        # Isolate the baseline region
        baseline_region = waveform[:baseline_samples]
        
        # Calculate the median of this region
        baseline_offset = np.median(baseline_region)
        print(f"Calculated baseline offset (median): {baseline_offset:.6f}")

        # --- 3. Apply the correction ---
        corrected_waveform = waveform - baseline_offset

        # --- 4. Plotting ---
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot the original waveform
        ax.plot(time_axis, waveform, label='Original Signal', alpha=0.7)
        
        # Plot the corrected waveform
        ax.plot(time_axis, corrected_waveform, label='Baseline Corrected Signal', linewidth=2)
        
        # Add a horizontal line at y=0 to make the correction obvious
        ax.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Level')

        # Add labels and title
        ax.set_title(f'Effect of Baseline Correction on Waveform #{waveform_index}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True)
        ax.legend()
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        plt.tight_layout()
        plt.show()

        return baseline_offset
    
    def calculate_convergence_metrics(self, 
                                      metrics: list, 
                                      max_samples: int = 4000, 
                                      increment: int = 10, 
                                      baseline_end_fraction: float = 0.4) -> dict:
        """
        Calculates convergence statistics for a list of functions in a single pass.
        This is highly efficient as it avoids re-reading data.

        Args:
            metrics (list): A list of callable functions to apply to the data slices
                            (e.g., [np.median, np.std]).
            max_samples (int): The maximum number of baseline samples to check.
            increment (int): The step size for increasing the number of samples.
            baseline_end_fraction (float): Fraction of the waveform to use as baseline.

        Returns:
            dict: A dictionary where keys are function names and values are dicts
                  containing the results ('splits', 'means', 'stds').
        """
        print(f"\n--- Calculating Convergence for {len(metrics)} metric(s) in a single pass ---")
        
        with h5py.File(self.dc.path, 'r') as f:
            num_waveforms = f['waveforms'].shape[0]
            total_points = f['waveforms'].shape[1]

        baseline_end_index = int(total_points * baseline_end_fraction)
        if max_samples > baseline_end_index:
            max_samples = baseline_end_index

        split_sizes = list(range(increment, max_samples + increment, increment))
        
        # Create a list of dictionaries to store raw results, one for each metric
        # e.g., all_results = [ {10:[...], 20:[...]}, {10:[...], 20:[...]} ]
        all_results = [{size: [] for size in split_sizes} for _ in metrics]

        # --- The Single, Efficient Loop ---
        for i in tqdm(range(num_waveforms), desc="Processing Waveforms"):
            waveform = self.dc.get_single_data(i)
            baseline_data = waveform[:baseline_end_index]

            for size in split_sizes:
                data_slice = baseline_data[:size]
                # Apply each function to the same data slice
                for func_index, func in enumerate(metrics):
                    result_value = func(data_slice)
                    all_results[func_index][size].append(result_value)
        
        # --- Process the raw results into a clean output dictionary ---
        final_output = {}
        for func_index, func in enumerate(metrics):
            raw_results_for_func = all_results[func_index]
            
            mean_of_results = [np.mean(raw_results_for_func[size]) for size in split_sizes]
            std_of_results = [np.std(raw_results_for_func[size]) for size in split_sizes]
            
            # Use the function's name as the key for the final dictionary
            func_name = func.__name__ if hasattr(func, '__name__') else f"metric_{func_index}"
            final_output[func_name] = {
                'splits': split_sizes,
                'means': mean_of_results,
                'stds': std_of_results
            }
            
        return final_output
    
    def run_fft(self) -> None:
        # First get data 
        data: np.array = self.dc.get_single_data(3)

        # Data needed for FFT correct plott
        N = data.shape[0]           # Sample size
        T = self.dc.get_timestep()  # Time increment 

        time_axis = np.arange(N) * T
        filtered_data = self.filter_data(data, T, 80e6, 5, FilterType.BUTTER)
        freq_axis, data_fft_magnitude = self.fft_nofilter(3, T)
        freq_axis, filt_fft_magnitude = self.fft_filter(3, T, 80e6, 5, FilterType.BUTTER)

        # Plot
        fig, (ax_time, ax_freq) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Time Domain
        ax_time.set_title("Waveform in Time Domain")
        ax_time.plot(time_axis, data, label='Original Signal')
        ax_time.plot(time_axis, filtered_data, label='Filtered Signal', alpha=0.8)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude (ADC counts or V)")
        ax_time.legend()
        ax_time.grid(True)

        # Plot 2: Frequency Domain (FFT)
        ax_freq.set_title("Frequency Spectrum (FFT)")
        ax_freq.plot(freq_axis, data_fft_magnitude, label='Original FFT')
        ax_freq.plot(freq_axis, filt_fft_magnitude, label='Filtered FFT')
        

        ax_freq.set_yscale('log')  
        ax_freq.set_xscale('log') 
        
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Amplitude")
        ax_freq.legend()
        ax_freq.grid(True, which="both")
        ax_freq.set_ylim(bottom=1e-7)

        plt.tight_layout()
        plt.show()

    def analyze_noise_distribution(self, num_waveforms: int, baseline_samples: int, num_bins: int = 100):
        """
        Collects baseline noise from multiple waveforms and plots its distribution
        as a histogram, comparing it to a Gaussian fit.

        Args:
            num_waveforms (int): The number of waveforms to sample noise from.
            baseline_samples (int): The number of samples from the start of each
                                    waveform to consider as baseline noise.
            num_bins (int): The number of bins to use for the histogram plot.
        """
        print(f"\n--- Analyzing Noise Distribution ---")
        print(f"Collecting baseline noise from {num_waveforms} waveforms...")

        # This list will hold the baseline-corrected noise from all waveforms
        noise_pool = []

        # Loop through the specified number of waveforms
        for i in tqdm(range(num_waveforms), desc="Sampling Noise"):
            waveform = self.dc.get_single_data(i)
            
            # Perform baseline correction on the fly
            baseline_region = waveform[:baseline_samples]
            baseline_offset = np.median(baseline_region)
            corrected_baseline = baseline_region - baseline_offset
            
            # Add the clean noise data to our pool
            noise_pool.append(corrected_baseline)
            
        # Concatenate the list of arrays into a single, flat 1D array
        all_noise_points = np.concatenate(noise_pool)
        
        print(f"Collected a total of {len(all_noise_points)} noise data points.")

        # --- Statistical Analysis ---
        # Calculate the mean and standard deviation of the collected noise
        noise_mean = np.mean(all_noise_points)
        noise_std = np.std(all_noise_points)
        
        print(f"Noise Mean: {noise_mean:.6f} (should be close to 0)")
        print(f"Noise Standard Deviation (σ): {noise_std:.6f}")

        # --- Plotting the Distribution ---
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the histogram of the noise. 'density=True' normalizes it to a probability distribution.
        ax.hist(all_noise_points, bins=num_bins, density=True, 
                label='Measured Noise Distribution', alpha=0.7)

        # --- Overlay a perfect Gaussian curve for comparison ---
        # Create an x-axis that spans the range of the noise
        x_fit = np.linspace(all_noise_points.min(), all_noise_points.max(), 1000)
        
        # Calculate the Gaussian probability density function (PDF) using the measured mean and std
        y_fit = norm.pdf(x_fit, loc=noise_mean, scale=noise_std)
        
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label='Gaussian Fit (μ, σ)')
        
        # --- Finalize the plot ---
        ax.set_title('Distribution of Baseline Noise')
        ax.set_xlabel('Amplitude (after baseline correction)')
        ax.set_ylabel('Probability Density')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def fit_noise_distribution(self, num_waveforms: int, baseline_samples: int, num_bins: int = 150):
        """
        Fits a Gaussian to the normalized noise distribution (probability density).
        This provides a direct visual comparison of the shapes.
        """
        print(f"\n--- Fitting Gaussian to Normalized Noise Distribution ---")
        
        # --- Step 1: Collect all the noise points (same as before) ---
        noise_pool = []
        for i in tqdm(range(num_waveforms), desc="Sampling Noise"):
            waveform = self.dc.get_single_data(i)
            baseline_offset = np.median(waveform[:baseline_samples])
            corrected_baseline = waveform[:baseline_samples] - baseline_offset
            noise_pool.append(corrected_baseline)
        all_noise_points = np.concatenate(noise_pool)

        # --- Step 2: Prepare the NORMALIZED histogram data ---
        # The key is the 'density=True' argument. This changes the y-axis from counts to probability density.
        hist_density, bin_edges = np.histogram(all_noise_points, bins=num_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # --- Step 3: Define the Gaussian PDF function ---
        # This is the mathematical formula for the PDF.
        def gaussian_pdf(x, mean, std_dev):
            # Note: The amplitude is now part of the formula and not a separate fitting parameter.
            return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) / std_dev)**2 / 2)

        # --- Step 4: Provide initial guesses ---
        mean_guess = np.mean(all_noise_points)
        std_dev_guess = np.std(all_noise_points)
        p0 = [mean_guess, std_dev_guess]

        # --- Step 5: Run the curve_fit ---
        try:
            # popt now contains the optimized [mean, std_dev]
            popt, pcov = curve_fit(gaussian_pdf, bin_centers, hist_density, p0=p0)
            perr = np.sqrt(np.diag(pcov))
            
            print("\n--- Fit Results (from Normalized Data) ---")
            print(f"Fitted Mean (μ):     {popt[0]:.6f} ± {perr[0]:.6f}")
            print(f"Fitted Std Dev (σ):  {popt[1]:.6f} ± {perr[1]:.6f}")

        except RuntimeError:
            print("Error: The curve fit failed to converge.")
            return

        # --- Step 6: Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot the normalized histogram
        ax.hist(all_noise_points, bins=num_bins, density=True, 
                label='Measured Noise Distribution', alpha=0.7)

        # Plot the fitted Gaussian PDF. It will now have the same scale!
        x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        ax.plot(x_fit, gaussian_pdf(x_fit, *popt), 'r-', linewidth=2, 
                label='Fitted Gaussian PDF')
        
        ax.set_title('Normalized Gaussian Fit to Baseline Noise')
        ax.set_xlabel('Amplitude (after baseline correction)')
        ax.set_ylabel('Probability Density') # <-- Note the changed label
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def fit_multi_gaussian_distribution(self, num_waveforms: int, baseline_samples: int, num_bins: int = 150):
        """
        Performs a multi-Gaussian fit to the quantized noise distribution.
        This correctly models the ADC quantization effect.
        """
        print(f"\n--- Performing Multi-Gaussian Fit to Noise Distribution ---")
        
        # --- Step 1: Collect noise and create histogram data ---
        noise_pool = []
        for i in tqdm(range(num_waveforms), desc="Sampling Noise"):
            waveform = self.dc.get_single_data(i)
            baseline_offset = np.median(waveform[:baseline_samples])
            corrected_baseline = waveform[:baseline_samples] - baseline_offset
            noise_pool.append(corrected_baseline)
        all_noise_points = np.concatenate(noise_pool)

        hist_counts, bin_edges = np.histogram(all_noise_points, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # --- Step 2: Find the prominent peaks in the histogram ---
        # We set a height threshold to ignore small noise spikes in the histogram itself.
        min_peak_height = np.max(hist_counts) * 0.1 
        peak_indices, _ = find_peaks(hist_counts, height=min_peak_height)
        
        num_peaks = len(peak_indices)
        print(f"Found {num_peaks} prominent ADC peaks to fit.")

        # --- Step 3: Define the multi-Gaussian model ---
        def multi_gaussian_model(x, *params):
            # params is a flat list: [amp1, center1, std1, amp2, center2, std2, ...]
            y = np.zeros_like(x)
            for i in range(0, len(params), 3):
                amp, center, std = params[i], params[i+1], params[i+2]
                y += amp * np.exp(-((x - center) / std)**2 / 2)
            return y

        # --- Step 4: Create excellent initial guesses for the fit ---
        p0 = []
        for i in peak_indices:
            amplitude_guess = hist_counts[i]
            center_guess = bin_centers[i]
            std_dev_guess = (bin_edges[1] - bin_edges[0]) * 2 # Guess std is a few bins wide
            p0.extend([amplitude_guess, center_guess, std_dev_guess])

        # --- Step 5: Run the curve_fit ---
        try:
            popt, pcov = curve_fit(multi_gaussian_model, bin_centers, hist_counts, p0=p0)
        except RuntimeError as e:
            print(f"Error: The multi-Gaussian fit failed to converge. {e}")
            return

        # --- Step 6: Plot the results ---
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.bar(bin_centers, hist_counts, width=(bin_centers[1]-bin_centers[0]), 
               label='Measured Noise Distribution', alpha=0.6)

        x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
        ax.plot(x_fit, multi_gaussian_model(x_fit, *popt), 'r-', linewidth=2, 
                label='Total Multi-Gaussian Fit')
        
        ax.set_title('Multi-Gaussian Fit to Quantized Noise')
        ax.set_xlabel('Amplitude (after baseline correction)')
        ax.set_ylabel('Counts per Bin')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # --- Step 7: Interpret and print the results ---
        print("\n--- Fit Results ---")
        stds = []
        centers = []
        for i in range(0, len(popt), 3):
            amp, center, std = popt[i], popt[i+1], popt[i+2]
            stds.append(std)
            centers.append(center)
            print(f"Peak {i//3 + 1}: Center = {center:.6f}, Std Dev (σ) = {np.abs(std):.6f}")

        avg_physical_std = np.mean(np.abs(stds))
        avg_adc_step = np.mean(np.diff(sorted(centers)))
        print("\n--- System Characterization ---")
        print(f"Average Physical Noise (σ_physical): {avg_physical_std:.6f}")
        print(f"Average ADC Voltage Step: {avg_adc_step:.6f}")
    
    def plot_convergence_results(self, results_data: dict, plot_title: str):
        """
        Plots the convergence data, including a plot of the relative change
        compared to the final value.

        Args:
            results_data (dict): The dictionary from calculate_convergence_metrics.
            plot_title (str): The main title for the plots.
        """
        # Create a figure with THREE subplots, stacked vertically
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
        fig.suptitle(plot_title, fontsize=16)

        # Loop through each metric in the results dictionary
        for func_name, data in results_data.items():
            split_sizes = np.array(data['splits'])
            means = np.array(data['means'])
            stds = np.array(data['stds'])

            # --- Plot 1: Mean of the metric ---
            ax1.plot(split_sizes, means, '.-', label=f'Mean of {func_name}')
            
            # --- Plot 2: Standard Deviation of the metric ---
            ax2.plot(split_sizes, stds, '.-', label=f'Std Dev of {func_name}')

            # --- Plot 3: NEW - Relative Change Calculation ---
            # The "final value" is the last one in the series (most stable)
            final_mean = means[-1]
            final_std = stds[-1]
            
            # Calculate relative change, avoiding division by zero
            if abs(final_mean) > 1e-9:
                relative_change_mean = np.abs((means - final_mean) / final_mean)
                ax3.plot(split_sizes, relative_change_mean, '.-', 
                         label=f'Rel. Change in Mean ({func_name})')

            if abs(final_std) > 1e-9:
                relative_change_std = np.abs((stds - final_std) / final_std)
                ax4.plot(split_sizes, relative_change_std, '.--', 
                         label=f'Rel. Change in Std ({func_name})')

        # --- Configure Axes ---
        ax1.set_title('Absolute Value of Metrics')
        ax1.set_ylabel('Mean of Metric')
        ax1.grid(True)
        ax1.legend()

        ax2.set_title('Uncertainty of Metrics')
        ax2.set_ylabel('Standard Deviation (Uncertainty)')
        ax2.set_yscale('log')
        ax2.grid(True, which="both")
        ax2.legend()
        
        relative = [ax3, ax4]
        for ax in relative:
            ax.set_title('Convergence: Relative Change Compared to Final Value')
            ax.set_title('Convergence: Relative Change Compared to Final Value')
            ax.set_xlabel('Number of Samples in Calculation')
            ax.set_ylabel('Relative Change (Fraction)')
            ax.set_yscale('log') # Log scale is essential here
            ax.grid(True, which="both")
            ax.legend()
            
            # Add a horizontal line for 1% (0.01) and 0.1% (0.001) convergence
            ax.axhline(0.01, color='gray', linestyle=':', linewidth=1, label='1% Change')
            ax.axhline(0.001, color='black', linestyle=':', linewidth=1, label='0.1% Change')
            ax.legend() # Re-call legend to include axhline labels

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust for suptitle
        plt.show()

FILE_PATH = './pmt_data/muon_test/MUON_TEST_V1_2025-08-22_13-29-35_1450.h5'

def absolute_mean(data: np.array) -> np.array:
    return np.mean(abs(data))

if __name__ == "__main__":
    conv = DataConverter(FILE_PATH, FileFormat.H5)
    #conv.info()

    anal = DataAnalyzer(conv)
    '''
    anal.analyze_noise_convergence()

    OPTIMAL_BASELINE_SAMPLES = 3500 # <-- CHANGE THIS based on your convergence plot

    # Now, plot the effect on the first waveform (index 0)
    anal.plot_baseline_correction_effect(
        waveform_index=0, 
        baseline_samples=OPTIMAL_BASELINE_SAMPLES
    )
    
    anal.fit_multi_gaussian_distribution(
        num_waveforms=500, 
        baseline_samples=3500,
        num_bins=150 # More bins can give a more detailed view
    )
    '''

    metrics_to_run = [np.mean, absolute_mean, np.median, np.std]

    # --- STEP 2: Run the calculation ONCE to get all the data ---
    # This is now much more efficient.
    convergence_data = anal.calculate_convergence_metrics(
        metrics=metrics_to_run,
        max_samples=4000
    )

    # --- STEP 3: Plot the results ---
    # The plotting function uses the data we just calculated.
    anal.plot_convergence_results(
        results_data=convergence_data,
        plot_title='Convergence of Baseline and Noise Level Metrics'
    )
    