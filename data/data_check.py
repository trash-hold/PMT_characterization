import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

MAX_WAVEFORMS_TO_PLOT = 100

# --- END OF CONFIGURATION ---


def plot_all_waveforms(file_path):
    """
    Loads an HDF5 file and plots all of its waveforms on a single graph.
    """
    # --- 1. Basic Checks and File Loading ---
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"Loading data from: {os.path.basename(file_path)}")

    try:
        with h5py.File(file_path, 'r') as f:
            # Check for the 'waveforms' dataset
            if 'waveforms' not in f:
                print("Error: HDF5 file does not contain a 'waveforms' dataset.")
                return
            
            dset = f['waveforms']
            
            # Read metadata to create the time axis
            x_increment = f.attrs.get('x_increment_seconds', 1e-9) # Default to 1ns if not found
            
            # --- 2. Prepare Data for Plotting ---
            num_waveforms, num_samples = dset.shape
            
            # Determine how many waveforms to actually plot based on the limit
            num_to_plot = min(num_waveforms, MAX_WAVEFORMS_TO_PLOT)
            
            print(f"File contains {num_waveforms} waveforms. Plotting the first {num_to_plot}.")
            
            # Read the required slice of data into memory
            waveforms_to_plot = dset[:num_to_plot, :]
            
            # Create the time axis in microseconds for better readability
            time_axis = np.arange(num_samples) * x_increment
            time_axis_us = time_axis * 1e6

            # --- 3. Create the Plot ---
            plt.figure(figsize=(12, 7))
            
            # Plot each waveform with transparency
            for i in range(num_to_plot):
                plt.plot(time_axis_us, waveforms_to_plot[i, :], alpha=0.3, linewidth=0.8)

            # --- 4. Formatting ---
            plot_title = (
                f"Overlay of {num_to_plot} Waveforms\n"
                f"File: {os.path.basename(file_path)}"
            )
            plt.title(plot_title, fontsize=16)
            plt.xlabel("Time (µs)", fontsize=12)
            plt.ylabel("Amplitude (Volts or ADC counts)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout() # Adjusts plot to ensure everything fits
            
            # Invert y-axis for typical negative PMT pulses
            # Comment this out if your pulses are positive
            if np.mean(waveforms_to_plot) < 0:
                 plt.gca().invert_yaxis()

            plt.show()

    except Exception as e:
        print(f"An error occurred while trying to read or plot the file: {e}")

def inspect_h5_file(filepath):
    """
    Opens an HDF5 file and prints the number of records and metadata.

    Args:
        filepath (str): The path to the HDF5 file.
    """
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"--- Inspecting File: {filepath} ---")

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

    except FileNotFoundError:
        print(f"[ERROR] File not found at the specified path: {filepath}")
    except OSError:
        print(f"[ERROR] Could not open the file. It might be corrupted or not a valid HDF5 file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- How to use the script ---

    FILE = "./pmt_data/voltage_test/CONVERTED_BAD_DATA_1970V.h5"
    plot_all_waveforms(FILE)