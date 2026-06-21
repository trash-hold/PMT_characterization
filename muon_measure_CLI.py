# measurement_console_app.py
import sys
import os
import time
from datetime import datetime
import h5py
import json
import numpy as np
import argparse

# Assuming 'drivers' is a package in your project
from drivers.common.driver_map import create_driver as OscilloscopeDriver

# --- Configuration ---
CONFIG = './secret/TDS3054C_profile.json'
SAVE_DIRECTORY = "./data/pmt_data/muon_test"
TITLE_PREFIX = "MUON_TEST_V1"


def run_acquisition(driver, num_acquisitions, filename, x_increment):
    """
    Main function to run the data acquisition loop.

    Args:
        driver: The initialized oscilloscope driver object.
        num_acquisitions (int): The number of waveforms to acquire.
        filename (str): The path to the HDF5 file for saving data.
        x_increment (float): The time interval between points in the waveform.
    """
    total_start_time = time.perf_counter()
    acquisitions_completed = 0
    
    try:
        with h5py.File(filename, 'w') as f:
            # Store metadata as attributes
            f.attrs['x_increment_seconds'] = x_increment
            f.attrs['start_time_utc'] = datetime.utcnow().isoformat()
            
            # Create resizable datasets for efficiency
            dset_waveforms = f.create_dataset('waveforms', (0, 10000), maxshape=(None, 10000), dtype='f4')
            dset_times = f.create_dataset('acquisition_times', (0,), maxshape=(None,), dtype='f4')
            dset_indices = f.create_dataset('run_indices', (0,), maxshape=(None,), dtype='i4')

            print(f"Starting acquisition of {num_acquisitions} waveforms...")
            print(f"Saving data to: {filename}")
            print("Press Ctrl+C to stop early.")

            for i in range(num_acquisitions):
                acquisitions_completed = i
                print(f"Running acquisition {i+1}/{num_acquisitions}...", end='\r')
                
                # Time the waveform reading operation
                loop_start_time = time.perf_counter()
                driver.sample(timeout=500)
                waveform = driver.get_waveform(1)
                loop_end_time = time.perf_counter()
                delta_t = loop_end_time - loop_start_time
                
                if waveform is not None:
                    # Resize datasets and append new data
                    dset_waveforms.resize(i + 1, axis=0)
                    dset_times.resize(i + 1, axis=0)
                    dset_indices.resize(i + 1, axis=0)
                    
                    dset_waveforms[i] = waveform
                    dset_times[i] = delta_t
                    dset_indices[i] = i
                else:
                    # Move to the next line to not overwrite the progress message
                    print(f"\nWarning: Failed to get waveform on run {i+1}")
            
            acquisitions_completed += 1 # To account for the last successful run

    except KeyboardInterrupt:
        print("\nAcquisition stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        total_end_time = time.perf_counter()
        duration = total_end_time - total_start_time
        avg_time = duration / acquisitions_completed if acquisitions_completed > 0 else 0
        
        print("\n--- Acquisition Summary ---")
        print(f"Finished. Completed {acquisitions_completed} acquisitions.")
        print(f"Total time: {duration:.2f}s.")
        print(f"Average time per acquisition: {avg_time:.2f}s.")
        print("--------------------------")


def main():
    """
    Main entry point for the console application.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Oscilloscope Data Acquisition Console App")
    parser.add_argument(
        '-n', '--num_acquisitions', 
        type=int, 
        default=1000, 
        help='Number of acquisitions to perform.'
    )
    args = parser.parse_args()

    # --- 1. Connect to the device ---
    osc_driver = None
    try:
        print("Attempting to connect to the oscilloscope...")
        with open(CONFIG, 'r') as f:
            config = json.load(f)
        osc_driver = OscilloscopeDriver("TDS3054C", config["connection_params"])

        if osc_driver.test_connection():
            x_increment = float(osc_driver.get_horizontal_increment())
            print("Device connected successfully.")
        else:
            print("Error: Device not responding. Please check the connection and configuration.")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{CONFIG}'")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during connection: {e}")
        sys.exit(1)

    # --- 2. Prepare for acquisition ---
    if not os.path.exists(SAVE_DIRECTORY):
        try:
            os.makedirs(SAVE_DIRECTORY)
            print(f"Created save directory: {SAVE_DIRECTORY}")
        except OSError as e:
            print(f"Error: Could not create save directory '{SAVE_DIRECTORY}'. {e}")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(SAVE_DIRECTORY, f"{TITLE_PREFIX}_{timestamp}.h5")
    
    # --- 3. Run the acquisition ---
    run_acquisition(osc_driver, args.num_acquisitions, filename, x_increment)


if __name__ == "__main__":
    main()