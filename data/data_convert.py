import os
import re
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Directory containing your raw CSV files.
SOURCE_DIRECTORY = "./pmt_data/voltage_test" 

# 2. Directory where the new HDF5 files will be saved.
TARGET_DIRECTORY = SOURCE_DIRECTORY

# 3. IMPORTANT: Number of header lines to skip in the CSV files.
CSV_HEADER_ROWS_TO_SKIP = 15 

# 4. Assumed time step in seconds since it's not in the CSV data.
ASSUMED_X_INCREMENT_SECONDS = 1e-9

# --- END OF CONFIGURATION ---


def convert_csv_groups_to_h5(source_dir, target_dir):
    """
    Finds CSV files in the source directory, groups them by voltage,
    and converts each group into a single HDF5 file.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    if not os.path.exists(target_dir):
        print(f"Target directory '{target_dir}' not found. Creating it...")
        os.makedirs(target_dir)

    voltage_pattern = re.compile(r'_(\d+)[Vv]_')
    
    # --- 1. Group files by voltage ---
    file_groups = {}
    print(f"Scanning for CSV files in '{source_dir}'...")
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.csv'):
            match = voltage_pattern.search(filename)
            if match:
                voltage = match.group(1)
                if voltage not in file_groups:
                    file_groups[voltage] = []
                file_groups[voltage].append(os.path.join(source_dir, filename))
            else:
                print(f"  - Warning: Could not determine voltage for '{filename}'. Skipping.")

    if not file_groups:
        print("No valid CSV files found to process.")
        return

    print(f"Found {len(file_groups)} voltage groups: {list(file_groups.keys())}")
    print(f"Using an assumed time step (x_increment) of {ASSUMED_X_INCREMENT_SECONDS} seconds for all files.")

    # --- 2. Process each group ---
    for voltage, file_list in file_groups.items():
        print(f"\n--- Processing group: {voltage}V ({len(file_list)} files) ---")
        
        all_waveforms = []

        try:
            for file_path in tqdm(file_list, desc=f"Reading {voltage}V files"):
                # *** THE FIX IS HERE ***
                # We remove 'usecols=[1]' to read the single column that exists.
                df = pd.read_csv(
                    file_path, 
                    header=None, 
                    skiprows=CSV_HEADER_ROWS_TO_SKIP
                )
                
                # Now, df.iloc[:, 0] will correctly access the first (and only) column.
                waveform_data = df.iloc[:, 0].values
                all_waveforms.append(waveform_data)

            if not all_waveforms:
                print("  - No valid waveforms were read for this group. Skipping.")
                continue

            # --- 3. Create and write to HDF5 file ---
            # Using np.vstack is more robust for combining arrays that might have
            # slightly different lengths, though it requires them to be consistent.
            waveforms_array = np.vstack(all_waveforms).astype('f4')
            
            output_filename = f"CONVERTED_BAD_DATA_{voltage}V.h5"
            output_path = os.path.join(target_dir, output_filename)
            
            print(f"  - Assembled {waveforms_array.shape[0]} waveforms of length {waveforms_array.shape[1]}.")
            print(f"  - Saving to '{output_path}'...")

            with h5py.File(output_path, 'w') as f:
                f.attrs['x_increment_seconds'] = ASSUMED_X_INCREMENT_SECONDS
                f.attrs['start_time_utc'] = datetime.utcnow().isoformat()
                f.attrs['conversion_source'] = f"Converted from {len(file_list)} CSV files."
                
                f.create_dataset('waveforms', data=waveforms_array)

            print(f"  - Successfully created HDF5 file for {voltage}V.")

        except Exception as e:
            print(f"\nAn error occurred while processing the {voltage}V group: {e}")
            print("This could be due to an inconsistent number of data points in the CSVs for this group, or a formatting error.")

    print("\nConversion process finished.")


if __name__ == "__main__":
    convert_csv_groups_to_h5(SOURCE_DIRECTORY, TARGET_DIRECTORY)