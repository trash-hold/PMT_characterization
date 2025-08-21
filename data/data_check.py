import h5py
import sys

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

    # Option 1: Pass the filename as a command-line argument
    # Example: python check_h5_records.py ./data/MUON_TEST_V1_...h5
    if len(sys.argv) > 1:
        file_to_check = sys.argv[1]
    
    # Option 2: Hardcode the filename here if you prefer
    else:
        # !!! IMPORTANT: Change this to the actual name of your data file !!!
        file_to_check = './MUON_TEST_V1_2025-08-21_09-27-25.h5'
        print(f"No file path provided. Using default: {file_to_check}\n")

    inspect_h5_file(file_to_check)