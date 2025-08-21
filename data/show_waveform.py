import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# !!! IMPORTANT: Change this to the path of your HDF5 file !!!
FILE_PATH = './MUON_TEST_V1_2025-08-21_09-27-25.h5'

# Choose which waveform to plot from the file (0 is the first one)
WAVEFORM_INDEX_TO_PLOT = 0

# --- Main Script ---
try:
    # Open the HDF5 file in read-only mode
    with h5py.File(FILE_PATH, 'r') as f:
        
        # 1. Read the time increment from the file's metadata (attributes)
        # This is the time step between each sample point
        time_increment = f.attrs['x_increment_seconds']
        
        # 2. Access the 'waveforms' dataset
        waveforms_dset = f['waveforms']
        
        # Check if the requested index is valid
        num_records = waveforms_dset.shape[0]
        if WAVEFORM_INDEX_TO_PLOT >= num_records:
            print(f"Error: Index {WAVEFORM_INDEX_TO_PLOT} is out of bounds. "
                  f"File only contains {num_records} records (indices 0 to {num_records-1}).")
            exit()
            
        # 3. Read the specific waveform into a NumPy array
        # This is the only point where data is loaded from the file into memory
        single_waveform = waveforms_dset[WAVEFORM_INDEX_TO_PLOT]
        
        # 4. Create the corresponding time axis for the plot
        num_points = len(single_waveform)
        time_axis = np.arange(num_points) * time_increment

        # --- Plotting ---
        
        # 5. Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 6. Plot the waveform data against the calculated time axis
        ax.plot(time_axis, single_waveform, label=f'Waveform #{WAVEFORM_INDEX_TO_PLOT}')
        
        # 7. Add labels, title, and a grid for better readability
        ax.set_title('Acquired Oscilloscope Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (ADC counts or Volts)')
        ax.grid(True)
        ax.legend()
        
        # Use scientific notation for the x-axis if needed
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 8. Display the plot
        plt.tight_layout()
        plt.show()

except FileNotFoundError:
    print(f"Error: The file was not found at the path: {FILE_PATH}")
except KeyError as e:
    print(f"Error: A required dataset or attribute was not found in the HDF5 file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")