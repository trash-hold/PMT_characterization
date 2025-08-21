# [ ] Reading data from file
# [ ] FFT analysis:
#   [ ] FFT
#   [ ] Filtering + Checking amplitude
#   [ ]
# [ ] Trying simple active threshold method -- read N entry samples and try to establish some treshold

# Core goals:
# a) Getting muon distribution -- for more optimized timeout definition
# b) Figuring out active threshold 

import h5py
import sys
from enum import Enum, auto
import dask.array as da
import numpy as np
import matplotlib.pyplot  as plt
from scipy.signal import butter, filtfilt


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


        

FILE_PATH = './MUON_TEST_V1_2025-08-21_09-27-25.h5'

if __name__ == "__main__":
    conv = DataConverter(FILE_PATH, FileFormat.H5)
    #conv.info()

    anal = DataAnalyzer(conv)
    anal.run_fft()
    