# measurement_app.py
import sys
import os
import time
from datetime import datetime
import h5py
import json
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLineEdit, QLabel, QSpinBox)
from PySide6.QtCore import QObject, QThread, Signal

import pyqtgraph as pg


from drivers.common.driver_map import create_driver as OscilloscopeDriver

# --- Configuration ---
CONFIG = './secret/TDS3054C_profile.json'
SAVE_DIRECTORY = "./data/pmt_data/muon_test"
TITLE_PREFIX = "MUON_TEST_V1"


class AcquisitionWorker(QObject):
    """
    Worker object that runs the data acquisition loop in a separate thread.
    """
    # Signals to communicate with the main GUI thread
    new_data = Signal(np.ndarray, int, float)  # waveform, index, delta_t
    progress = Signal(str)
    finished = Signal(str)

    def __init__(self, driver, num_acquisitions, filename, x_increment):
        super().__init__()
        self.driver = driver
        self.num_acquisitions = num_acquisitions
        self.x_increment = x_increment
        self.filename = filename
        self.is_running = True

    def run(self):
        """The main acquisition loop."""
        total_start_time = time.perf_counter()
        
        try:
            with h5py.File(self.filename, 'w') as f:
                # Create resizable datasets. This is efficient.
                # We store 10k points per waveform.
                f.attrs['x_increment_seconds'] = self.x_increment
                f.attrs['start_time_utc'] = datetime.utcnow().isoformat()
                dset_waveforms = f.create_dataset('waveforms', (0, 10000), maxshape=(None, 10000), dtype='f4')
                dset_times = f.create_dataset('acquisition_times', (0,), maxshape=(None,), dtype='f4')
                dset_indices = f.create_dataset('run_indices', (0,), maxshape=(None,), dtype='i4')

                for i in range(self.num_acquisitions):
                    if not self.is_running:
                        self.progress.emit("Acquisition stopped by user.")
                        break
                    
                    self.progress.emit(f"Running acquisition {i+1}/{self.num_acquisitions}...")
                    
                    # 3. Time the waveform reading operation
                    loop_start_time = time.perf_counter()
                    self.driver.sample()
                    waveform = self.driver.get_waveform(1)
                    loop_end_time = time.perf_counter()
                    delta_t = loop_end_time - loop_start_time
                    
                    if waveform is not None:
                        # 4. Save data to HDF5 file
                        # Resize datasets before writing new data
                        dset_waveforms.resize(i + 1, axis=0)
                        dset_times.resize(i + 1, axis=0)
                        dset_indices.resize(i + 1, axis=0)
                        
                        dset_waveforms[i] = waveform
                        dset_times[i] = delta_t
                        dset_indices[i] = i
                        
                        # Emit signal to update the plot in the GUI
                        self.new_data.emit(waveform, i + 1, delta_t)
                    else:
                        self.progress.emit(f"Warning: Failed to get waveform on run {i+1}")

        except Exception as e:
            self.progress.emit(f"Error: {e}")
        finally:
            total_end_time = time.perf_counter()
            duration = total_end_time - total_start_time
            avg_time = duration / (i + 1) if i > 0 else 0
            # 5. Report final statistics
            final_message = f"Finished. Total time: {duration:.2f}s for {i+1} acquisitions. Average: {avg_time:.2f}s/acq."
            self.finished.emit(final_message)

    def stop(self):
        """Stops the acquisition loop gracefully."""
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oscilloscope Data Acquisition")
        self.setGeometry(100, 100, 800, 600)

        # --- UI Elements ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Plotting widget
        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.plot(pen='y')
        self.plot_widget.setLabel('left', 'Voltage', 'V')
        self.plot_widget.setLabel('bottom', 'Time', 's')
        self.plot_widget.showGrid(x=True, y=True)

        # Controls
        controls_layout = QHBoxLayout()
        self.num_acq_label = QLabel("Number of Acquisitions:")
        self.num_acq_input = QSpinBox()
        self.num_acq_input.setRange(1, 100000)
        self.num_acq_input.setValue(1000)
        
        self.start_button = QPushButton("Start Acquisition")
        self.stop_button = QPushButton("Stop Acquisition")
        self.stop_button.setEnabled(False)

        self.status_label = QLabel("Status: Idle. Connect to device...")

        controls_layout.addWidget(self.num_acq_label)
        controls_layout.addWidget(self.num_acq_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)

        layout.addWidget(self.plot_widget)
        layout.addLayout(controls_layout)
        layout.addWidget(self.status_label)

        # --- Connect signals to slots ---
        self.start_button.clicked.connect(self.start_acquisition)
        self.stop_button.clicked.connect(self.stop_acquisition)

        # --- Oscilloscope Driver ---
        self.x_increment = None
        self.time_axis = None
        self.osc_driver = None
        self.connect_to_device()

    def connect_to_device(self):
        try:
            # To use your real driver, you might need to load a json config first
            with open(CONFIG, 'r') as f:
                config = json.load(f)
            self.osc_driver = OscilloscopeDriver("TDS3054C", config["connection_params"])
        

            if self.osc_driver.test_connection():
                self.x_increment = float(self.osc_driver.get_horizontal_increment())
                num_points = 10000
                self.time_axis = np.arange(num_points) * self.x_increment

                self.status_label.setText("Status: Device connected. Ready to start.")
            else:
                self.status_label.setText("Status: Device not responding.")
                self.start_button.setEnabled(False)
        except Exception as e:
            self.status_label.setText(f"Status: Failed to connect - {e}")
            self.start_button.setEnabled(False)
            
    def start_acquisition(self):
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)

        if self.x_increment is None:
            self.status_label.setText("Status: Error - Cannot start, timebase not acquired from device.")
            return

        num_acquisitions = self.num_acq_input.value()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(SAVE_DIRECTORY, f"{TITLE_PREFIX}_{timestamp}.h5")
        
        # 1. Create a QThread and a worker
        self.thread = QThread()
        self.worker = AcquisitionWorker(
            self.osc_driver, num_acquisitions, filename, self.x_increment
        )
        self.worker.moveToThread(self.thread)

        # 2. Connect worker signals to GUI slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_finished)
        self.worker.progress.connect(self.update_status)
        self.worker.new_data.connect(self.update_plot)

        # 3. Start the thread
        self.thread.start()

        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.num_acq_input.setEnabled(False)
        self.status_label.setText(f"Starting... Saving data to {filename}")

    def stop_acquisition(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
            self.stop_button.setEnabled(False) # Prevent multiple clicks

    def update_status(self, message):
        self.status_label.setText(f"Status: {message}")

    def update_plot(self, waveform, index, delta_t):
        self.plot_item.setData(waveform)
        self.plot_widget.setTitle(f"Acquisition #{index} (took {delta_t:.3f}s)")

    def on_finished(self, final_message):
        self.status_label.setText(f"Status: {final_message}")
        self.thread.quit()
        self.thread.wait()
        
        # Update UI state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.num_acq_input.setEnabled(True)

    def closeEvent(self, event):
        """Ensure the worker thread is stopped when closing the window."""
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())