import sys
import time
import logging
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from scipy.signal import butter, sosfilt, sosfilt_zi

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 250
BUFFER_SECONDS = 5
CHANNELS = ["Fp1", "Fp2", "O1", "O2"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class OnlineFilter:
    def __init__(self, sfreq, n_channels):
        # 1. Bandpass 1-30Hz (Butterworth SOS)
        nyq = 0.5 * sfreq
        self.sos = butter(4, [1.0/nyq, 30.0/nyq], btype='band', output='sos')
        
        # Initial filter state
        self.zi = np.zeros((n_channels, self.sos.shape[0], 2))
        self.initialized = False
    
    def process(self, data_chunk):
        # data_chunk: (n_channels, n_samples)
        
        # Initialize state to match the first sample's DC offset
        # This prevents the massive "ringing" at startup
        if not self.initialized:
            # We construct the initial state so the filter starts "steady" at the first sample value
            for c in range(data_chunk.shape[0]):
                start_val = data_chunk[c, 0]
                # Scale the initial zi by the starting value
                self.zi[c] = sosfilt_zi(self.sos) * start_val
            self.initialized = True
            
        filtered, self.zi = sosfilt(self.sos, data_chunk, axis=-1, zi=self.zi)
        return filtered

class EEGWorker(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.running = True
        self.eeg = None
        self.scale_factor = 1.0
        self.scale_determined = False

    def run(self):
        from brainaccess.utils import acquisition
        from brainaccess.core.eeg_manager import EEGManager
        
        self.eeg = acquisition.EEG(mode="roll")
        cap = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
        
        try:
            with EEGManager() as mgr:
                logging.info(f"Connecting to {DEVICE_NAME}...")
                
                # Correct buffer size for 'roll' mode
                driver_buffer = int(SFREQ * BUFFER_SECONDS)
                
                self.eeg.setup(mgr, device_name=DEVICE_NAME, cap=cap, sfreq=SFREQ, zeros_at_start=driver_buffer)
                self.eeg.start_acquisition()
                time.sleep(1.0) # Warmup

                while self.running:
                    mne_data = self.eeg.get_mne(tim=0.04) # 40ms chunks
                    
                    if mne_data is not None and len(mne_data) > 0:
                        raw = mne_data.get_data()[:4, :]
                        
                        if raw.size > 0:
                            # --- AUTO SCALING LOGIC ---
                            if not self.scale_determined:
                                # Check mean amplitude
                                mean_val = np.mean(np.abs(raw))
                                if mean_val < 1.0:
                                    self.scale_factor = 1e6 # It's in Volts, convert to uV
                                    logging.info(f"Detected Volts. Scaling by 1e6. (Mean: {mean_val:.6f})")
                                else:
                                    self.scale_factor = 1.0 # It's already uV
                                    logging.info(f"Detected uV. Scale = 1.0. (Mean: {mean_val:.2f})")
                                self.scale_determined = True

                            # Apply scale
                            raw = raw * self.scale_factor
                            self.data_signal.emit(raw)
                    
                    time.sleep(0.01)
        except Exception as e:
            logging.error(f"Acquisition Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.eeg: 
                try: self.eeg.stop_acquisition()
                except: pass

    def stop(self):
        self.running = False
        self.wait()

class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroHackathon: Real-Time BCI")
        self.resize(1000, 600)
        
        # Layout
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout()
        cw.setLayout(layout)

        # Plot
        self.graph = pg.PlotWidget(title="EEG Live Stream")
        self.graph.showGrid(x=True, y=True, alpha=0.3)
        self.graph.setYRange(-500, 1500)
        layout.addWidget(self.graph)

        self.curves = []
        colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFFF00']
        for i in range(4):
            c = self.graph.plot(pen=pg.mkPen(color=colors[i], width=2), name=CHANNELS[i])
            self.curves.append(c)

        # Data
        self.buffer_len = int(SFREQ * BUFFER_SECONDS)
        self.data_buffer = np.zeros((4, self.buffer_len))
        self.filter = OnlineFilter(SFREQ, 4)
        
        # Thread
        self.worker = EEGWorker()
        self.worker.data_signal.connect(self.update_data)
        self.worker.start()

    def update_data(self, chunk):
        # 1. Filter
        clean_chunk = self.filter.process(chunk)
        
        # 2. Update Buffer
        n = clean_chunk.shape[1]
        self.data_buffer = np.roll(self.data_buffer, -n, axis=1)
        self.data_buffer[:, -n:] = clean_chunk
        
        # 3. Plot (with offset)
        offset = 200
        for i in range(4):
            # Plot every 2nd sample to reduce CPU load
            self.curves[i].setData(self.data_buffer[i, ::2] + (i * offset))

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    # Dark Theme
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    app.setPalette(palette)
    
    window = Dashboard()
    window.show()
    sys.exit(app.exec())