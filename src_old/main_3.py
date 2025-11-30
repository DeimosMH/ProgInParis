import sys
import time
import logging
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore, QtGui
from scipy.signal import butter, sosfilt, sosfilt_zi

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 500
BUFFER_SECONDS = 20
CHANNELS = ["Fp1", "Fp2", "O1", "O2"]

# --- THRESHOLDS (Tunable) ---
THRESH_BLINK_UV = 150.0      # Frontal must exceed this to be a blink/movement
THRESH_ARTIFACT_UV = 40.0    # If Occipital (O1/O2) exceeds this, it's a HEAD MOVEMENT (Ignore)
THRESH_EMG_NOISE = 10.0      # If signal is too "fuzzy" (high freq), it's JAW/READING (Ignore)
COOLDOWN_MS = 600            # Time to wait after a detection before detecting again

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class FeatureDetector:
    def __init__(self):
        self.last_detection_time = 0
        self.state = "NEUTRAL"

    def process(self, chunk_uv):
        """
        chunk_uv: Shape (4, N) -> [Fp1, Fp2, O1, O2]
        Returns: "BLINK", "LEFT", "RIGHT", "ARTIFACT", or None
        """
        now = time.time() * 1000
        if now - self.last_detection_time < COOLDOWN_MS:
            return None

        # 1. Extract Metrics
        # Peak-to-Peak amplitude in this chunk
        ptp = np.ptp(chunk_uv, axis=1) 
        fp_max = np.max(ptp[:2])  # Max of Fp1, Fp2
        occ_max = np.max(ptp[2:]) # Max of O1, O2
        
        # "Roughness" (High freq noise) - indicative of Jaw/EMG
        # Calculate mean absolute difference between consecutive samples
        roughness = np.mean(np.abs(np.diff(chunk_uv, axis=1)), axis=1)
        avg_roughness = np.mean(roughness)

        # 2. Logic Tree
        
        # A. CHECK FOR NOISE/JAW (Reading artifact often has high EMG)
        if avg_roughness > THRESH_EMG_NOISE:
            return "IGNORED: NOISE/JAW"

        # B. CHECK FOR HEAD MOVEMENT (Global Artifact)
        # If Frontal is high, but Occipital is ALSO high -> Head movement
        if fp_max > THRESH_BLINK_UV and occ_max > THRESH_ARTIFACT_UV:
            self.last_detection_time = now # Trigger cooldown to ignore the whole shake
            return "IGNORED: HEAD MOVEMENT"

        # C. CHECK FOR BLINK / EYE MOVEMENT
        if fp_max > THRESH_BLINK_UV:
            # It's a strong frontal signal, and Occipital is quiet.
            
            # Check Correlation between Fp1 and Fp2 to distinguish Blink vs Saccade
            # Correlation requires variance. 
            if np.std(chunk_uv[0]) > 0 and np.std(chunk_uv[1]) > 0:
                corr = np.corrcoef(chunk_uv[0], chunk_uv[1])[0, 1]
            else:
                corr = 0

            self.last_detection_time = now
            
            # Blinks are highly correlated (Both eyes close together)
            if corr > 0.8:
                return "DETECTED: BLINK"
            
            # Eye Movements (L/R) often show lower correlation or anti-correlation
            # (EOG artifacts: Looking left makes one channel positive, one negative)
            elif corr < 0.6:
                # Simple heuristic: Which one spiked higher?
                if ptp[0] > ptp[1]:
                    return "DETECTED: LOOK LEFT" # Approximation
                else:
                    return "DETECTED: LOOK RIGHT" # Approximation
            
            return "DETECTED: EYE MOVEMENT"

        return None

class OnlineFilter:
    def __init__(self, sfreq, n_channels):
        # 1. Bandpass 1-30Hz (Butterworth SOS)
        nyq = 0.5 * sfreq
        self.sos = butter(4, [1.0/nyq, 30.0/nyq], btype='band', output='sos')
        self.zi = np.zeros((n_channels, self.sos.shape[0], 2))
        self.initialized = False
    
    def process(self, data_chunk):
        if not self.initialized:
            for c in range(data_chunk.shape[0]):
                start_val = data_chunk[c, 0]
                self.zi[c] = sosfilt_zi(self.sos) * start_val
            self.initialized = True
        filtered, self.zi = sosfilt(self.sos, data_chunk, axis=-1, zi=self.zi)
        return filtered

class EEGWorker(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object)
    status_signal = QtCore.pyqtSignal(str) # Send text updates to UI

    def __init__(self):
        super().__init__()
        self.running = True
        self.detector = FeatureDetector()
        self.scale_factor = 1.0
        self.scale_determined = False

    def run(self):
        from brainaccess.utils import acquisition
        from brainaccess.core.eeg_manager import EEGManager
        
        eeg = acquisition.EEG(mode="roll")
        cap = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
        
        try:
            with EEGManager() as mgr:
                logging.info(f"Connecting to {DEVICE_NAME}...")
                driver_buffer = int(SFREQ * BUFFER_SECONDS)
                eeg.setup(mgr, device_name=DEVICE_NAME, cap=cap, sfreq=SFREQ, zeros_at_start=driver_buffer)
                eeg.start_acquisition()
                time.sleep(1.0) 

                while self.running:
                    # Get 100ms chunk for detection (needs enough samples for correlation)
                    mne_data = eeg.get_mne(tim=0.1) 
                    
                    if mne_data is not None and len(mne_data) > 0:
                        raw = mne_data.get_data()[:4, :]
                        if raw.size > 0:
                            # Auto-scale
                            if not self.scale_determined:
                                if np.mean(np.abs(raw)) < 1.0: self.scale_factor = 1e6 
                                self.scale_determined = True
                            
                            # Clean Data
                            raw_uv = raw * self.scale_factor
                            
                            # Emit for Plotting
                            self.data_signal.emit(raw_uv)
                            
                            # Detect Features
                            result = self.detector.process(raw_uv)
                            if result:
                                self.status_signal.emit(result)

                    time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            try: eeg.stop_acquisition()
            except: pass

    def stop(self):
        self.running = False
        self.wait()

class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroHackathon: Robust Classifier")
        self.resize(1000, 700)
        
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout()
        cw.setLayout(layout)

        # 1. Status Label (Huge font for demo)
        self.lbl_status = QtWidgets.QLabel("READY")
        self.lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 48px; font-weight: bold; color: #00FF00; background-color: #222;")
        self.lbl_status.setFixedHeight(100)
        layout.addWidget(self.lbl_status)

        # 2. Plot
        self.graph = pg.PlotWidget(title="EEG Live Stream")
        self.graph.showGrid(x=True, y=True, alpha=0.3)
        self.graph.setYRange(-500, 1500)
        layout.addWidget(self.graph)

        self.curves = []
        colors = ['#00FFFF', '#FF00FF', '#00FF00', '#FFFF00'] # Cyan, Magenta, Green, Yellow
        for i in range(4):
            c = self.graph.plot(pen=pg.mkPen(color=colors[i], width=2), name=CHANNELS[i])
            self.curves.append(c)

        # Buffers
        self.buffer_len = int(SFREQ * BUFFER_SECONDS)
        self.data_buffer = np.zeros((4, self.buffer_len))
        self.filter = OnlineFilter(SFREQ, 4)
        
        # Thread
        self.worker = EEGWorker()
        self.worker.data_signal.connect(self.update_plot)
        self.worker.status_signal.connect(self.update_status)
        self.worker.start()

    def update_status(self, text):
        self.lbl_status.setText(text)
        
        # Color coding
        if "IGNORED" in text:
            self.lbl_status.setStyleSheet("font-size: 48px; font-weight: bold; color: #555555; background-color: #111;") # Dim gray
        elif "BLINK" in text:
            self.lbl_status.setStyleSheet("font-size: 48px; font-weight: bold; color: #FFFFFF; background-color: #00AA00;") # Green
        elif "LEFT" in text or "RIGHT" in text:
            self.lbl_status.setStyleSheet("font-size: 48px; font-weight: bold; color: #000000; background-color: #00FFFF;") # Cyan
        
        # Reset to Neutral after 1s
        QtCore.QTimer.singleShot(1000, lambda: self.reset_style())

    def reset_style(self):
         self.lbl_status.setStyleSheet("font-size: 48px; font-weight: bold; color: #00FF00; background-color: #222;")
         self.lbl_status.setText("MONITORING")

    def update_plot(self, chunk):
        clean_chunk = self.filter.process(chunk)
        n = clean_chunk.shape[1]
        self.data_buffer = np.roll(self.data_buffer, -n, axis=1)
        self.data_buffer[:, -n:] = clean_chunk
        
        offset = 200
        for i in range(4):
            self.curves[i].setData(self.data_buffer[i, ::2] + (i * offset))

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Dark Theme
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    p.setColor(QtGui.QPalette.ColorRole.WindowText, QtCore.Qt.GlobalColor.white)
    p.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(15, 15, 15))
    p.setColor(QtGui.QPalette.ColorRole.Text, QtCore.Qt.GlobalColor.white)
    app.setPalette(p)

    window = Dashboard()
    window.show()
    sys.exit(app.exec())