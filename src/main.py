import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons
import time
import numpy as np
import logging
from scipy.signal import butter, filtfilt, iirnotch, welch, coherence

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 250
BUFFER_DURATION_S = 2.0  
UPDATE_INTERVAL_S = 0.1
WARMUP_S = 3.0

# Frequency Bands
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45)
}

# Thresholds
BLINK_THRESHOLD_UV = 150.0
NOISE_THRESHOLD_UV = 500.0 # Rejection threshold

# --- DEPENDENCIES ---
try:
    from brainaccess.utils import acquisition
    from brainaccess.core.eeg_manager import EEGManager
except ImportError:
    logging.error("BrainAccess libraries missing.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(message)s')

class DataBuffer:
    def __init__(self, n_channels, sfreq, duration_s):
        self.buffer_len = int(sfreq * duration_s)
        self.buffer = np.zeros((n_channels, self.buffer_len))

    def update(self, new_data):
        n_new = new_data.shape[1]
        if n_new == 0: return
        if n_new >= self.buffer_len:
            self.buffer = new_data[:, -self.buffer_len:]
        else:
            self.buffer = np.roll(self.buffer, -n_new, axis=1)
            self.buffer[:, -n_new:] = new_data

    def get(self):
        return self.buffer.copy()

class SignalProcessor:
    def __init__(self, sfreq):
        self.sfreq = sfreq
        self.nyq = 0.5 * sfreq
        # Pre-compute filters
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, sfreq)
        self.b_bp, self.a_bp = butter(2, [1.0 / self.nyq, 45.0 / self.nyq], btype='band')

    def process(self, data, scale_factor=1.0):
        # 1. DC Offset
        data = data - np.mean(data, axis=1, keepdims=True)
        # 2. Notch
        data = filtfilt(self.b_notch, self.a_notch, data, axis=1)
        # 3. Bandpass
        data = filtfilt(self.b_bp, self.a_bp, data, axis=1)
        return data * scale_factor

    def get_psd_features(self, data):
        """Calculates band powers for a window"""
        nperseg = min(data.shape[1], self.sfreq)
        freqs, psd = welch(data, self.sfreq, nperseg=nperseg)
        
        features = {}
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                # Integrate power in band
                features[band] = np.trapz(psd[:, idx], freqs[idx], axis=1)
            else:
                features[band] = np.zeros(data.shape[0])
        return features

    def get_coherence(self, ch1, ch2):
        """Calculates avg coherence between two channels in Alpha band"""
        f, Cxy = coherence(ch1, ch2, fs=self.sfreq, nperseg=int(self.sfreq/2))
        # Focus on Alpha/Beta sync for resting state (8-30Hz)
        idx = np.logical_and(f >= 8, f <= 30)
        if np.any(idx):
            return np.mean(Cxy[idx])
        return 0.0

class EEGApp:
    def __init__(self):
        self.modes = [
            "Monitor (Raw)", 
            "Meditation Trainer", 
            "Keyboard/Mouse", 
            "Focus Tracker", 
            "Ad Testing", 
            "Driver Safety"
        ]
        self.current_mode = self.modes[0]
        self.scale_factor = 1.0
        self.scale_determined = False
        
        # Setup GUI
        self.init_gui()
        
        # Logic
        self.processor = SignalProcessor(SFREQ)
        self.buffer = DataBuffer(4, SFREQ, BUFFER_DURATION_S)
        
        # Metric History (for plotting trends)
        self.history = {
            "focus_ratio": np.zeros(50),
            "asymmetry": np.zeros(50),
            "alpha": np.zeros(50)
        }

    def init_gui(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 4, figure=self.fig)

        # 1. Raw EEG Plot (Top, spans all cols)
        self.ax_raw = self.fig.add_subplot(gs[0, :3])
        self.lines_raw = []
        colors = ['#00f0f0', '#f0a000', '#00ff00', '#ff4040']
        labels = ['Fp1', 'Fp2', 'O1', 'O2']
        for i in range(4):
            line, = self.ax_raw.plot([], [], color=colors[i], lw=1, label=labels[i])
            self.lines_raw.append(line)
        self.ax_raw.set_ylim(-300, 1100) # Waterfall offset
        self.ax_raw.set_title("Filtered EEG (uV)")
        self.ax_raw.legend(loc="upper right", fontsize="x-small")

        # 2. Mode Visualization (Bottom Left, spans 2 cols)
        self.ax_mode = self.fig.add_subplot(gs[1:, :3])
        
        # 3. Metrics Text (Bottom Right, small)
        self.ax_metrics = self.fig.add_subplot(gs[1, 3])
        self.ax_metrics.axis('off')
        self.text_metrics = self.ax_metrics.text(0, 0.5, "Initializing...", va='center', fontsize=9)

        # 4. Controls (Radio Buttons)
        self.ax_controls = self.fig.add_subplot(gs[2, 3])
        self.ax_controls.set_title("Select Mode")
        self.radio = RadioButtons(self.ax_controls, self.modes, activecolor='cyan')
        self.radio.on_clicked(self.change_mode)

        plt.tight_layout()
        plt.ion()

    def change_mode(self, label):
        self.current_mode = label
        self.ax_mode.clear()
        # Reset axes depending on mode
        if label == "Meditation Trainer":
            self.ax_mode.set_ylim(0, 50)
            self.ax_mode.set_title("Occipital Alpha Power (Relaxation)")
        elif label == "Focus Tracker":
            self.ax_mode.set_ylim(0, 3)
            self.ax_mode.set_title("Beta / Theta Ratio (Concentration)")
        elif label == "Ad Testing":
            self.ax_mode.set_ylim(-1, 1)
            self.ax_mode.set_title("Frontal Asymmetry (>0 Approach, <0 Withdrawal)")
        
    def determine_scaling(self, chunk):
        if np.mean(np.abs(chunk)) < 1.0:
            return 1e6
        return 1.0

    def update(self, raw_chunk):
        # Auto-scale logic
        if not self.scale_determined:
            self.scale_factor = self.determine_scaling(raw_chunk)
            self.scale_determined = True

        # Update Buffer
        self.buffer.update(raw_chunk)
        full_data = self.buffer.get()
        
        # Process
        clean_data = self.processor.process(full_data, self.scale_factor)
        
        # --- CALCULATE METRICS ---
        psd_pow = self.processor.get_psd_features(clean_data)
        
        # 1. Asymmetry (Fp1 vs Fp2 Alpha)
        # Using Alpha Asymmetry: ln(Right) - ln(Left) often used, or (R-L)/(R+L)
        # Here: (Fp1 - Fp2) / Total. 
        fp1_alpha = psd_pow["Alpha"][0]
        fp2_alpha = psd_pow["Alpha"][1]
        asymmetry = (fp1_alpha - fp2_alpha) / (fp1_alpha + fp2_alpha + 1e-6)

        # 2. Coherence (O1 vs O2)
        coherence_o1_o2 = self.processor.get_coherence(clean_data[2], clean_data[3])

        # 3. Focus Ratio (Beta / Theta) on Frontal
        beta_front = np.mean(psd_pow["Beta"][:2])
        theta_front = np.mean(psd_pow["Theta"][:2])
        focus_ratio = beta_front / (theta_front + 1e-6)

        # --- UPDATE VISUALIZATIONS ---
        
        # A. Raw Plot (Always Active)
        samples = np.arange(clean_data.shape[1])
        for i in range(4):
            # Waterfall offset: 250uV per channel
            self.lines_raw[i].set_data(samples, clean_data[i] + (i * 250))
        
        # B. Mode Specific Plot
        self.update_mode_plot(clean_data, psd_pow, asymmetry, focus_ratio, coherence_o1_o2)

        # C. Text Metrics (Right Panel)
        metrics_str = (
            f"--- GLOBAL STATS ---\n"
            f"Alpha Sync: {coherence_o1_o2:.2f}\n"
            f"Asym: {asymmetry:.2f}\n"
            f"Focus: {focus_ratio:.2f}\n\n"
            f"--- BAND POWER ---\n"
            f"Delta: {np.mean(psd_pow['Delta']):.1f}\n"
            f"Theta: {np.mean(psd_pow['Theta']):.1f}\n"
            f"Alpha: {np.mean(psd_pow['Alpha']):.1f}\n"
            f"Beta:  {np.mean(psd_pow['Beta']):.1f}\n"
            f"Gamma: {np.mean(psd_pow['Gamma']):.1f}"
        )
        self.text_metrics.set_text(metrics_str)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_mode_plot(self, clean_data, psd, asym, focus, coh):
        ax = self.ax_mode
        ax.clear() # Necessary for switching plot types

        if self.current_mode == "Meditation Trainer":
            # Bar Chart of O1/O2 Alpha
            o1_a = psd["Alpha"][2]
            o2_a = psd["Alpha"][3]
            bars = ax.bar(["O1 Alpha", "O2 Alpha"], [o1_a, o2_a], color=['green', 'lime'])
            ax.set_ylim(0, 50) # Adjust based on user baseline
            ax.set_title(f"MEDITATION: Synchronization {coh:.2f}")
            ax.axhline(10, color='white', linestyle='--', alpha=0.5, label='Target')

        elif self.current_mode == "Keyboard/Mouse":
            # Logic: Detect Blinks (Fp1/Fp2)
            fp1_ptp = np.ptp(clean_data[0, -100:]) # Last 400ms
            fp2_ptp = np.ptp(clean_data[1, -100:])
            
            status = "Waiting..."
            color = "gray"
            if max(fp1_ptp, fp2_ptp) > BLINK_THRESHOLD_UV:
                status = "CLICK DETECTED"
                color = "red"
            
            ax.text(0.5, 0.5, status, fontsize=30, ha='center', va='center', color=color)
            ax.axis('off')
            ax.set_title("Blink to Click")

        elif self.current_mode == "Focus Tracker":
            # Scrolling line graph of Focus Ratio
            self.history["focus_ratio"] = np.roll(self.history["focus_ratio"], -1)
            self.history["focus_ratio"][-1] = focus
            
            ax.plot(self.history["focus_ratio"], color='cyan', lw=2)
            ax.set_ylim(0, 3)
            ax.axhline(1.0, color='yellow', linestyle='--', label='Zone In')
            ax.set_title(f"Focus (Beta/Theta): {focus:.2f}")
            ax.grid(True, alpha=0.3)

        elif self.current_mode == "Ad Testing":
            # Scrolling line graph of Asymmetry
            self.history["asymmetry"] = np.roll(self.history["asymmetry"], -1)
            self.history["asymmetry"][-1] = asym
            
            ax.plot(self.history["asymmetry"], color='magenta', lw=2)
            ax.axhline(0, color='white')
            ax.set_ylim(-1, 1)
            ax.fill_between(range(50), 0, self.history["asymmetry"], alpha=0.3, color='magenta')
            ax.set_title("Emotional Valence (Asymmetry)")
            ax.set_ylabel("Withdrawal <---> Approach")

        elif self.current_mode == "Driver Safety":
            # Logic: Blink Duration + Theta
            # Simplified: Show Theta gauge
            theta_avg = np.mean(psd["Theta"])
            
            # Draw a gauge
            ax.barh(["Drowsiness (Theta)"], [theta_avg], color='orange')
            ax.set_xlim(0, 50)
            ax.set_title("Fatigue Monitor")
            if theta_avg > 20:
                ax.text(10, 0, "WARNING: FATIGUE", color='red', fontweight='bold')

        else: # Raw / Default
            ax.text(0.5, 0.5, "Select a specific mode\nto see analysis.", ha='center', color='gray')
            ax.axis('off')

def main():
    app = EEGApp()
    
    eeg = acquisition.EEG(mode="roll")
    cap = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
    
    try:
        with EEGManager() as mgr:
            logging.info(f"Connecting to {DEVICE_NAME}...")
            driver_buffer = int(BUFFER_DURATION_S * SFREQ)
            eeg.setup(mgr, device_name=DEVICE_NAME, cap=cap, sfreq=SFREQ, zeros_at_start=driver_buffer)
            eeg.start_acquisition()
            time.sleep(WARMUP_S)
            
            logging.info("Starting GUI Loop...")
            
            while True:
                start_t = time.time()
                
                mne_chunk = eeg.get_mne(tim=UPDATE_INTERVAL_S)
                if mne_chunk is not None and len(mne_chunk) > 0:
                    raw = mne_chunk.get_data()[:4, :]
                    app.update(raw)
                
                # Maintain loop timing
                elapsed = time.time() - start_t
                remaining = UPDATE_INTERVAL_S - elapsed
                if remaining > 0:
                    plt.pause(remaining)
                else:
                    plt.pause(0.001)

    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        eeg.stop_acquisition()
        plt.close()

if __name__ == "__main__":
    main()