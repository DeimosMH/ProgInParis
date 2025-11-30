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
UPDATE_INTERVAL_S = 0.05 
WARMUP_S = 3.0

# --- THRESHOLDS ---
BLINK_THRESHOLD_UV = 150.0

# REVISED NOISE THRESHOLDS
# 1. High Frequency Noise (Muscle/Jaw): 60uV
NOISE_GAMMA_THRESHOLD = 60.0  
# 2. Amplitude Noise (Movement/Loose Contact): 
# If signal exceeds 800uV, it's likely hitting the rails or loose.
NOISE_AMP_THRESHOLD = 800.0   

# Frequency Bands
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45)
}

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
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, sfreq)
        self.b_bp, self.a_bp = butter(2, [1.0 / self.nyq, 45.0 / self.nyq], btype='band')

    def process(self, data, scale_factor=1.0):
        data = np.nan_to_num(data)
        # 1. DC Offset
        data = data - np.mean(data, axis=1, keepdims=True)
        # 2. Notch
        data = filtfilt(self.b_notch, self.a_notch, data, axis=1)
        # 3. Bandpass
        data = filtfilt(self.b_bp, self.a_bp, data, axis=1)
        return data * scale_factor

    def get_psd_features(self, data):
        nperseg = min(data.shape[1], self.sfreq)
        freqs, psd = welch(data, self.sfreq, nperseg=nperseg)
        psd = np.nan_to_num(psd)
        
        features = {}
        for band, (low, high) in BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                features[band] = np.trapz(psd[:, idx], freqs[idx], axis=1)
            else:
                features[band] = np.zeros(data.shape[0])
        return features

    def get_coherence(self, ch1, ch2):
        try:
            f, Cxy = coherence(ch1, ch2, fs=self.sfreq, nperseg=int(self.sfreq/2))
            Cxy = np.nan_to_num(Cxy)
            idx = np.logical_and(f >= 8, f <= 30) # Alpha/Beta sync
            if np.any(idx):
                return np.mean(Cxy[idx])
        except:
            return 0.0
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
        self.buffer_len_samples = int(BUFFER_DURATION_S * SFREQ)

        self.processor = SignalProcessor(SFREQ)
        self.buffer = DataBuffer(4, SFREQ, BUFFER_DURATION_S)
        
        self.history_len = 100
        self.history = {
            "focus_ratio": np.zeros(self.history_len),
            "asymmetry": np.zeros(self.history_len)
        }
        
        self.is_noisy = False
        self.noise_reason = ""

        self.init_gui()

    def init_gui(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(13, 8), constrained_layout=True)
        gs = gridspec.GridSpec(3, 5, figure=self.fig, width_ratios=[1, 1, 1, 1, 1.2])

        # 1. Raw EEG Plot
        self.ax_raw = self.fig.add_subplot(gs[0, :4])
        self.lines_raw = []
        colors = ['#00f0f0', '#f0a000', '#00ff00', '#ff4040']
        labels = ['Fp1', 'Fp2', 'O1', 'O2']
        
        x_init = np.arange(self.buffer_len_samples)
        for i in range(4):
            line, = self.ax_raw.plot(x_init, np.zeros(self.buffer_len_samples), 
                                     color=colors[i], lw=1, label=labels[i])
            self.lines_raw.append(line)
            
        self.ax_raw.set_xlim(0, self.buffer_len_samples)
        self.ax_raw.set_ylim(-300, 1100)
        self.ax_raw.set_title("Filtered EEG (uV)")
        self.ax_raw.legend(loc="upper right", fontsize="x-small", framealpha=0.3)
        self.ax_raw.tick_params(left=False, labelleft=True)

        # 2. Mode Visualization
        self.ax_mode = self.fig.add_subplot(gs[1:, :4])
        
        # 3. Metrics Text
        self.ax_metrics = self.fig.add_subplot(gs[1, 4])
        self.ax_metrics.axis('off')
        self.text_metrics = self.ax_metrics.text(0.05, 0.95, "Initializing...", 
                                                 va='top', ha='left', fontsize=8, family='monospace')

        # 4. Controls
        self.ax_controls = self.fig.add_subplot(gs[2, 4])
        self.ax_controls.set_title("Select Mode", fontsize=10)
        self.ax_controls.set_facecolor('#1e1e1e')
        self.radio = RadioButtons(self.ax_controls, self.modes, activecolor='cyan')
        
        for label in self.radio.labels:
            label.set_fontsize(9)
            label.set_color("white")
            
        self.radio.on_clicked(self.change_mode)

    def change_mode(self, label):
        self.current_mode = label
        self.ax_mode.clear()

    def determine_scaling(self, chunk):
        mean_val = np.mean(np.abs(chunk))
        if mean_val < 1.0 and mean_val > 0:
            return 1e6
        return 1.0

    def update(self, raw_chunk):
        if not self.scale_determined:
            self.scale_factor = self.determine_scaling(raw_chunk)
            self.scale_determined = True

        self.buffer.update(raw_chunk)
        full_data = self.buffer.get()
        
        # Process
        clean_data = self.processor.process(full_data, self.scale_factor)
        
        # --- UPDATE RAW PLOT ---
        samples = np.arange(clean_data.shape[1])
        for i in range(4):
            self.lines_raw[i].set_data(samples, clean_data[i] + (i * 250))
        self.ax_raw.set_xlim(0, len(samples))

        # --- METRICS & NOISE CHECK ---
        psd_pow = self.processor.get_psd_features(clean_data)
        
        # 1. Gamma Check
        gamma_avg = np.mean(psd_pow["Gamma"])
        
        # 2. Amplitude Check (Absolute max value)
        max_amp = np.max(np.abs(clean_data))
        
        # Logic: Noise if Gamma is extreme OR Amplitude is railing
        if max_amp > NOISE_AMP_THRESHOLD:
            self.is_noisy = True
            self.noise_reason = f"Movement ({int(max_amp)}uV)"
        elif gamma_avg > NOISE_GAMMA_THRESHOLD:
            self.is_noisy = True
            self.noise_reason = f"Muscle/Jaw ({int(gamma_avg)}uV)"
        else:
            self.is_noisy = False
            self.noise_reason = ""

        # Stats
        fp1_a, fp2_a = psd_pow["Alpha"][0], psd_pow["Alpha"][1]
        asymmetry = (fp1_a - fp2_a) / (fp1_a + fp2_a + 1e-6)
        coherence_o1_o2 = self.processor.get_coherence(clean_data[2], clean_data[3])
        beta_front = np.mean(psd_pow["Beta"][:2])
        theta_front = np.mean(psd_pow["Theta"][:2])
        focus_ratio = beta_front / (theta_front + 1e-6)

        # Update Text Panel
        status_text = f"NOISE: {self.noise_reason}" if self.is_noisy else "SIGNAL: OK"
        
        metrics_str = (
            f"{status_text}\n"
            f"Gamma: {gamma_avg:.1f} uV (Limit: {NOISE_GAMMA_THRESHOLD})\n"
            f"MaxAmp: {max_amp:.0f} uV\n"
            f"----------------------\n"
            f"STATS:\n"
            f"Alpha Sync:  {coherence_o1_o2:.2f}\n"
            f"Asymmetry:   {asymmetry:.2f}\n"
            f"Focus Ratio: {focus_ratio:.2f}\n"
            f"----------------------\n"
            f"BAND POWER (Avg):\n"
            f"Delta: {np.mean(psd_pow['Delta']):.1f}\n"
            f"Theta: {np.mean(psd_pow['Theta']):.1f}\n"
            f"Alpha: {np.mean(psd_pow['Alpha']):.1f}\n"
            f"Beta:  {np.mean(psd_pow['Beta']):.1f}\n"
        )
        self.text_metrics.set_text(metrics_str)
        self.text_metrics.set_color('#ff5555' if self.is_noisy else 'white')

        # --- UPDATE MODE PLOT ---
        self.update_mode_plot(clean_data, psd_pow, asymmetry, focus_ratio, coherence_o1_o2)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_mode_plot(self, clean_data, psd, asym, focus, coh):
        ax = self.ax_mode
        ax.clear() 
        ax.set_facecolor('black')

        # --- DRAW PLOTS FIRST (So we can see data even if noisy) ---
        
        if self.current_mode == "Meditation Trainer":
            o1_a = psd["Alpha"][2]
            o2_a = psd["Alpha"][3]
            ax.bar(["O1 Alpha", "O2 Alpha"], [o1_a, o2_a], color=['green', 'lime'])
            ax.set_ylim(0, 100) 
            ax.set_title(f"MEDITATION (Alpha Power) - Sync: {coh:.2f}")

        elif self.current_mode == "Keyboard/Mouse":
            fp1_ptp = np.ptp(clean_data[0, -100:]) 
            fp2_ptp = np.ptp(clean_data[1, -100:])
            status = "Waiting..."
            color = "gray"
            if max(fp1_ptp, fp2_ptp) > BLINK_THRESHOLD_UV:
                status = "CLICK!"
                color = "cyan"
            ax.text(0.5, 0.5, status, fontsize=30, ha='center', va='center', color=color)
            ax.axis('off')
            ax.set_title("Blink to Click")

        elif self.current_mode == "Focus Tracker":
            self.history["focus_ratio"] = np.roll(self.history["focus_ratio"], -1)
            self.history["focus_ratio"][-1] = focus
            ax.plot(self.history["focus_ratio"], color='cyan', lw=2)
            ax.set_ylim(0, 3)
            ax.set_xlim(0, self.history_len)
            ax.axhline(1.0, color='yellow', linestyle='--', label='Zone In')
            ax.set_title(f"Focus (Beta/Theta): {focus:.2f}")
            ax.grid(True, alpha=0.3)

        elif self.current_mode == "Ad Testing":
            self.history["asymmetry"] = np.roll(self.history["asymmetry"], -1)
            self.history["asymmetry"][-1] = asym
            ax.plot(self.history["asymmetry"], color='magenta', lw=2)
            ax.axhline(0, color='white')
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, self.history_len)
            ax.fill_between(range(self.history_len), 0, self.history["asymmetry"], alpha=0.3, color='magenta')
            ax.set_title("Emotional Valence (Asymmetry)")

        elif self.current_mode == "Driver Safety":
            theta_avg = np.mean(psd["Theta"])
            ax.barh(["Drowsiness"], [theta_avg], color='orange')
            ax.set_xlim(0, 50)
            ax.set_title("Fatigue Monitor (Theta Power)")
            ax.text(min(theta_avg, 45), 0, f"{theta_avg:.1f}", va='center', ha='left', color='white', fontweight='bold')
            if theta_avg > 25:
                ax.text(25, 0, "WARNING: FATIGUE", color='red', fontweight='bold', ha='center', bbox=dict(facecolor='black', alpha=0.7))

        else: 
            ax.text(0.5, 0.5, "Monitor Mode\n(See raw data above)", ha='center', color='gray')
            ax.axis('off')

        # --- IF NOISY, ADD OVERLAY (Don't block, just warn) ---
        if self.is_noisy:
            # Create a semi-transparent overlay
            ax.patch.set_facecolor('red')
            ax.patch.set_alpha(0.2) # 20% red tint
            
            # Warning text (Top center, not blocking middle)
            ax.text(0.5, 0.9, f"⚠️ HIGH NOISE: {self.noise_reason}", 
                    transform=ax.transAxes,
                    fontsize=12, color='red', ha='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.8, edgecolor='red'))

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
                    if raw.size > 0:
                        app.update(raw)
                
                plt.pause(0.001)

    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        try: eeg.stop_acquisition()
        except: pass
        plt.close()

if __name__ == "__main__":
    main()