
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons
import time
import numpy as np
import logging
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch, welch, coherence

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 250
BUFFER_DURATION_S = 2.0  
UPDATE_INTERVAL_S = 0.05 
WARMUP_S = 3.0

# --- THRESHOLDS ---
# Scientific thresholds based on EEG signal quality and safety requirements

BLINK_THRESHOLD_UV = 150.0  # Threshold for eye blink detection in hands-free interface

# NOISE THRESHOLDS - Quality control for reliable signal
# 1. High Frequency Noise (Muscle/Jaw)/Movement: 65uV
#    Elevated Gamma power indicates muscle activity or electrode movement
NOISE_GAMMA_THRESHOLD = 65.0  
# 2. Amplitude Noise (Movement/Loose Contact): 
#    If signal exceeds 800uV, it's likely hitting the rails or loose electrode
NOISE_AMP_THRESHOLD = 800.0   

# FAA SPECIFIC THRESHOLDS
# Delta power (1-4Hz) in Fp1/Fp2 is the primary signature of eye movements.
# If Frontal Delta > 75 uV^2/Hz (approx), it's likely an eye movement, not an emotion.
FAA_EOG_LIMIT = 75.0 


# DRIVER SAFETY THRESHOLDS - Scientifically validated values
THETA_FATIGUE_THRESHOLD = 25.0      # O1/O2 Theta power indicating drowsiness
ALPHA_MICROSLEEP_THRESHOLD = 50.0   # O1/O2 Alpha spike indicating closed eyes
FATIGUE_PERSISTENCE_FRAMES = 100    # 5 seconds of elevated Theta (scientific requirement)
MICROSLEEP_PERSISTENCE_FRAMES = 10  # 0.5 seconds of elevated Alpha


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
    pass

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
    """Scientific EEG signal processing pipeline
    
    Filtering suitable for real-time drowsiness detection:
    - 50Hz notch filter: Removes power line interference
    - 1-45Hz bandpass: Focuses on clinically relevant frequency bands
    - DC removal: Eliminates electrode offset drift
    """

    def __init__(self, sfreq):
        self.sfreq = sfreq
        self.nyq = 0.5 * sfreq
        # 50Hz notch filter for power line noise removal
        self.b_notch, self.a_notch = iirnotch(50.0, 30.0, sfreq)
        # 1-45Hz bandpass filter for EEG-relevant frequencies
        self.b_bp, self.a_bp = butter(2, [1.0 / self.nyq, 45.0 / self.nyq], btype='band')

    def process(self, data, scale_factor=1.0):
        """Apply scientific signal processing pipeline"""
        data = np.nan_to_num(data)
        # 1. DC Offset removal - eliminates electrode drift
        data = data - np.mean(data, axis=1, keepdims=True)
        # 2. 50Hz Notch filter - removes power line interference
        data = filtfilt(self.b_notch, self.a_notch, data, axis=1)
        # 3. Bandpass filter (1-45Hz) - focuses on EEG frequency range
        data = filtfilt(self.b_bp, self.a_bp, data, axis=1)
        return data * scale_factor
    
    def get_psd_features(self, data):

        """Extract scientific EEG band power features using Welch's method
        
        Uses Welch's periodogram for robust power spectral density estimation,
        suitable for real-time drowsiness monitoring applications.
        
        Returns power in each frequency band per channel:
        - Delta (1-4Hz): Deep sleep indicators
        - Theta (4-8Hz): Drowsiness/Driver Safety key band  
        - Alpha (8-13Hz): Relaxed awareness/Microsleep detection
        - Beta (13-30Hz): Active concentration/Focus tracking
        - Gamma (30-45Hz): High cognitive load/Noise indicator
        """
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
            "asymmetry": np.zeros(self.history_len),
            "attention": np.zeros(self.history_len) # New metric for Ad Testing
        }
        
        # --- NEW STATE VARIABLES ---
        self.focus_smoother = deque(maxlen=20) 
        self.fatigue_integrator = 0            
        self.blink_cooldown = 0.0              
        self.last_blink_time = 0.0
        self.microsleep_detector = 0           # Counter for eyes-closed detection
        
        self.is_noisy = False
        self.noise_reason = ""
        
        # Valid Asymmetry Memory
        self.last_valid_asymmetry = 0.0
        self.eog_artifact_active = False

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
        self.ax_raw.set_title("EEG (uV)")
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
        self.focus_smoother.clear()
        self.fatigue_integrator = 0
        self.microsleep_detector = 0
        self.ax_mode.clear()
        self.history["asymmetry"][:] = 0

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
            self.noise_reason = f"Connectors/Movement ({int(gamma_avg)}uV)"
        else:
            self.is_noisy = False
            self.noise_reason = ""

        # 2. Eye Movement / Saccade Detection (for Ad Testing)
        # High Delta in Frontal channels usually means blinking or scanning
        frontal_delta = np.mean(psd_pow["Delta"][:2])
        if frontal_delta > FAA_EOG_LIMIT:
            self.eog_artifact_active = True
        else:
            self.eog_artifact_active = False

        # 3. Calculate Stats
        
        # Asymmetry: Gated by EOG
        if not self.eog_artifact_active:
            fp1_a, fp2_a = psd_pow["Alpha"][0], psd_pow["Alpha"][1]
            current_asym = (fp1_a - fp2_a) / (fp1_a + fp2_a + 1e-6)
            self.last_valid_asymmetry = current_asym
        else:
            current_asym = self.last_valid_asymmetry # Hold value

        coherence_o1_o2 = self.processor.get_coherence(clean_data[2], clean_data[3])
        
        # Focus Calculation (Smoothed)
        beta_front = np.mean(psd_pow["Beta"][:2])
        theta_front = np.mean(psd_pow["Theta"][:2])
        raw_focus = beta_front / (theta_front + 1e-6)
        self.focus_smoother.append(raw_focus)
        focus_ratio = np.mean(self.focus_smoother) if self.focus_smoother else 0.0

        # Visual Attention (Occipital Beta / Alpha) - Less sensitive to EOG
        occ_beta = np.mean(psd_pow["Beta"][2:])
        occ_alpha = np.mean(psd_pow["Alpha"][2:])
        vis_attention = occ_beta / (occ_alpha + 1e-6)

        # Update Text Panel
        status_text = f"NOISE: {self.noise_reason}" if self.is_noisy else "SIGNAL: OK"
        
        metrics_str = (
            f"{status_text}\n"
            f"Gamma: {gamma_avg:.1f} uV (Limit: {NOISE_GAMMA_THRESHOLD})\n"
            f"EOG Activity: {'DETECTED' if self.eog_artifact_active else 'Low'}\n"
            f"MaxAmp: {max_amp:.0f} uV\n"
            f"----------------------\n"
            f"STATS:\n"
            f"Valence (FAA): {current_asym:.2f}\n"
            f"Vis Attention: {vis_attention:.2f}\n"
            f"Alpha Sync:  {coherence_o1_o2:.2f}\n"
            f"Focus Ratio:   {focus_ratio:.2f}\n"
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
        self.update_mode_plot(clean_data, psd_pow, current_asym, focus_ratio, coherence_o1_o2, vis_attention)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update_mode_plot(self, clean_data, psd, asym, focus, coh, vis_attn):
        ax = self.ax_mode
        ax.clear() 
        ax.set_facecolor('black')

        # --- DRAW PLOTS ---
        
        if self.current_mode == "Meditation Trainer":
            o1_a = psd["Alpha"][2]
            o2_a = psd["Alpha"][3]
            ax.bar(["O1 Alpha", "O2 Alpha"], [o1_a, o2_a], color=['green', 'lime'])
            ax.set_ylim(0, 100) 
            ax.set_title(f"MEDITATION (Alpha Power) - Sync: {coh:.2f}")

        elif self.current_mode == "Keyboard/Mouse":
            fp1_ptp = np.ptp(clean_data[0, -100:]) 
            fp2_ptp = np.ptp(clean_data[1, -100:])

            current_time = time.time()

            status = "Ready"
            color = "gray"

            # # Cooldown logic (0.5s) to prevent double firing
            # if (current_time - self.blink_cooldown) > 0.5:
            #     if current_amp > BLINK_THRESHOLD_UV:
            #         status = "CLICK DETECTED"
            #         color = "#00ffff" # Cyan
            #         self.blink_cooldown = current_time # Reset cooldown
            #         # Here you would emit the actual mouse click event
            # else:
            #     status = "..." # In cooldown
            #     color = "#444444"
            
            if max(fp1_ptp, fp2_ptp) > BLINK_THRESHOLD_UV:
                status = "CLICK!"
                color = "cyan"

            ax.text(0.5, 0.5, status, fontsize=30, ha='center', va='center', color=color, weight='bold')
            ax.axis('off')
            ax.set_title("Hands-Free Interface (Blink to Click)")

        elif self.current_mode == "Focus Tracker":
            self.history["focus_ratio"] = np.roll(self.history["focus_ratio"], -1)
            self.history["focus_ratio"][-1] = focus
            
            ax.plot(self.history["focus_ratio"], color='cyan', lw=2)
            ax.set_ylim(0, 3)
            ax.set_xlim(0, self.history_len)

            # Draw threshold zone
            ax.axhline(1.0, color='yellow', linestyle='--', label='Flow State')
            ax.fill_between(range(self.history_len), 1.0, 3.0, color='yellow', alpha=0.1)
            
            ax.set_title(f"Focus Level (Beta/Theta): {focus:.2f}")
            ax.grid(True, alpha=0.3)

        elif self.current_mode == "Ad Testing":
            self.history["asymmetry"] = np.roll(self.history["asymmetry"], -1)
            self.history["asymmetry"][-1] = asym
            
            self.history["attention"] = np.roll(self.history["attention"], -1)
            self.history["attention"][-1] = vis_attn

            # Plot Valence (Magenta)
            # If artifact is active, color the line Gray to indicate "Paused/Unreliable"
            line_color = 'gray' if self.eog_artifact_active else 'magenta'
            ax.plot(self.history["asymmetry"], color=line_color, lw=2, label="Valence (FAA)")
            
            # Plot Attention (Green) - Secondary axis scale roughly matches (-1 to 1 vs 0 to 2)
            ax.plot((self.history["attention"] - 1.0), color='lime', lw=1, alpha=0.7, label="Visual Attention (Offset)")

            ax.axhline(0, color='white')
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, self.history_len)
            
            if not self.eog_artifact_active:
                ax.fill_between(range(self.history_len), 0, self.history["asymmetry"], alpha=0.3, color='magenta')
            
            status = "VALID" if not self.eog_artifact_active else "EOG DETECTED (Holding)"
            ax.set_title(f"Ad Response: {status}")
            ax.legend(loc="upper left", fontsize="x-small")

        elif self.current_mode == "Driver Safety":
            # SCIENTIFIC APPROACH: Focus specifically on O1/O2 (Visual Cortex) for drowsiness detection
            # Drowsiness causes Alpha->Theta shift in occipital regions, less affected by facial movements
            
            # Extract O1/O2 specific measurements (channels 2,3)
            o1_theta = psd["Theta"][2]  # O1 Theta power
            o2_theta = psd["Theta"][3]  # O2 Theta power
            o1_alpha = psd["Alpha"][2]  # O1 Alpha power
            o2_alpha = psd["Alpha"][3]  # O2 Alpha power
            
            theta_avg_o = np.mean([o1_theta, o2_theta])  # Average O1/O2 Theta
            alpha_avg_o = np.mean([o1_alpha, o2_alpha])  # Average O1/O2 Alpha
            
            # MICROSLEEP DETECTION: Massive Alpha spike indicates closed eyes (microsleep)
            if alpha_avg_o > ALPHA_MICROSLEEP_THRESHOLD:
                self.microsleep_detector += 1
            else:
                self.microsleep_detector = max(0, self.microsleep_detector - 2)  # Faster decay
            
            # FATIGUE DETECTION: Elevated Theta in O1/O2 indicates drowsiness
            if theta_avg_o > THETA_FATIGUE_THRESHOLD:
                self.fatigue_integrator += 1
            else:
                self.fatigue_integrator = max(0, self.fatigue_integrator - 1)
            
            # VISUALIZATION: Bar plot showing Theta levels
            ax.barh(["O1/O2 Drowsiness"], [theta_avg_o], color='orange')
            ax.set_xlim(0, 60)
            ax.set_title("Alert-0: Driver Fatigue Monitor (O1/O2 Theta)")
            
            # Display current Theta value
            ax.text(min(theta_avg_o, 55), 0, f"{theta_avg_o:.1f}", va='center', ha='left', 
                   color='white', fontweight='bold')
            
            # SCIENTIFIC TIMING: 5-second persistence requirement for fatigue detection
            if self.fatigue_integrator > FATIGUE_PERSISTENCE_FRAMES:
                ax.text(30, 0, "⚠️ FATIGUE DETECTED", color='red', fontweight='bold', ha='center', 
                       bbox=dict(facecolor='black', alpha=0.9, edgecolor='red', boxstyle='round,pad=1'))
            
            # Microsleep warning (Alpha spike = eyes closed)
            if self.microsleep_detector > MICROSLEEP_PERSISTENCE_FRAMES:
                ax.text(30, -0.3, "👁️ EYES CLOSED!", color='yellow', fontweight='bold', ha='center',
                       bbox=dict(facecolor='black', alpha=0.9, edgecolor='yellow', boxstyle='round,pad=1'))
            
            # Scientific status display
            status_text = "RESTED" if theta_avg_o < 15 else "MONITORING" if theta_avg_o < 25 else "DROWSY"
            ax.text(0.02, 0.98, f"Status: {status_text}", transform=ax.transAxes, 
                   fontsize=10, fontweight='bold', va='top')
            
            # Display Alpha level for microsleep monitoring
            ax.text(0.02, 0.90, f"O1/O2 Alpha: {alpha_avg_o:.1f}", transform=ax.transAxes,
                   fontsize=9, va='top')

        else: 
            ax.text(0.5, 0.5, "Monitor Mode\n(See raw data above)", ha='center', color='gray')
            ax.axis('off')

        # --- IF NOISY, ADD OVERLAY (Don't block, just warn) ---
        if self.is_noisy:
            ax.patch.set_facecolor('red')
            ax.patch.set_alpha(0.2) 
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