import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import logging

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 250
CYCLE_DURATION_S = 0.1   
BUFFER_DURATION_S = 2.0  
WARMUP_S = 4.0           

# Frequencies
BP_LOW = 1.0
BP_HIGH = 30.0
NOTCH_FREQ = 50.0

# Detection Thresholds
BLINK_THRESHOLD_UV = 150.0 
ALPHA_THRESHOLD_REL = 0.4 

# --- DEPENDENCIES ---
try:
    from scipy.signal import butter, filtfilt, iirnotch, welch
    from brainaccess.utils import acquisition
    from brainaccess.core.eeg_manager import EEGManager
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("Required libraries missing.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DataBuffer:
    def __init__(self, n_channels, sfreq, duration_s):
        self.sfreq = sfreq
        self.buffer_len = int(sfreq * duration_s)
        self.n_channels = n_channels
        self.buffer = np.zeros((n_channels, self.buffer_len))

    def update(self, new_data):
        n_new = new_data.shape[1]
        if n_new == 0: return
        
        if n_new >= self.buffer_len:
            self.buffer = new_data[:, -self.buffer_len:]
        else:
            self.buffer = np.roll(self.buffer, -n_new, axis=1)
            self.buffer[:, -n_new:] = new_data

    def get_data(self):
        return self.buffer.copy()

def apply_robust_filters(data, sfreq):
    # 1. Remove DC
    data = data - np.mean(data, axis=1, keepdims=True)
    # 2. Notch (50Hz)
    b_notch, a_notch = iirnotch(NOTCH_FREQ, 30.0, sfreq)
    data = filtfilt(b_notch, a_notch, data, axis=1)
    # 3. Bandpass (1-30Hz)
    nyquist = 0.5 * sfreq
    b_bp, a_bp = butter(2, [BP_LOW / nyquist, BP_HIGH / nyquist], btype='band')
    data = filtfilt(b_bp, a_bp, data, axis=1)
    return data

def get_band_power(data, sfreq, band_limits):
    n_samples = data.shape[-1]
    nperseg = min(n_samples, sfreq)
    freqs, psd = welch(data, sfreq, nperseg=nperseg)
    
    freq_res = freqs[1] - freqs[0]
    total_power = np.trapz(psd, dx=freq_res) + 1e-10
    idx_band = np.logical_and(freqs >= band_limits[0], freqs <= band_limits[1])
    band_power = np.trapz(psd[..., idx_band], dx=freq_res)
    return band_power / total_power

def detect_artifacts_and_waves(clean_data_uv, sfreq):
    """
    Expects data ALREADY scaled to Microvolts (uV).
    """
    n_samples_check = int(0.5 * sfreq) 
    recent_data = clean_data_uv[:, -n_samples_check:]
    
    if recent_data.shape[1] < 10: return "WAITING"

    # Blink (Fp1=0, Fp2=1)
    # Data is already uV, do NOT multiply by 1e6 again
    fp1_ptp = np.ptp(recent_data[0]) 
    fp2_ptp = np.ptp(recent_data[1]) 

    # Alpha (O1=2, O2=3)
    o1_alpha_rel = get_band_power(clean_data_uv[2], sfreq, [8, 12])
    o2_alpha_rel = get_band_power(clean_data_uv[3], sfreq, [8, 12])
    avg_alpha = (o1_alpha_rel + o2_alpha_rel) / 2

    if fp1_ptp > BLINK_THRESHOLD_UV or fp2_ptp > BLINK_THRESHOLD_UV:
        return f"BLINK ({int(max(fp1_ptp, fp2_ptp))}uV)"
    
    if avg_alpha > ALPHA_THRESHOLD_REL:
        return f"ALPHA ({avg_alpha:.2f})"
        
    return "NONE"

def update_plot(ax, clean_data_uv, ch_names):
    ax.clear()
    samples = np.arange(clean_data_uv.shape[1])
    for i in range(clean_data_uv.shape[0]):
        # Data is already uV. Add 200uV offset per channel for clarity.
        signal_uv = clean_data_uv[i] + (i * 200) 
        ax.plot(samples, signal_uv, label=ch_names[i], linewidth=1)
    
    ax.set_ylim(-200, 1000)
    ax.set_title("Filtered EEG (uV)")
    ax.legend(loc="upper right", fontsize="small")
    plt.pause(0.001)

def determine_scaling(data_chunk):
    """
    Heuristic: Standard EEG is ~50uV.
    If Mean > 1.0, it's likely already uV (or counts). Scale = 1.0
    If Mean < 1.0 (e.g. 0.00005), it's Volts. Scale = 1e6.
    """
    mean_val = np.mean(np.abs(data_chunk))
    if mean_val > 1.0:
        return 1.0
    return 1e6

def main():
    eeg = acquisition.EEG(mode="roll")
    
    try:
        matplotlib.use('TkAgg')
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_enabled = True
    except Exception:
        logging.warning("Plotting disabled.")
        plot_enabled = False
    
    last_blink_time = 0
    cap = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
    
    # Auto-scaling state
    scale_factor = 1.0
    scale_determined = False
    
    try:
        with EEGManager() as mgr:
            logging.info(f"Connecting to {DEVICE_NAME}...")
            # Ensure buffer size is adequate in driver
            driver_buffer = int(BUFFER_DURATION_S * SFREQ)
            eeg.setup(mgr, device_name=DEVICE_NAME, cap=cap, sfreq=SFREQ, zeros_at_start=driver_buffer)

            eeg.start_acquisition()
            logging.info("Acquisition started. Warming up...")
            time.sleep(2.0)

            data_buffer = DataBuffer(n_channels=4, sfreq=SFREQ, duration_s=BUFFER_DURATION_S)
            start_time = time.time()
            
            while True:
                loop_start = time.time()
                
                # Fetch Data
                mne_chunk = eeg.get_mne(tim=CYCLE_DURATION_S)
                if mne_chunk is None or len(mne_chunk) == 0:
                    time.sleep(0.01)
                    continue
                
                # Get raw array (Channels x Samples)
                raw_chunk = mne_chunk.get_data()[:4, :].copy()
                
                # Safety check for empty data
                if raw_chunk.size == 0: continue

                # --- AUTO SCALING ---
                # Determine units on the first valid chunk
                if not scale_determined:
                    scale_factor = determine_scaling(raw_chunk)
                    logging.info(f"Auto-detected Scale Factor: {scale_factor} (Mean amp: {np.mean(np.abs(raw_chunk)):.4f})")
                    scale_determined = True

                # Update Buffer
                data_buffer.update(raw_chunk)
                
                if time.time() - start_time > WARMUP_S:
                    # Get 2s window
                    full_window = data_buffer.get_data()
                    
                    try:
                        # 1. Filter
                        clean_window = apply_robust_filters(full_window, SFREQ)
                        
                        # 2. Scale to Microvolts (uV)
                        clean_window_uv = clean_window * scale_factor
                        
                        # 3. Detect
                        decision = detect_artifacts_and_waves(clean_window_uv, SFREQ)
                        
                        if "BLINK" in decision:
                            if (time.time() - last_blink_time) > 1.0:
                                logging.info(f"--- DETECTED: {decision} ---")
                                last_blink_time = time.time()
                        elif "ALPHA" in decision:
                            logging.info(f"State: {decision}")

                        # 4. Plot
                        if plot_enabled:
                            update_plot(ax, clean_window_uv, ["Fp1", "Fp2", "O1", "O2"])
                            
                    except Exception as e:
                        logging.error(f"Processing error: {e}")

                elapsed = time.time() - loop_start
                sleep_t = CYCLE_DURATION_S - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Fatal Error: {e}")
    finally:
        try: eeg.stop_acquisition()
        except: pass
        plt.close()

if __name__ == "__main__":
    main()