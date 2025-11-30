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
BP_HIGH = 45.0  # Increased slightly to allow EMG detection before filtering cuts it off completely
NOTCH_FREQ = 50.0

# --- DETECTION THRESHOLDS ---
# 1. Error Rejection (Muscle/Movement Noise)
# Threshold for power in 30Hz+ band. If exceeded, ignore window.
EMG_THRESHOLD_REL = 30.0 

# 2. Eye Events (Frontal Channels)
BLINK_THRESHOLD_UV = 150.0      # Hard Blinks are usually > 150uV
EYE_MOVE_THRESHOLD_UV = 70.0    # Eye movements are usually 70-150uV

# 3. Brain States (Occipital Channels)
ALPHA_THRESHOLD_REL = 1.5       # Relative PSD power

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
    # 3. Bandpass (1-45Hz)
    nyquist = 0.5 * sfreq
    b_bp, a_bp = butter(2, [BP_LOW / nyquist, BP_HIGH / nyquist], btype='band')
    data = filtfilt(b_bp, a_bp, data, axis=1)
    return data

def get_band_power(data, sfreq, band_limits):
    """
    Returns absolute power in a specific band.
    """
    n_samples = data.shape[-1]
    nperseg = min(n_samples, sfreq) # 1 sec window for Welch
    freqs, psd = welch(data, sfreq, nperseg=nperseg)
    
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band_limits[0], freqs <= band_limits[1])
    band_power = np.trapz(psd[..., idx_band], dx=freq_res)
    return band_power

def detect_artifacts_and_waves(clean_data_uv, sfreq):
    """
    Priority:
    1. Detect NOISE (Head movement, Jaw clench/Reading) -> IGNORE
    2. Detect BLINK (High amplitude)
    3. Detect EYE MOVEMENT (Medium amplitude, shifts)
    4. Detect ALPHA (Resting state)
    """
    n_samples_check = int(0.5 * sfreq) 
    # Frontal channels (Fp1, Fp2) usually indices 0 and 1
    # Occipital channels (O1, O2) usually indices 2 and 3
    
    # Analyze only the most recent 0.5s for events, but use full buffer for frequency analysis
    recent_frontal = clean_data_uv[0:2, -n_samples_check:]
    full_frontal = clean_data_uv[0:2, :]
    full_occipital = clean_data_uv[2:4, :]
    
    if recent_frontal.shape[1] < 10: return "WAITING"

    # --- 1. NOISE / ERROR DETECTION ---
    # The 'err' images (Head movement, Reading) generate high frequency noise (EMG).
    # We check for high power in Gamma band (30-45Hz in filtered data, or higher if raw).
    # Since we low-pass at 45Hz, we check the upper edge.
    emg_power = get_band_power(full_frontal, sfreq, [30, 45])
    # Average across Fp1/Fp2
    avg_emg = np.mean(emg_power)
    
    # Heuristic: If high frequency power is massive, it's not a clean blink/move, it's noise.
    if avg_emg > EMG_THRESHOLD_REL:
        return f"WAITING (High Noise: {avg_emg:.1f})"

    # --- 2. BLINK DETECTION ---
    # Blinks are characterized by very high amplitude peaks in Frontal channels.
    fp1_ptp = np.ptp(recent_frontal[0]) 
    fp2_ptp = np.ptp(recent_frontal[1]) 
    max_amp = max(fp1_ptp, fp2_ptp)

    if max_amp > BLINK_THRESHOLD_UV:
        return f"BLINK (Amp: {int(max_amp)}uV)"

    # --- 3. EYE MOVEMENT DETECTION ---
    # Look for "Left to Right" or "Up Down" (non-blink).
    # These are usually significant potential shifts but often lower amplitude than a hard blink,
    # or wider. For this simple logic, we use an amplitude window below Blink but above EEG.
    if max_amp > EYE_MOVE_THRESHOLD_UV:
        return f"EYE MOVEMENT (Amp: {int(max_amp)}uV)"
    
    # --- 4. ALPHA DETECTION ---
    # Only check if eyes are relatively still
    o1_alpha = get_band_power(full_occipital[0], sfreq, [8, 12])
    o2_alpha = get_band_power(full_occipital[1], sfreq, [8, 12])
    
    # Normalize by Delta/Theta to get relative alpha (prevents 1/f detection)
    o1_low = get_band_power(full_occipital[0], sfreq, [1, 5]) + 1e-10
    o2_low = get_band_power(full_occipital[1], sfreq, [1, 5]) + 1e-10
    
    ratio = ((o1_alpha/o1_low) + (o2_alpha/o2_low)) / 2

    if ratio > ALPHA_THRESHOLD_REL:
        return f"ALPHA WAVE (Ratio: {ratio:.2f})"
        
    return "NONE"

def update_plot(ax, clean_data_uv, ch_names, detection_text):
    ax.clear()
    samples = np.arange(clean_data_uv.shape[1])
    for i in range(clean_data_uv.shape[0]):
        # Add offset per channel for waterfall plot
        signal_uv = clean_data_uv[i] + (i * 250) 
        ax.plot(samples, signal_uv, label=ch_names[i], linewidth=1)
    
    ax.set_ylim(-300, 1300)
    ax.set_title(f"EEG Monitor - State: {detection_text}")
    ax.legend(loc="upper right", fontsize="small")
    plt.pause(0.001)

def determine_scaling(data_chunk):
    mean_val = np.mean(np.abs(data_chunk))
    if mean_val > 1.0:
        return 1.0
    return 1e6

def main():
    eeg = acquisition.EEG(mode="roll")
    
    try:
        matplotlib.use('TkAgg')
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_enabled = True
    except Exception:
        logging.warning("Plotting disabled.")
        plot_enabled = False
    
    last_event_time = 0
    cap = {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"}
    
    scale_factor = 1.0
    scale_determined = False
    
    current_decision = "Init"

    try:
        with EEGManager() as mgr:
            logging.info(f"Connecting to {DEVICE_NAME}...")
            driver_buffer = int(BUFFER_DURATION_S * SFREQ)
            eeg.setup(mgr, device_name=DEVICE_NAME, cap=cap, sfreq=SFREQ, zeros_at_start=driver_buffer)

            eeg.start_acquisition()
            logging.info("Acquisition started. Stabilization...")
            time.sleep(2.0)

            data_buffer = DataBuffer(n_channels=4, sfreq=SFREQ, duration_s=BUFFER_DURATION_S)
            start_time = time.time()
            
            while True:
                loop_start = time.time()
                
                mne_chunk = eeg.get_mne(tim=CYCLE_DURATION_S)
                if mne_chunk is None or len(mne_chunk) == 0:
                    time.sleep(0.01)
                    continue
                
                raw_chunk = mne_chunk.get_data()[:4, :].copy()
                if raw_chunk.size == 0: continue

                if not scale_determined:
                    scale_factor = determine_scaling(raw_chunk)
                    logging.info(f"Scale Factor: {scale_factor}")
                    scale_determined = True

                data_buffer.update(raw_chunk)
                
                if time.time() - start_time > WARMUP_S:
                    full_window = data_buffer.get_data()
                    
                    try:
                        # 1. Filter
                        clean_window = apply_robust_filters(full_window, SFREQ)
                        clean_window_uv = clean_window * scale_factor
                        
                        # 2. Detect with Error Ignoring Logic
                        decision = detect_artifacts_and_waves(clean_window_uv, SFREQ)
                        
                        # Logic to prevent console spam
                        if "BLINK" in decision or "EYE" in decision:
                            if (time.time() - last_event_time) > 0.8:
                                logging.info(f">>> {decision}")
                                current_decision = decision
                                last_event_time = time.time()
                        elif "WAITING" in decision:
                            # Do not log standard waiting, only update plot state
                            current_decision = decision
                        elif "ALPHA" in decision:
                            current_decision = decision
                            if time.time() - last_event_time > 1.0:
                                logging.info(f"State: {decision}")
                                last_event_time = time.time()
                        else:
                            current_decision = "Monitoring..."

                        # 3. Plot
                        if plot_enabled:
                            update_plot(ax, clean_window_uv, ["Fp1", "Fp2", "O1", "O2"], current_decision)
                            
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