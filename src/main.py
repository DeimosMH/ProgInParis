""" EEG measurement example

Example how to get measurements and
save to fif format
using acquisition class from brainaccess.utils

Change Bluetooth device name
"""

import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import os
from scipy.signal import savgol_filter, butter, filtfilt

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

# Force TkAgg backend for real-time plotting
matplotlib.use("TKAgg", force=True)

eeg = acquisition.EEG()

# --- CONFIGURATION ---

# 1. Device Settings
device_name = "BA HALO 031"

# Define electrode locations depending on your device
halo: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}


cap: dict = {
 0: "F3",
 1: "F4",
 2: "C3",
 3: "C4",
 4: "P3",
 5: "P4",
 6: "O1",
 7: "O2",
}

# 2. Recording Settings
RECORDING_DURATION = 1.0  # Cycle duration in seconds
DEBUG_MODE = True         # Save .fif file every cycle
PLOT_UPDATE_RATE = 0.05   # Update plot every 50ms (20 FPS)

# 3. Signal Processing / Denoising Options
# Options: "raw", "moving_average", "savgol", "bandpass"
PROCESSING_CONFIG = {
    "method": "savgol",       # Active method
    "save_filtered": False,   # If True, saves filtered data. If False, saves RAW data (Recommended: False)
    
    # Settings for specific methods:
    "mov_avg_window": 5,      # Window size for Moving Average (samples)
    "savgol_window": 11,      # Window length for Savitzky-Golay (must be odd)
    "savgol_poly": 3,         # Polynomial order for Savitzky-Golay
    "bp_low": 1,              # Bandpass low cut (Hz)
    "bp_high": 30             # Bandpass high cut (Hz)
}

# Ensure data directory exists
os.makedirs('./data', exist_ok=True)

def sleep_ms(milliseconds):
    """Sleep for the specified number of milliseconds"""
    time.sleep(milliseconds / 1000.0)

def apply_signal_processing(data_chunk, sfreq=250):
    """
    Applies the configured smoothing/denoising to a numpy array (Channels x Samples).
    """
    method = PROCESSING_CONFIG["method"]
    
    # Return immediately if raw or empty
    if method == "raw" or data_chunk.shape[1] == 0:
        return data_chunk

    try:
        if method == "moving_average":
            # Simple boxcar smoothing
            window = PROCESSING_CONFIG["mov_avg_window"]
            kernel = np.ones(window) / window
            # Convolve along the last axis (time)
            # mode='same' keeps the output size equal to input
            return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data_chunk)

        elif method == "savgol":
            # Savitzky-Golay filter (smooths while preserving peak features)
            window = PROCESSING_CONFIG["savgol_window"]
            poly = PROCESSING_CONFIG["savgol_poly"]
            # Check if data is long enough for the window
            if data_chunk.shape[1] > window:
                return savgol_filter(data_chunk, window, poly, axis=1)
            else:
                return data_chunk

        elif method == "bandpass":
            # Butterworth Bandpass Filter
            # Note: filtering short chunks (like 250ms) creates edge artifacts.
            # This is best used on the full 1-second epoch, not the small live chunks.
            low = PROCESSING_CONFIG["bp_low"]
            high = PROCESSING_CONFIG["bp_high"]
            nyquist = 0.5 * sfreq
            b, a = butter(2, [low / nyquist, high / nyquist], btype='band')
            # filtfilt applies filter forward and backward (zero phase)
            # We use a try/except because filtfilt requires a minimum signal length
            if data_chunk.shape[1] > 15:
                return filtfilt(b, a, data_chunk, axis=1)
            else:
                return data_chunk

    except Exception as e:
        print(f"Filter error ({method}): {e}")
        return data_chunk
    
    return data_chunk

# start EEG acquisition setup
with EEGManager() as mgr:
    # IMPORTANT: Set zeros_at_start=1 to prevent IndexError on high-speed loops
    eeg.setup(mgr, device_name=device_name, cap=halo, sfreq=250, zeros_at_start=1)

    eeg.start_acquisition()
    print("Acquisition started")
    print(f"Cycle: {RECORDING_DURATION}s | Filter: {PROCESSING_CONFIG['method']}")
    time.sleep(1) 

    start_time = time.time()
    last_plot_time = time.time()
    annotation_counter = 1
    
    plt.ion()
    fig, ax = plt.subplots()
    eeg.annotate(str(annotation_counter))

    try:
        while True:
            # 1. Minimal sleep for high-speed loop
            time.sleep(0.001)
            
            current_time = time.time()
            elapsed = current_time - start_time

            # 2. Check if Cycle Finished
            if elapsed >= RECORDING_DURATION:
                if DEBUG_MODE:
                    # Fetch final data for this cycle
                    eeg.get_mne()
                    
                    # Optional: Apply filter to the SAVED data
                    # (Usually EEG is saved raw, but user configuration allows override)
                    if PROCESSING_CONFIG["save_filtered"]:
                        # We must extract data, filter, and put it back (complex for MNE object)
                        # Or simpler: Use MNE's built-in filter method
                        if PROCESSING_CONFIG["method"] == "bandpass":
                             eeg.data.mne_raw.filter(PROCESSING_CONFIG["bp_low"], 
                                                     PROCESSING_CONFIG["bp_high"], 
                                                     verbose=False)
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'./data/{timestamp}-cycle_{annotation_counter}.fif'
                    eeg.data.save(filename)
                    print(f"[{timestamp}] Saved cycle {annotation_counter} ({elapsed:.3f}s)")
                
                # --- RESET BUFFERS ---
                with eeg.lock:
                    n_chans = len(eeg.info['ch_names'])
                    zeros_at_start = eeg.data.zeros_at_start
                    eeg.data.data = [np.zeros((n_chans, zeros_at_start))]
                    eeg.data.annotations = {}
                    if hasattr(eeg.data, 'mne_raw'):
                        del eeg.data.mne_raw
                
                start_time = time.time()
                annotation_counter += 1
                eeg.annotate(str(annotation_counter))
                continue

            # 3. Visualization (Throttled)
            if current_time - last_plot_time > PLOT_UPDATE_RATE:
                try:
                    eeg.get_mne()
                    
                    if hasattr(eeg.data, 'mne_raw'):
                        mne_raw = eeg.data.mne_raw
                        data, _ = mne_raw.get_data(return_times=True)
                        
                        # Visualization Window (last 250ms)
                        sfreq = 250
                        window_samples = int(0.25 * sfreq) 
                        
                        if data.shape[1] > 1: 
                            start_idx = -window_samples if data.shape[1] > window_samples else 0
                            raw_chunk = data[:, start_idx:]
                            
                            # --- APPLY DENOISING FOR PLOT ---
                            # This does not affect the MNE object/saved data (unless configured above)
                            clean_chunk = apply_signal_processing(raw_chunk, sfreq)

                            ax.clear()
                            
                            for i in range(clean_chunk.shape[0]):
                                offset = i * 0.0001 
                                label_name = halo.get(i, f"Ch{i}")
                                
                                # Demean and plot
                                if clean_chunk.shape[1] > 0:
                                    signal = clean_chunk[i] - np.mean(clean_chunk[i]) + offset
                                    ax.plot(signal, label=label_name)
                            
                            ax.set_title(f"Live ({PROCESSING_CONFIG['method']}) | Cycle: {annotation_counter}")
                            ax.legend(loc='upper right', fontsize='x-small', framealpha=0.5)
                            
                            plt.draw()
                            plt.pause(0.001)
                except Exception as e:
                    pass
                
                last_plot_time = time.time()

    except KeyboardInterrupt:
        print("\nManually interrupted.")
    finally:
        print("Stopping acquisition...")
        eeg.stop_acquisition()
        mgr.disconnect()
        plt.close(fig)
        print("Done.")