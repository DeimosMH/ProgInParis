import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import os
from scipy.signal import savgol_filter, butter, filtfilt

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
import brainaccess.core.eeg_channel as eeg_channel

# --- CONFIGURATION ---

# 1. Device & BCI Settings
DEVICE_NAME = "BA HALO 031"
CYCLE_DURATION_S = 1.0  # How often to process data and make a decision (in seconds)

# 2. Channel Mapping (Corrected: 'Sample' channel removed)
# The 'Sample' channel is handled automatically by the library.
# Defining it here causes the montage error.
HALO_CAP: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}

# 3. Signal Processing / Denoising Options
PROCESSING_CONFIG = {
    "method": "savgol",
    "savgol_window": 11,
    "savgol_poly": 3,
    "bp_low": 1,
    "bp_high": 30
}

# 4. Visualization Settings
PLOT_ENABLED = True
PLOT_UPDATE_RATE_S = 0.1 # Update plot 10 times per second
PLOT_WINDOW_S = 2.0       # Show the last 2 seconds of data in the plot

# --- END OF CONFIGURATION ---

def apply_signal_processing(data, sfreq=250):
    """Applies configured denoising to a numpy array (Channels x Samples)."""
    method = PROCESSING_CONFIG["method"]
    if method == "raw" or data.shape[1] == 0:
        return data
    try:
        if method == "savgol":
            window, poly = PROCESSING_CONFIG["savgol_window"], PROCESSING_CONFIG["savgol_poly"]
            if data.shape[1] > window:
                return savgol_filter(data, window, poly, axis=1)
        elif method == "bandpass":
            low, high, nyquist = PROCESSING_CONFIG["bp_low"], PROCESSING_CONFIG["bp_high"], 0.5 * sfreq
            b, a = butter(2, [low / nyquist, high / nyquist], btype='band')
            if data.shape[1] > 15: # filtfilt requires a minimum data length
                return filtfilt(b, a, data, axis=1)
    except Exception as e:
        print(f"Filter error ({method}): {e}")
    return data

def detect_mouse_click(eeg_data, channel_names):
    """
    Placeholder for your pattern recognition logic.
    
    Args:
        eeg_data (np.ndarray): A (channels x samples) numpy array of processed EEG data.
        channel_names (list): A list of channel names corresponding to the rows in eeg_data.
    
    Returns:
        str: "LEFT", "RIGHT", or "NONE"
    """
    # --- YOUR LOGIC GOES HERE ---
    # Example: Find the index for the 'Fp1' channel
    try:
        fp1_index = channel_names.index('Fp1')
        fp1_data = eeg_data[fp1_index]
        
        # This is a placeholder. You will replace this with your actual
        # slope detection or pattern recognition algorithm.
        if np.mean(fp1_data) > 1e-5: # Example threshold
             return "LEFT CLICK"
             
    except ValueError:
        pass # Channel not found

    return "NONE"

def main():
    """Main function to run the real-time BCI loop."""
    # This must be at the top of the function
    global PLOT_ENABLED
    
    eeg = acquisition.EEG()
    
    if PLOT_ENABLED:
        try:
            matplotlib.use("TKAgg", force=True)
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 6))
        except ImportError:
            print("Warning: TkAgg backend not available. Disabling real-time plotting.")
            PLOT_ENABLED = False

    with EEGManager() as mgr:
        print(f"Setting up device: {DEVICE_NAME}...")
        # Note: cap now correctly uses the modified HALO_CAP dictionary
        eeg.setup(mgr, device_name=DEVICE_NAME, cap=HALO_CAP, sfreq=250, zeros_at_start=1)
        eeg.start_acquisition()
        print("Acquisition started. Starting BCI loop...")
        time.sleep(1)

        last_plot_time = time.time()
        
        try:
            while True:
                # 1. Get the most recent chunk of data
                mne_raw = eeg.get_mne(tim=CYCLE_DURATION_S)
                if not mne_raw or mne_raw.get_data().shape[1] == 0:
                    time.sleep(0.1)
                    continue

                # 2. Process the data
                raw_data, _ = mne_raw.get_data(return_times=True)
                clean_data = apply_signal_processing(raw_data.copy(), eeg.sfreq)
                
                # 3. Make a decision
                decision = detect_mouse_click(clean_data, mne_raw.ch_names)
                print(f"Cycle decision: {decision}")

                # 4. Update plot (throttled)
                if PLOT_ENABLED and (time.time() - last_plot_time > PLOT_UPDATE_RATE_S):
                    update_plot(eeg, ax)
                    last_plot_time = time.time()

                # 5. Wait for the next cycle
                time.sleep(CYCLE_DURATION_S)

        except KeyboardInterrupt:
            print("\nManually interrupted.")
        finally:
            print("Stopping acquisition...")
            eeg.stop_acquisition()
            mgr.disconnect()
            if PLOT_ENABLED:
                plt.ioff()
                plt.close()
            print("Done.")

def update_plot(eeg, ax):
    """Fetches recent data and updates the matplotlib plot."""
    try:
        # Get a slightly larger window for plotting to see context
        mne_raw_plot = eeg.get_mne(tim=PLOT_WINDOW_S)
        if mne_raw_plot:
            data, _ = mne_raw_plot.get_data(return_times=True)
            if data.shape[1] > 1:
                clean_data = apply_signal_processing(data.copy(), eeg.sfreq)
                ax.clear()
                for i, ch_name in enumerate(mne_raw_plot.ch_names):
                    offset = i * 100 
                    # Convert to microvolts for plotting
                    signal = (clean_data[i] * 1e6) - np.mean(clean_data[i] * 1e6) + offset
                    ax.plot(signal, label=ch_name)
                ax.set_title(f"Live EEG ({PROCESSING_CONFIG['method']})")
                ax.set_ylabel("Amplitude (uV) + Offset")
                ax.legend(loc='upper right', fontsize='small')
                plt.draw()
                plt.pause(0.001)
    except Exception as e:
        # This can happen if the plot window is closed manually
        print(f"Plotting error: {e}")
        global PLOT_ENABLED
        PLOT_ENABLED = False


if __name__ == "__main__":
    main()