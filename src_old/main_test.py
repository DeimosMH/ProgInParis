import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import os
from scipy.signal import savgol_filter, butter, filtfilt

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

# --- CONFIGURATION ---

# 1. Device Settings
DEVICE_CONFIG = {
    "name": "BA HALO 031",
    "gain": 8,
    "sfreq": 250 # The requested sample frequency
}

# 2. BCI Logic Settings
BCI_CONFIG = {
    "cycle_duration_s": 1.0  # How often to process data and make a decision
}

# 3. ADC & Conversion Parameters (Based on ADS1299 Datasheet)
ADC_PARAMS = {
    "v_ref": 4.5,  # Volts
    "adc_bits": 24
}
# Automatically calculate the LSB value based on gain
LSB_uV = ((2 * ADC_PARAMS["v_ref"]) / (DEVICE_CONFIG["gain"] * (2**ADC_PARAMS["adc_bits"]))) * 1e6

# 4. Channel Mapping
# 'Sample' channel is handled automatically and should not be in the cap.
HALO_CAP: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}

# 5. Signal Processing / Denoising Options
PROCESSING_CONFIG = {
    "method": "savgol",
    "savgol_window": 11,
    "savgol_poly": 3,
    "bp_low": 1,
    "bp_high": 30
}

# 6. Visualization Settings
PLOT_CONFIG = {
    "enabled": True,
    "update_rate_s": 0.1,
    "window_s": 2.0
}

# --- END OF CONFIGURATION ---

def print_device_info(mgr):
    """Queries and prints key information from the connected device."""
    print("\n--- Device Information ---")
    try:
        info = mgr.get_device_info()
        battery_info = mgr.get_battery_info()
        
        print(f"  Device Name:    {info.name}")
        print(f"  Serial Number:  {info.serial_number}")
        print(f"  Firmware Ver:   {info.firmware_version}")
        print(f"  Battery Level:  {battery_info.level}%")
    except Exception as e:
        print(f"  Could not retrieve all device info: {e}")
    print("--------------------------")

def print_configuration(confirmed_sfreq):
    """Prints a formatted summary of the script's configuration."""
    print("\n--- Running Configuration ---")
    # Device
    print(f"[Device]")
    print(f"  Name:             {DEVICE_CONFIG['name']}")
    print(f"  Gain:             x{DEVICE_CONFIG['gain']}")
    print(f"  Requested SFREQ:  {DEVICE_CONFIG['sfreq']} Hz")
    print(f"  Confirmed SFREQ:  {confirmed_sfreq} Hz (Using this value)")
    print(f"  LSB Conversion:   {LSB_uV:.5f} µV per ADC count")
    # BCI
    print(f"[BCI Logic]")
    print(f"  Cycle Duration:   {BCI_CONFIG['cycle_duration_s']} s")
    # Processing
    print(f"[Signal Processing]")
    print(f"  Method:           {PROCESSING_CONFIG['method']}")
    if PROCESSING_CONFIG['method'] == 'savgol':
        print(f"  Savgol Window:    {PROCESSING_CONFIG['savgol_window']} samples")
        print(f"  Savgol Poly:      {PROCESSING_CONFIG['savgol_poly']}")
    elif PROCESSING_CONFIG['method'] == 'bandpass':
        print(f"  Bandpass Range:   {PROCESSING_CONFIG['bp_low']}-{PROCESSING_CONFIG['bp_high']} Hz")
    # Plotting
    print(f"[Visualization]")
    print(f"  Enabled:          {PLOT_CONFIG['enabled']}")
    if PLOT_CONFIG['enabled']:
        print(f"  Update Rate:      {PLOT_CONFIG['update_rate_s']} s")
        print(f"  Display Window:   {PLOT_CONFIG['window_s']} s")
    print("---------------------------\n")

def apply_signal_processing(data_uV, sfreq):
    """Applies configured denoising to data in microvolts."""
    method = PROCESSING_CONFIG["method"]
    if method == "raw" or data_uV.shape[1] == 0: return data_uV
    try:
        if method == "savgol":
            window, poly = PROCESSING_CONFIG["savgol_window"], PROCESSING_CONFIG["savgol_poly"]
            if data_uV.shape[1] > window: return savgol_filter(data_uV, window, poly, axis=1)
        elif method == "bandpass":
            low, high, nyquist = PROCESSING_CONFIG["bp_low"], PROCESSING_CONFIG["bp_high"], 0.5 * sfreq
            b, a = butter(2, [low / nyquist, high / nyquist], btype='band')
            if data_uV.shape[1] > 15: return filtfilt(b, a, data_uV, axis=1)
    except Exception as e:
        print(f"Filter error ({method}): {e}")
    return data_uV

def detect_mouse_click(eeg_data_uV, channel_names):
    """Placeholder for your pattern recognition logic. Data is in microvolts (µV)."""
    try:
        fp1_index = channel_names.index('Fp1')
        fp1_data = eeg_data_uV[fp1_index]
        if np.mean(fp1_data) > 50.0:
             print(f"Decision Trigger: Fp1 mean is {np.mean(fp1_data):.2f} µV")
             return "LEFT CLICK"
    except (ValueError, IndexError):
        pass
    return "NONE"

def main():
    """Main function to run the real-time BCI loop."""
    eeg = acquisition.EEG()
    
    if PLOT_CONFIG['enabled']:
        try:
            matplotlib.use("TKAgg", force=True)
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 6))
        except ImportError:
            print("Warning: TkAgg backend not available. Disabling real-time plotting.")
            PLOT_CONFIG['enabled'] = False

    with EEGManager() as mgr:
        print(f"Connecting to device: {DEVICE_CONFIG['name']}...")
        eeg.setup(mgr, device_name=DEVICE_CONFIG['name'], cap=HALO_CAP, sfreq=DEVICE_CONFIG['sfreq'], gain=DEVICE_CONFIG['gain'], zeros_at_start=1)
        
        # --- STARTUP INFORMATION SEQUENCE ---
        print_device_info(mgr)
        
        # Update config with actual sample rate from device
        confirmed_sfreq = eeg.info['sfreq']
        print_configuration(confirmed_sfreq)
        # --- END STARTUP ---

        eeg.start_acquisition()
        print("Acquisition started. BCI loop is running...")
        time.sleep(1)

        last_cycle_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                
                if current_time - last_cycle_time >= BCI_CONFIG['cycle_duration_s']:
                    mne_raw = eeg.get_mne(tim=BCI_CONFIG['cycle_duration_s'])
                    if not mne_raw or mne_raw.get_data().shape[1] == 0: continue
                    
                    raw_counts, _ = mne_raw.get_data(return_times=True)
                    data_uV = raw_counts * LSB_uV
                    clean_data_uV = apply_signal_processing(data_uV.copy(), confirmed_sfreq)
                    decision = detect_mouse_click(clean_data_uV, mne_raw.ch_names)
                    print(f"Cycle decision: {decision}")
                    last_cycle_time = current_time

                if PLOT_CONFIG['enabled']:
                    update_plot(eeg, ax, confirmed_sfreq)
                
                time.sleep(PLOT_CONFIG['update_rate_s'])

        except KeyboardInterrupt:
            print("\nManually interrupted.")
        finally:
            print("Stopping acquisition...")
            eeg.stop_acquisition()
            mgr.disconnect()
            if PLOT_CONFIG['enabled']:
                plt.ioff(); plt.close()
            print("Done.")

def update_plot(eeg, ax, sfreq):
    """Fetches, converts, and plots recent data."""
    try:
        mne_raw_plot = eeg.get_mne(tim=PLOT_CONFIG['window_s'])
        if mne_raw_plot:
            data_counts, _ = mne_raw_plot.get_data(return_times=True)
            if data_counts.shape[1] > 1:
                data_uV = data_counts * LSB_uV
                clean_data_uV = apply_signal_processing(data_uV.copy(), sfreq)
                
                ax.clear()
                for i, ch_name in enumerate(mne_raw_plot.ch_names):
                    offset = i * 100 
                    signal = clean_data_uV[i] - np.mean(clean_data_uV[i]) + offset
                    ax.plot(signal, label=ch_name)
                ax.set_title(f"Live EEG ({PROCESSING_CONFIG['method']})")
                ax.set_ylabel("Amplitude (µV) + Offset")
                ax.legend(loc='upper right', fontsize='small')
                plt.draw(); plt.pause(0.001)
    except Exception as e:
        print(f"Plotting error: {e}")
        PLOT_CONFIG['enabled'] = False

if __name__ == "__main__":
    main()