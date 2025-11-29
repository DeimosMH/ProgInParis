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

from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager

# Force TkAgg backend for real-time plotting
matplotlib.use("TKAgg", force=True)

eeg = acquisition.EEG()

# --- CONFIGURATION ---
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

# Define device name
device_name = "BA HALO 031"

# Recording settings
RECORDING_DURATION = 1.0  # Cycle duration in seconds
DEBUG_MODE = True         # If True, save file every cycle.
PLOT_UPDATE_RATE = 0.02   # Update plot every 50ms (20 FPS) to save CPU

# Ensure data directory exists
os.makedirs('./data', exist_ok=True)

def sleep_ms(milliseconds):
    """Sleep for the specified number of milliseconds"""
    time.sleep(milliseconds / 1000.0)

# start EEG acquisition setup
with EEGManager() as mgr:
    # Setup with 250Hz sampling frequency
    # IMPORTANT: Set zeros_at_start=1 to prevent IndexError when accessing data before packets arrive
    eeg.setup(mgr, device_name=device_name, cap=halo, sfreq=250, zeros_at_start=1)

    # Start acquiring data
    eeg.start_acquisition()
    print("Acquisition started")
    print(f"Cycle duration: {RECORDING_DURATION}s | Debug Mode: {DEBUG_MODE}")
    time.sleep(1) # Allow device to stabilize

    # Initialize loop variables
    start_time = time.time()
    last_plot_time = time.time()
    annotation_counter = 1
    
    # Setup real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    
    # Initial annotation
    eeg.annotate(str(annotation_counter))

    try:
        while True:
            # 1. Minimal sleep for high-speed loop
            time.sleep(0.001)
            
            current_time = time.time()
            elapsed = current_time - start_time

            # 2. Check if Cycle Finished (1 Second)
            if elapsed >= RECORDING_DURATION:
                # --- END OF CYCLE ---
                
                if DEBUG_MODE:
                    # Fetch final data for this cycle
                    eeg.get_mne()
                    
                    # Generate filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'./data/{timestamp}-cycle_{annotation_counter}.fif'
                    
                    # Save data
                    eeg.data.save(filename)
                    print(f"[{timestamp}] Saved cycle {annotation_counter} ({elapsed:.3f}s)")
                
                # --- RESET BUFFERS ---
                # We lock to ensure thread safety while resetting the buffer
                with eeg.lock:
                    n_chans = len(eeg.info['ch_names'])
                    zeros_at_start = eeg.data.zeros_at_start
                    
                    # Clear raw data list, re-initializing with the safe zero-padding
                    eeg.data.data = [np.zeros((n_chans, zeros_at_start))]
                    
                    # Clear annotations
                    eeg.data.annotations = {}
                    
                    # Delete MNE object to force regeneration next cycle
                    if hasattr(eeg.data, 'mne_raw'):
                        del eeg.data.mne_raw
                
                # Update Cycle Variables
                start_time = time.time()
                annotation_counter += 1
                
                # Send annotation for the NEW cycle
                # print("->")
                eeg.annotate(str(annotation_counter))
                continue # Skip the rest of the loop to start fresh immediately

            # 3. Visualization (Throttled)
            if current_time - last_plot_time > PLOT_UPDATE_RATE:
                # Wrap visualization in try/except to prevent loop crash on transient data states
                try:
                    eeg.get_mne()
                    
                    if hasattr(eeg.data, 'mne_raw'):
                        mne_raw = eeg.data.mne_raw
                        data, _ = mne_raw.get_data(return_times=True)
                        
                        # Use last 250ms for visualization
                        sfreq = 250
                        window_samples = int(0.25 * sfreq) 
                        
                        # Only plot if we have more data than just the initialization zeros
                        if data.shape[1] > 1: 
                            
                            start_idx = -window_samples if data.shape[1] > window_samples else 0
                            chunk = data[:, start_idx:]
                            
                            ax.clear()
                            
                            # Plot channels
                            for i in range(chunk.shape[0]):
                                offset = i * 0.0001 
                                label_name = halo.get(i, f"Ch{i}")
                                
                                if chunk.shape[1] > 0:
                                    ax.plot(chunk[i] - np.mean(chunk[i]) + offset, label=label_name)
                            
                            ax.set_title(f"Live Monitor (Cycle: {annotation_counter} | {elapsed:.2f}s)")
                            ax.set_xlabel("Samples")
                            ax.legend(loc='upper right', fontsize='small', framealpha=0.5)
                            
                            plt.draw()
                            plt.pause(0.001)
                except Exception as e:
                    # Ignore visualization errors to keep acquisition running
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