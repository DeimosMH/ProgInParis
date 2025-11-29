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

matplotlib.use("TKAgg", force=True)

eeg = acquisition.EEG()

# define electrode locations depending on your device
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

# define device name
device_name = "BA HALO 031"


def sleep_ms(milliseconds):
    """Sleep for the specified number of milliseconds"""
    time.sleep(milliseconds / 1000.0)

# Ensure data directory exists
os.makedirs('./data', exist_ok=True)

# start EEG acquisition setup
with EEGManager() as mgr:
    # Setup with 250Hz sampling frequency
    eeg.setup(mgr, device_name=device_name, cap=halo, sfreq=250)

    # Start acquiring data
    eeg.start_acquisition()
    print("Acquisition started")
    time.sleep(3)

    start_time = time.time()
    annotation = 1
    
    # Setup real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    
    # Define duration for the recording loop (e.g., 120 seconds)
    recording_duration = 5 

    try:
        while True:
            sleep_ms(250)

            # send annotation to the device
            print("-> ")
            eeg.annotate(str(annotation))
            annotation += 1

            # Retrieve data from the device buffer into the MNE object
            eeg.get_mne()
            
            # Read and prepare part of data to plot/print
            mne_raw = eeg.data.mne_raw
            
            # Get data as numpy array (Channels x Samples)
            data, _ = mne_raw.get_data(return_times=True)
            
            # Calculate number of samples in 250ms
            sfreq = 250
            n_samples = int(0.25 * sfreq)

            # Ensure we have enough data to slice
            if data.shape[1] >= n_samples:
                # print part of data from 250ms ago
                chunk = data[:, -n_samples:]
                
                # Print average voltage of the first channel
                print(f"Data shape: {chunk.shape} | Ch0 Mean (last 250ms): {np.mean(chunk[0]):.2e}")

                # update plot part of data from 250ms ago
                ax.clear()
                
                # Plot channels with an offset for visibility
                for i in range(chunk.shape[0] - 1):
                    # Center the data and add offset
                    offset = i * 0.0001 
                    
                    # Get label from the dictionary if available, otherwise generic ID
                    label_name = halo.get(i, f"Ch{i}")
                    
                    ax.plot(chunk[i] - np.mean(chunk[i]) + offset, label=label_name)
                
                ax.set_title(f"Live Data (Cycle Time: {int(time.time() - start_time)}s)")
                ax.set_xlabel("Samples (last 250ms)")
                
                # Add Legend based on dictionary data
                ax.legend(loc='upper right', fontsize='small', framealpha=0.5)
                
                plt.draw()
                plt.pause(0.001)

                # Check if recording duration has passed to finish cycle and reset
                if time.time() - start_time > recording_duration:
                    print(f"\n--- Cycle finished ({recording_duration}s) ---")

                    ## Plot all data (Snapshot) ##
                    full_data, full_times = mne_raw.get_data(return_times=True)
                    ax.clear()
                    for i in range(full_data.shape[0]):
                        offset = i * 0.0001
                        label_name = halo.get(i, f"Ch{i}")
                        ax.plot(full_times, full_data[i] - np.mean(full_data[i]) + offset, label=label_name)
                    
                    ax.set_title(f"Full Cycle Data saved at {time.strftime('%H:%M:%S')}")
                    ax.set_xlabel("Time (s)")
                    ax.legend(loc='upper right', fontsize='small')
                    plt.draw()
                    plt.pause(1.0) 

                    ## Save EEG data ##
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f'./data/{timestamp}-raw.fif'
                    eeg.data.save(filename)
                    print(f"Data saved to: {filename}")

                    ## Clean data and retrieve again ##
                    print("Resetting internal buffers...")
                    with eeg.lock:
                        n_chans = len(eeg.info['ch_names'])
                        zeros_at_start = eeg.data.zeros_at_start
                        eeg.data.data = [np.zeros((n_chans, zeros_at_start))]
                        eeg.data.annotations = {}
                        if hasattr(eeg.data, 'mne_raw'):
                            del eeg.data.mne_raw

                    start_time = time.time()
                    annotation = 1
                    print("Buffer cleared. Starting new cycle.\n")

    except KeyboardInterrupt:
        print("\nManually interrupted.")
    finally:
        print("Stopping acquisition...")
        eeg.stop_acquisition()
        mgr.disconnect()
        plt.close(fig)
        print("Done.")