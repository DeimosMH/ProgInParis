""" EEG measurement example

Example how to get measurements and
save to fif format
using acquisition class from brainaccess.utils

Change Bluetooth device name
"""

import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np  # Added for array manipulation

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
    
    # Define duration for the recording loop (e.g., 20 seconds)
    recording_duration = 20 

    while True:
        # time.sleep(1)
        sleep_ms(250)

        # # Check if recording duration has passed to exit loop
        # if time.time() - start_time > recording_duration:
        #     break

        # send annotation to the device
        # print(f"Sending annotation {annotation} to the device")
        
        print("-> ")
        eeg.annotate(str(annotation))
        annotation += 1

        # Retrieve data from the device buffer into the MNE object
        # Note: calling get_mne() without arguments retrieves the entire recording history.
        eeg.get_mne()
        
        # Read and prepare part of data to plot/print
        mne_raw = eeg.data.mne_raw
        
        # Get data as numpy array (Channels x Samples)
        data, _ = mne_raw.get_data(return_times=True)
        
        # Calculate number of samples in 250ms (0.25s * 250Hz = ~62 samples)
        sfreq = 250
        n_samples = int(0.25 * sfreq)

        # Ensure we have enough data to slice
        if data.shape[1] >= n_samples:
            # print part of data from 250ms ago
            chunk = data[:, -n_samples:]
            
            # Print average voltage of the first channel in the last 250ms
            # (Multiplied by 1e6 to show meaningful microvolt-scale numbers if raw is in Volts)
            print(f"Data shape: {chunk.shape} | Ch0 Mean (last 250ms): {np.mean(chunk[0]):.2e}")

            # update plot part of data from 250ms ago
            ax.clear()
            
            # Plot channels with an offset for visibility
            for i in range(chunk.shape[0]):
                # Center the data and add offset
                offset = i * 0.0001 
                ax.plot(chunk[i] - np.mean(chunk[i]) + offset, label=f'Ch{i}')
            
            ax.set_title("Last 250ms Data")
            ax.set_xlabel("Samples")
            plt.draw()
            plt.pause(0.001)

    print("Preparing to plot data")
    # Close the realtime plot
    plt.close(fig)
    time.sleep(2)

    # get all eeg data and stop acquisition
    eeg.get_mne()
    eeg.stop_acquisition()
    mgr.disconnect()

# Access MNE Raw object
mne_raw = eeg.data.mne_raw
print(f"MNE Raw object: {mne_raw}")

# Access data as NumPy arrays
data, times = mne_raw.get_data(return_times=True)
print(f"Data shape: {data.shape}")

# save EEG data to MNE fif format
eeg.data.save(f'./data/{time.strftime("%Y%m%d_%H%M")}-raw.fif')

# Close brainaccess library
eeg.close()

# conversion to microvolts
mne_raw.apply_function(lambda x: x*10**-6)

# Show recorded data (post-processing plot)
# Turn off interactive mode so plt.show() blocks
plt.ioff() 
mne_raw.filter(1, 40).plot(scalings="auto", verbose=False)
plt.show()