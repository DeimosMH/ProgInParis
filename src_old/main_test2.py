import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
import logging

# Handle optional dependencies gracefully
try:
    from scipy.signal import butter, lfilter, filtfilt, iirnotch, lfilter_zi
    from scipy import signal as sp_signal
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("Scipy not available - signal processing will be limited")
    SCIPY_AVAILABLE = False
    # Create dummy functions for when scipy is not available
    def butter(order, frequencies, btype='band'):
        return np.ones(3), np.ones(3)
    def lfilter(b, a, data, axis=1, zi=None):
        if zi is not None:
            return data, zi
        return data
    def filtfilt(b, a, data, axis=1):
        return data
    def iirnotch(freq, q, sfreq):
        return np.ones(3), np.ones(3)
    def lfilter_zi(b, a):
        return np.zeros(max(len(b), len(a)) - 1)

try:
    from brainaccess.utils import acquisition
    from brainaccess.core.eeg_manager import EEGManager
    from brainaccess.utils.exceptions import BrainAccessException
    BRAINACCESS_AVAILABLE = True
except ImportError:
    logging.warning("BrainAccess not available - hardware functionality disabled")
    BRAINACCESS_AVAILABLE = False
    # Create dummy classes for testing if brainaccess is not installed
    class acquisition:
        class EEG:
            def __init__(self, mode="accumulate"): 
                self.sfreq = 250
            def setup(self, *args, **kwargs): 
                logging.info("Dummy EEG setup.")
            def start_acquisition(self): 
                logging.info("Dummy acquisition started.")
            def stop_acquisition(self): 
                logging.info("Dummy acquisition stopped.")
            def get_mne(self, *args, **kwargs): 
                return None
    class EEGManager:
        def __enter__(self): 
            return self
        def __exit__(self, *args): 
            pass
    class BrainAccessException(Exception): 
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
DEVICE_NAME = "BA HALO 031"
SFREQ = 250
CYCLE_DURATION_S = 0.1  # Process data every 100ms
BUFFER_SIZE_S = 3.0     # The total size of the rolling buffer

HALO_CAP: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}

PROCESSING_CONFIG = {
    "method": "bandpass_causal",
    "bp_low": 1,
    "bp_high": 40,  # Increased to capture more EMG/EOG
    "notch_freq": 50.0,
    "notch_q": 30.0
}

PLOT_ENABLED = True
PLOT_UPDATE_RATE_S = 0.1
PLOT_WINDOW_S = 2.5

# --- Global filter state for the main causal filter ---
filter_state = { 'bandpass': None, 'notch': None }

# --- END OF CONFIGURATION ---

def bandpower(data, sfreq, band, relative=False):
    """Compute the average power of the signal in a frequency band using Welch's method."""
    if not SCIPY_AVAILABLE or data.ndim == 0 or data.shape[-1] < sfreq * 0.25:
        return 0.0
    
    band = np.asarray(band)
    low, high = band
    nperseg = min(data.shape[-1], sfreq) # Use 1-second windows or less
    
    freqs, psd = sp_signal.welch(data, sfreq, nperseg=nperseg)
    
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    if not np.any(idx_band):
        return 0.0

    freq_res = freqs[1] - freqs[0]
    power = np.trapz(psd[..., idx_band], dx=freq_res)
    
    if relative:
        total_power = np.trapz(psd, dx=freq_res)
        return power / (total_power + 1e-10)
    return power

def apply_causal_filters(data, sfreq):
    """Applies configured causal filtering, protected against instability."""
    global filter_state

    # This clip is still essential to prevent the filter from exploding after a large artifact.
    voltage_limit = 500e-6  # 500 microvolts
    np.clip(data, -voltage_limit, voltage_limit, out=data)

    if not SCIPY_AVAILABLE or data.shape[1] == 0:
        return data

    try:
        nyquist = 0.5 * sfreq
        b_bp, a_bp = butter(2, [PROCESSING_CONFIG["bp_low"] / nyquist, PROCESSING_CONFIG["bp_high"] / nyquist], btype='band')
        b_notch, a_notch = iirnotch(PROCESSING_CONFIG["notch_freq"], PROCESSING_CONFIG["notch_q"], sfreq)

        n_channels = data.shape[0]
        
        # --- FIX: ROBUST FILTER STATE INITIALIZATION USING BROADCASTING ---
        if filter_state['bandpass'] is None or filter_state['bandpass'].shape[1] != n_channels:
            zi_1d = lfilter_zi(b_bp, a_bp)
            # Use broadcasting to create the (order, n_channels) array
            filter_state['bandpass'] = zi_1d[:, np.newaxis] * np.ones((1, n_channels))
            
        if filter_state['notch'] is None or filter_state['notch'].shape[1] != n_channels:
            zi_1d = lfilter_zi(b_notch, a_notch)
            filter_state['notch'] = zi_1d[:, np.newaxis] * np.ones((1, n_channels))
        # --- END FIX ---

        filtered_data, filter_state['notch'] = lfilter(b_notch, a_notch, data, axis=1, zi=filter_state['notch'])
        filtered_data, filter_state['bandpass'] = lfilter(b_bp, a_bp, filtered_data, axis=1, zi=filter_state['bandpass'])
        
        return filtered_data
    except Exception as e:
        logging.error(f"Causal filter error: {e}")
        filter_state = {'bandpass': None, 'notch': None} # Reset state on error
        return data


def detect_artifact(data, ch_names, channels, threshold_v):
    """Generic artifact detector using Peak-to-Peak amplitude."""
    try:
        for ch in channels:
            if ch not in ch_names: return False
        
        indices = [ch_names.index(ch) for ch in channels]
        ptp_values = [np.ptp(data[i]) for i in indices]

        return all(ptp > threshold_v for ptp in ptp_values)
    except (ValueError, IndexError):
        return False

def detect_alpha_waves(data, ch_names, sfreq, relative_power_threshold=0.35):
    """Detects alpha waves on any available occipital channel."""
    try:
        for ch_name in ['O1', 'O2']:
            if ch_name in ch_names:
                idx = ch_names.index(ch_name)
                if data.shape[1] < sfreq * 0.5: continue
                
                relative_alpha = bandpower(data[idx], sfreq, [8, 13], relative=True)
                if relative_alpha > relative_power_threshold:
                    return True
        return False
    except Exception as e:
        logging.warning(f"Alpha detection failed: {e}")
        return False

def make_decision(eeg_data, channel_names, sfreq):
    """Main pattern recognition function."""
    if detect_artifact(eeg_data, channel_names, ['Fp1', 'Fp2'], threshold_v=100e-6):
        return "BLINK"
    # NOTE: Jaw clench detection requires T7/T8 which are not in the current HALO_CAP
    if detect_alpha_waves(eeg_data, channel_names, sfreq):
        return "ALPHA_WAVES"
    return "NONE"

def update_plot(eeg, ax):
    """Fetches and plots data using its own non-causal filter for visualization."""
    global PLOT_ENABLED
    if not eeg or not PLOT_ENABLED: return
    try:
        mne_raw_plot = eeg.get_mne(tim=PLOT_WINDOW_S)
        if mne_raw_plot and mne_raw_plot.get_data().shape[1] > 30:
            data, _ = mne_raw_plot.get_data(return_times=True)
            
            b, a = butter(2, [PROCESSING_CONFIG["bp_low"] / (0.5*eeg.sfreq), PROCESSING_CONFIG["bp_high"] / (0.5*eeg.sfreq)], btype='band')
            clean_data = filtfilt(b, a, data, axis=1)

            ax.clear()
            for i, ch_name in enumerate(mne_raw_plot.ch_names):
                offset = i * 100
                signal = (clean_data[i] * 1e6) - np.mean(clean_data[i] * 1e6) + offset
                ax.plot(signal, label=ch_name, linewidth=0.8)
            ax.set_title(f"Live EEG ({PROCESSING_CONFIG['method']})")
            ax.set_ylabel("Amplitude (uV) + Offset")
            ax.legend(loc='upper right', fontsize='small')
            plt.pause(0.001)
    except Exception as e:
        logging.error(f"Plotting error: {e}")
        PLOT_ENABLED = False

def main():
    """Main function to run the real-time BCI loop."""
    global PLOT_ENABLED, filter_state
    if not BRAINACCESS_AVAILABLE or not SCIPY_AVAILABLE:
        logging.error("Missing critical libraries (BrainAccess or Scipy). Cannot run.")
        return

    filter_state = {'bandpass': None, 'notch': None}
    eeg = acquisition.EEG(mode="roll")
    
    fig, ax = None, None
    if PLOT_ENABLED:
        try:
            matplotlib.use("TKAgg", force=True)
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 6))
        except Exception:
            logging.warning("GUI backend not available. Disabling real-time plotting.")
            PLOT_ENABLED = False

    try:
        with EEGManager() as mgr:
            logging.info(f"Setting up device: {DEVICE_NAME}...")
            buffer_size_samples = int(BUFFER_SIZE_S * SFREQ)
            eeg.setup(mgr, device_name=DEVICE_NAME, cap=HALO_CAP, sfreq=SFREQ, zeros_at_start=buffer_size_samples)
            
            eeg.start_acquisition()
            logging.info("Acquisition started. BCI loop running... (Press Ctrl+C to stop)")
            time.sleep(1.5)

            last_plot_time = time.time()
            
            while True:
                cycle_start_time = time.time()
                
                mne_raw = eeg.get_mne(tim=CYCLE_DURATION_S)
                if not mne_raw or mne_raw.get_data().shape[1] == 0:
                    time.sleep(0.01)
                    continue

                raw_data, _ = mne_raw.get_data(return_times=True)
                clean_data = apply_causal_filters(raw_data.copy(), eeg.sfreq)
                
                decision = make_decision(clean_data, mne_raw.ch_names, eeg.sfreq)
                if decision != "NONE":
                    logging.info(f"Decision: {decision}")

                if PLOT_ENABLED and (time.time() - last_plot_time > PLOT_UPDATE_RATE_S):
                    update_plot(eeg, ax)
                    last_plot_time = time.time()

                processing_time = time.time() - cycle_start_time
                sleep_duration = CYCLE_DURATION_S - processing_time
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                elif processing_time > CYCLE_DURATION_S:
                    logging.warning(f"Cycle overrun by {int((processing_time - CYCLE_DURATION_S) * 1000)} ms.")

    except (KeyboardInterrupt, SystemExit):
        logging.info("User interrupted. Shutting down.")
    except BrainAccessException as e:
        logging.error(f"A BrainAccess hardware error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected fatal error occurred: {e}", exc_info=True)
    finally:
        logging.info("Stopping acquisition...")
        if 'eeg' in locals() and eeg:
            eeg.stop_acquisition()
        if PLOT_ENABLED and plt.get_fignums():
            plt.ioff()
            plt.close()
        logging.info("BCI system shutdown complete.")

if __name__ == "__main__":
    main()