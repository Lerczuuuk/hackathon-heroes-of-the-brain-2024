import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from mne import pick_types

matplotlib.use("TKAgg", force=True)

# Initialize EEG acquisition
eeg = acquisition.EEG()

# Define electrode locations
cap: dict = {
    0: "Fp1",
    1: "Fp2",
    2: "O1",
    3: "O2",
}

# Define device name
device_name = "BA HALO 018"

# Blink detection parameters
BLINK_THRESHOLD = 1000  # Threshold for detecting a blink
BLINK_MIN_DURATION = 0.1  # Minimum duration in seconds for detecting a blink (e.g., 0.1s = 10 samples at 100Hz sampling rate)

# Initialize blink detection counters
blink_count_f1 = 0
blink_count_f2 = 0

# Start EEG acquisition setup
with EEGManager() as mgr:
    eeg.setup(mgr, device_name=device_name, cap=cap)

    # Start acquiring data
    eeg.start_acquisition()
    time.sleep(3)

    start_time = time.time()
    annotation = 1
    while time.time() - start_time < 10:
        time.sleep(1)
        
        # Get MNE-compatible raw data from EEG
        raw = eeg.get_mne()

        # Pick EEG channels (Fp1 and Fp2)
        picks = pick_types(raw.info, eeg=True)
        data, times = raw[picks, :]  # Data from EEG channels

        # Extract Fp1 and Fp2 signals (first two channels in the cap dictionary)
        f1_signal = data[0, :]  # Fp1 is the first channel
        f2_signal = data[1, :]  # Fp2 is the second channel

        # Check for blink in Fp1 (F1) and Fp2 (F2)
        for i in range(len(f1_signal)):
            if abs(f1_signal[i]) > BLINK_THRESHOLD:
                blink_count_f1 += 1  # Increase count for Fp1 blink detection
            else:
                blink_count_f1 = 0  # Reset count if the signal is below threshold
            
            if abs(f2_signal[i]) > BLINK_THRESHOLD:
                blink_count_f2 += 1  # Increase count for Fp2 blink detection
            else:
                blink_count_f2 = 0  # Reset count if the signal is below threshold

        # If the blink has been sustained for the required duration (e.g., 0.1 seconds)
        if blink_count_f1 > (BLINK_MIN_DURATION * raw.info['sfreq']):
            print("Blink detected on Fp1!")
            blink_count_f1 = 0  # Reset after detecting blink
        
        if blink_count_f2 > (BLINK_MIN_DURATION * raw.info['sfreq']):
            print("Blink detected on Fp2!")
            blink_count_f2 = 0  # Reset after detecting blink
        
        # Send annotation to the device
        print(f"Sending annotation {annotation} to the device")
        eeg.annotate(str(annotation))
        annotation += 1

    # Stop acquisition
    eeg.stop_acquisition()
    mgr.disconnect()

# Save EEG data to MNE FIF format
eeg.data.save(f'{time.strftime("%Y%m%d_%H%M")}-raw.fif')

# Close the EEG library
eeg.close()

# Show recorded data
eeg.data.mne_raw.filter(1, 40).plot(scalings="auto", verbose=False)
plt.show()
