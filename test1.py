import matplotlib.pyplot as plt
import matplotlib
import time
import numpy as np
from brainaccess.utils import acquisition
from brainaccess.core.eeg_manager import EEGManager
from mne.preprocessing import ICA
from mne import pick_types
from mne.io import RawArray

matplotlib.use("TKAgg", force=True)

# Initialize EEG acquisition
eeg = acquisition.EEG()

# Define electrode locations
cap: dict = {
    0: "F1",
    1: "F2",
    2: "O1",
    3: "O2",
}

# Define device name
device_name = "BA MINI 018"

# Blink detection thresholds
BLINK_THRESHOLD_F1 = 1000  # Increased threshold for F1 channel (experiment with higher value)
BLINK_THRESHOLD_F2 = 1000  # Increased threshold for F2 channel (experiment with higher value)

# Minimum duration (in seconds) the signal should exceed threshold for detection
BLINK_MIN_DURATION = 0.1  # Duration to consider it as a blink (adjustable)

# Blink detection flags to prevent false positives
blink_detected_f1 = False
blink_detected_f2 = False

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

        # Retrieve MNE-compatible raw data from EEG
        raw = eeg.get_mne()

        # Pick EEG channels (F1 and F2)
        picks = pick_types(raw.info, eeg=True)
        data, times = raw[picks, :]  # Data from EEG channels

        # Extract F1 and F2 signals (first two channels in the cap dictionary)
        f1_signal = data[0, :]  # Assuming F1 is the first channel
        f2_signal = data[1, :]  # Assuming F2 is the second channel

        # Check for blinks in F1 and F2 based on threshold
        blink_f1 = np.any(np.abs(f1_signal) > BLINK_THRESHOLD_F1)
        blink_f2 = np.any(np.abs(f2_signal) > BLINK_THRESHOLD_F2)

        # Implement minimum duration check for blink detection
        if blink_f1 and not blink_detected_f1:
            blink_detected_f1 = True
            print("Blink detected on F1!")
            time.sleep(BLINK_MIN_DURATION)  # Prevent multiple detections for the same blink
        elif not blink_f1:
            blink_detected_f1 = False

        if blink_f2 and not blink_detected_f2:
            blink_detected_f2 = True
            print("Blink detected on F2!")
            time.sleep(BLINK_MIN_DURATION)  # Prevent multiple detections for the same blink
        elif not blink_f2:
            blink_detected_f2 = False

        # Send annotation to the device (if necessary)
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
