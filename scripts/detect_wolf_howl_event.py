#!/usr/bin/env python3
import rospy
import numpy as np
import signal
import sys
from scipy.signal import butter, filtfilt, correlate
from std_msgs.msg import Int8
from miro2 import lib
import wave

EVENT_HOWL = 5

FS_MIC = 20000        # MiRo microphone rate
FS_SPK = 8000         # MiRo speaker playback rate
RATIO = FS_MIC / FS_SPK   # 2.5 exactly
THRESHOLD = 0.05      # normalised correlation threshold (volume-independent)

# ----------------------------------------------------
# Load and preprocess template
# ----------------------------------------------------

HOWL_PATH = "/root/mdk/catkin_ws/src/team04/sound_files/distress_howl_test.wav"

def load_and_process_template(path):
    wf = wave.open(path, 'rb')
    assert wf.getframerate() == FS_MIC
    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    wf.close()

    # Normalise
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Downsample 20k → 8k exactly like MiRo speaker
    j = np.arange(len(audio))
    i = np.arange(0, len(audio), RATIO)
    audio_ds = np.interp(i, j, audio)

    # Bandpass filter (cleaner for detection)
    b, a = butter(4, [350/(FS_SPK/2), 1600/(FS_SPK/2)], btype='band')
    audio_ds = filtfilt(b, a, audio_ds)

    # Final normalisation
    audio_ds /= (np.linalg.norm(audio_ds) + 1e-6)

    return audio_ds


template = load_and_process_template(HOWL_PATH)
WIN = len(template)


# ============================================================
# HELPER CLASS
# ============================================================

class HowlDetector:

    def __init__(self):

        # 1) INITIALISE BUFFERS *FIRST*
        self.buffer = np.zeros((WIN,), dtype=np.float32)
        self.pub_event = rospy.Publisher("/miro/events/detect_audio_event", Int8, queue_size=1)

        # 2) NOW create RobotInterface (starts mic stream immediately!)
        self.interface = lib.RobotInterface()

        # 3) THEN register callback
        self.interface.register_callback("microphones", self.mic_callback)

        rospy.loginfo("Howl detector READY")

        signal.signal(signal.SIGINT, self.shutdown)

    # --------------------------------------------------------
    # Microphone callback
    # --------------------------------------------------------
    def mic_callback(self, msg):
        data = msg.data.astype(np.float32)  # shape (N,4)

        # Combine all channels for robust inference
        frame = np.mean(data, axis=1)
        L = len(frame)

        # Sliding buffer
        if L < WIN:
            self.buffer = np.roll(self.buffer, -L)
            self.buffer[-L:] = frame
        else:
            self.buffer = frame[-WIN:]

        # Preprocess buffer same as template
        # Downsample to match template rate
        j = np.arange(len(self.buffer))
        i = np.arange(0, len(self.buffer), RATIO)

        if len(i) < len(template):
            return

        buf_ds = np.interp(i, j, self.buffer)
        buf_ds = buf_ds[:len(template)]

        # Bandpass
        b, a = butter(4, [350/(FS_SPK/2), 1600/(FS_SPK/2)], btype='band')
        buf_ds = filtfilt(b, a, buf_ds)

        # Normalise
        buf_ds /= (np.linalg.norm(buf_ds) + 1e-6)

        # Correlation
        corr = correlate(buf_ds, template, mode="valid")
        score = float(np.max(corr))

        if score > THRESHOLD:
            rospy.loginfo(f"[HOWL DETECTED] normalised_score={score:.3f}")
            self.pub_event.publish(Int8(EVENT_HOWL))

    # --------------------------------------------------------
    def shutdown(self, *args):
        try:
            self.interface.disconnect()
        except:
            pass
        rospy.signal_shutdown("User exit")
        sys.exit(0)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    h = HowlDetector()
    rospy.spin()
