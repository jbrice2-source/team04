from pydub import AudioSegment, effects
import numpy as np
import scipy.signal as signal
import wave

# ----------------------------------------------
# CONFIG
# ----------------------------------------------

INPUT  = "catkin_ws/src/team04/sound_files/distress_howl_test.wav"
OUTPUT = "catkin_ws/src/team04/sound_files/distress_howl_testB.wav"

# MiRo rates
FS_IN  = 20000         # input expected samplerate
FS_OUT = 8000          # MiRo speaker effective playback rate
RATIO  = FS_IN / FS_OUT  # 2.5

print("[1/6] Loading original howl...")
audio = AudioSegment.from_file(INPUT).set_channels(1)

# Ensure 20 kHz as expected by detector + MiRo preprocessing
audio = audio.set_frame_rate(FS_IN)

# Convert to numpy array for precise DSP
samples = np.array(audio.get_array_of_samples()).astype(np.float32)

# Normalize to -1 dBFS pre-processing
peak = np.max(np.abs(samples))
samples = samples / (peak + 1e-6)
samples *= 0.89    # ~ -1 dBFS

# ----------------------------------------------
# [2] Apply bandpass 350–1600 Hz (MiRo detection band)
# ----------------------------------------------

print("[2/6] Applying bandpass 350–1600 Hz...")

b, a = signal.butter(
    4,
    [350 / (FS_IN / 2), 1600 / (FS_IN / 2)],
    btype="band"
)
filtered = signal.filtfilt(b, a, samples)

# ----------------------------------------------
# [3] Downsample 20 kHz → 8 kHz (MiRo speaker domain)
# ----------------------------------------------

print("[3/6] Downsampling 20 kHz → 8 kHz (factor 2.5)...")

j = np.arange(len(filtered))
i = np.arange(0, len(filtered), RATIO)
downsampled = np.interp(i, j, filtered)

# ----------------------------------------------
# [4] Fade in/out 5 ms
# ----------------------------------------------

print("[4/6] Applying 5 ms fade...")

fade_samples = int(0.005 * FS_OUT)  # fade in speaker domain
window = np.ones_like(downsampled)

# Fade in
window[:fade_samples] = np.linspace(0, 1, fade_samples)

# Fade out
window[-fade_samples:] = np.linspace(1, 0, fade_samples)

downsampled *= window

# ----------------------------------------------
# [5] Normalize again post-processing
# ----------------------------------------------

print("[5/6] Normalizing processed howl...")

max_amp = np.max(np.abs(downsampled))
downsampled = downsampled / (max_amp + 1e-6)
downsampled *= 0.95  # keep some headroom

# Convert to int16
pcm16 = np.int16(np.clip(downsampled, -1.0, 1.0) * 32767)

# ----------------------------------------------
# [6] Export WAV 16-bit PCM @ 8 kHz
# ----------------------------------------------

print("[6/6] Exporting final processed howl...")

with wave.open(OUTPUT, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(FS_OUT)
    wf.writeframes(pcm16.tobytes())

print("Done! Saved processed howl to:", OUTPUT)
