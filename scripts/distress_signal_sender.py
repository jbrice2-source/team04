#!/usr/bin/env python3
#
# Distress howl sender for lost MiRo
# - Uses same buffer logic as official "echo" example
# - Loads a 20 kHz mono WAV, down-samples to ~8 kHz for playback
# - Streams via /<MIRO_ROBOT_NAME>/control/stream
# - No beeps, no digit encoding – just the howl, repeated
#

import rospy
from std_msgs.msg import UInt16MultiArray, Int16MultiArray

import time
import os
import numpy as np
import wave

import miro2 as miro

# ----------------------------------------------------
# STREAMING CONSTANTS (from official example)
# ----------------------------------------------------

BUFFER_STUFF_SAMPLES = 4000           # target buffer fill (in playback samples)
MAX_STREAM_MSG_SIZE   = (4096 - 48)   # max samples per message
BUFFER_MARGIN         = 1000
BUFFER_MAX            = BUFFER_STUFF_SAMPLES + BUFFER_MARGIN
BUFFER_MIN            = BUFFER_STUFF_SAMPLES - BUFFER_MARGIN

# Microphone sample rate (your WAV is 20 kHz)
MIC_SAMPLE_RATE   = 20000
# Effective speaker playback rate used in CQR echo (approx 8 kHz)
SPKR_SAMPLE_RATE  = 8000.0            # float to match 20k / 2.5

# ----------------------------------------------------
# HOWL CONFIG
# ----------------------------------------------------

HOWL_WAV_PATH        = "/root/mdk/catkin_ws/src/team04/sound_files/distress_howl_test.wav"
HOWL_VOLUME_PERCENT  = 5.0           # 0–100 %
CYCLE_PAUSE_SECONDS  = 2.0           # pause between howls


class DistressHowlClient:

    def __init__(self):

        # Connect to the robot (sets up topics, etc.)
        self.interface = miro.lib.RobotInterface()

        # Buffer state from /sensors/stream
        self.buffer_space = 0
        self.buffer_total = 0
        self.buffer_stuff = 0

        # Playback index
        self.playsamp = 0

        # Load and pre-process howl
        self.outbuf = self.load_and_downsample_howl(HOWL_WAV_PATH, HOWL_VOLUME_PERCENT)

        # Resolve robot name and set up topics
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Publisher: speaker stream
        topic = topic_base_name + "/control/stream"
        rospy.loginfo("Publish: %s", topic)
        self.pub_stream = rospy.Publisher(topic, Int16MultiArray, queue_size=0)

        # Subscriber: buffer feedback
        topic = topic_base_name + "/sensors/stream"
        rospy.loginfo("Subscribe: %s", topic)
        self.sub_stream = rospy.Subscriber(
            topic,
            UInt16MultiArray,
            self.callback_stream,
            queue_size=1,
            tcp_nodelay=True
        )

    # ------------------------------------------------
    # WAV LOADING + DOWNSAMPLING
    # ------------------------------------------------

    def load_and_downsample_howl(self, path, volume_percent):
        rospy.loginfo("Loading howl WAV from: %s", path)

        wf = wave.open(path, 'rb')

        num_frames  = wf.getnframes()
        sample_rate = wf.getframerate()
        sampwidth   = wf.getsampwidth()
        nchannels   = wf.getnchannels()

        # Sanity checks
        assert sampwidth == 2, "WAV must be 16-bit PCM"
        assert nchannels == 1, "WAV must be mono"
        assert sample_rate == MIC_SAMPLE_RATE, f"WAV must be {MIC_SAMPLE_RATE} Hz"

        frames = wf.readframes(num_frames)
        wf.close()

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

        # Apply volume scaling
        gain = max(0.0, min(1.0, float(volume_percent) / 100.0))
        audio *= gain

        # Downsample 20 kHz -> ~8 kHz using same pattern as echo example (factor 2.5)
        ratio = MIC_SAMPLE_RATE / SPKR_SAMPLE_RATE  # = 2.5
        j = np.arange(0, audio.shape[0])            # original index
        i = np.arange(0, audio.shape[0], ratio)     # target index positions
        audio_ds = np.interp(i, j, audio)

        # Clip + convert to int16
        audio_ds = np.clip(audio_ds, -32768, 32767).astype(np.int16)

        dur_in  = audio.shape[0] / float(MIC_SAMPLE_RATE)
        dur_out = audio_ds.shape[0] / SPKR_SAMPLE_RATE

        rospy.loginfo(
            "Loaded howl (mic domain): %d samples (%.2f s at %.0f Hz)",
            audio.shape[0], dur_in, MIC_SAMPLE_RATE
        )
        rospy.loginfo(
            "Downsampled howl (speaker domain): %d samples (%.2f s at %.0f Hz)",
            audio_ds.shape[0], dur_out, SPKR_SAMPLE_RATE
        )

        return audio_ds

    # ------------------------------------------------
    # CALLBACK
    # ------------------------------------------------

    def callback_stream(self, msg):
        """
        As in the official example:
        /sensors/stream reports [buffer_space, buffer_total]
        """
        self.buffer_space = msg.data[0]
        self.buffer_total = msg.data[1]
        self.buffer_stuff = self.buffer_total - self.buffer_space

    # ------------------------------------------------
    # STREAM HOWL ONCE
    # ------------------------------------------------

    def stream_howl_once(self):
        """
        Streams the pre-processed howl buffer using the same
        buffer-stuffing algorithm as the official "echo" demo.
        """
        outbuf = self.outbuf
        self.playsamp = 0

        # Wait until we have valid buffer info from /sensors/stream
        rospy.loginfo("Waiting for buffer status from /sensors/stream...")
        while not rospy.core.is_shutdown() and self.buffer_total == 0:
            time.sleep(0.02)

        rospy.loginfo("Starting howl streaming...")

        start_time = time.time()

        while not rospy.core.is_shutdown():
            # If buffer is low, stuff more audio
            if self.buffer_stuff < BUFFER_MIN:

                if self.playsamp == 0:
                    rospy.loginfo("Stuffing output buffer with howl audio...")

                # Target amount to send
                n_samp = BUFFER_MAX - self.buffer_stuff

                # Limit by receiver buffer space
                n_samp = min(n_samp, self.buffer_space)

                # Limit by remaining samples
                n_samp = min(n_samp, outbuf.shape[0] - self.playsamp)

                # Limit by maximum message size
                n_samp = min(n_samp, MAX_STREAM_MSG_SIZE)

                if n_samp <= 0:
                    # Nothing left to send or no space: we’re done
                    break

                # Prepare slice
                end_idx = self.playsamp + n_samp
                spkrdata = outbuf[self.playsamp:end_idx]
                self.playsamp = end_idx

                # Publish to speaker
                msg = Int16MultiArray()
                msg.data = [int(i) for i in spkrdata]
                self.pub_stream.publish(msg)

                # Fake update until we get real feedback again
                self.buffer_stuff = BUFFER_MIN

                # Finished sending all samples?
                if self.playsamp >= outbuf.shape[0]:
                    rospy.loginfo("Howl playback samples exhausted.")
                    break

            # Sleep exactly as in the official client
            time.sleep(0.02)

        elapsed  = time.time() - start_time
        expected = outbuf.shape[0] / SPKR_SAMPLE_RATE

        rospy.loginfo(
            "Howl playback complete. Expected ~%.2fs, actual ~%.2fs",
            expected, elapsed
        )

    # ------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------

    def loop(self):
        rospy.loginfo("Distress howl client loop started.")
        while not rospy.core.is_shutdown():
            self.stream_howl_once()
            rospy.loginfo(
                "Howl cycle complete, waiting %.1f seconds...",
                CYCLE_PAUSE_SECONDS
            )
            t_end = time.time() + CYCLE_PAUSE_SECONDS
            while time.time() < t_end and not rospy.core.is_shutdown():
                time.sleep(0.1)


if __name__ == "__main__":
    client = DistressHowlClient()
    client.loop()
