#!/usr/bin/env python3
import rospy
import time
import numpy as np

from std_msgs.msg import UInt16MultiArray, Int16MultiArray


class BrainstemPhase1(object):

    # ==============================================================
    # INITIALISATION
    # ==============================================================

    def __init__(self):

        rospy.init_node("miro_phase1_brainstem")

        # ----------------------------------------------------------
        # ROLE HANDLING
        # ----------------------------------------------------------
        # If _role:=lost is provided → it is LOST.
        # If nothing provided → starts as HELPER_CANDIDATE.
        role_param = rospy.get_param("~role", "auto")

        if role_param == "lost":
            self.physical_role = "lost"
            self.state = "LOST_ACTIVE_HEARING"
            rospy.loginfo("[INIT] Starting as LOST MiRo → LOST_ACTIVE_HEARING")
        else:
            self.physical_role = "helper_candidate"
            self.state = "HELPER_PASSIVE_LISTENING"
            rospy.loginfo("[INIT] Starting as HELPER CANDIDATE → HELPER_PASSIVE_LISTENING")

        # ----------------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------------
        self.sample_rate = 20000          # Hz – approx mic rate
        self.fft_window = 2048            # samples per FFT

        # Thresholds (you will tune these experimentally)
        self.howl_thresh = 1e7            # 800 Hz
        self.confirm_thresh = 8e6         # 500 Hz
        self.rumble_thresh = 5e6          # 200 Hz

        # Lost MiRo distress timing
        self.start_time = time.time()
        self.distress_delay = 2.0         # sec before lost enters distress call
        self.howl_interval = 5.0          # sec between howls
        self.last_howl_time = 0.0

        # Flags set by FFT
        self.detected_howl = False
        self.detected_confirm = False
        self.detected_rumble = False

        # Phase completion flags
        self.phase1_complete = False
        self.phase2_ready = False
        self.local_ping_detected = False

        # Microphone buffer
        self.mic_buffer = []

        # ----------------------------------------------------------
        # PUBLISHER – TONE OUTPUT
        # ----------------------------------------------------------
        self.pub_tone = rospy.Publisher(
            "/miro/control/tone", UInt16MultiArray, queue_size=1
        )

        # ----------------------------------------------------------
        # SUBSCRIBER – MICS (ACTIVE + PASSIVE LISTENING)
        # ----------------------------------------------------------
        rospy.Subscriber("/miro/sensors/mics", Int16MultiArray, self.cb_mics)

        rospy.loginfo("[INIT] Hearing engine active (mic subscriber + FFT)")

        # ----------------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------------
        self.loop()

    # ==============================================================
    # MIC CALLBACK
    # ==============================================================

    def cb_mics(self, msg: Int16MultiArray):
        data = np.array(msg.data, dtype=np.int16)
        ch0 = data[0::4]                   # use channel 0
        self.mic_buffer.extend(ch0.tolist())

        if len(self.mic_buffer) >= self.fft_window:
            chunk = np.array(self.mic_buffer[:self.fft_window])
            self.mic_buffer = self.mic_buffer[self.fft_window:]
            self.process_audio_chunk(chunk)

    # ==============================================================
    # AUDIO PROCESSING (FFT DETECTION)
    # ==============================================================

    def process_audio_chunk(self, chunk):
        fft_vals = np.fft.rfft(chunk)
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / self.sample_rate)
        mags = np.abs(fft_vals)

        def band_energy(center, width):
            mask = (freqs >= center - width / 2.0) & (freqs <= center + width / 2.0)
            return np.sum(mags[mask])

        # Energy bands
        howl_energy = band_energy(800, 200)       # 700–900 Hz
        confirm_energy = band_energy(500, 200)    # 400–600 Hz
        rumble_energy = band_energy(200, 200)     # 100–300 Hz

        # ======================================================
        # LOST MIRO LISTENING
        # ======================================================
        if self.physical_role == "lost":
            # Lost ONLY listens for confirmation tone
            if confirm_energy > self.confirm_thresh:
                if not self.detected_confirm:
                    rospy.loginfo("[AUDIO LOST] Confirmation tone detected.")
                self.detected_confirm = True
            return

        # ======================================================
        # HELPER CANDIDATE LISTENING
        # ======================================================
        if self.state == "HELPER_PASSIVE_LISTENING":
            # Waiting for howl
            if howl_energy > self.howl_thresh:
                if not self.detected_howl:
                    rospy.loginfo("[AUDIO HELPER] Distress howl detected.")
                self.detected_howl = True

        # ======================================================
        # PROMOTED HELPER LISTENING
        # ======================================================
        if self.state in ["HELPER_ACTIVE_HEARING", "HELPER_WAIT_FOR_PINGS"]:
            if rumble_energy > self.rumble_thresh:
                if not self.detected_rumble:
                    rospy.loginfo("[AUDIO HELPER] Rumble beacon detected.")
                self.detected_rumble = True

    # ==============================================================
    # TONE OUTPUT FUNCTIONS
    # ==============================================================

    def _tone(self, freq, dur_ms, amp):
        msg = UInt16MultiArray()
        msg.data = [int(freq), int(dur_ms), int(amp)]
        self.pub_tone.publish(msg)

    def play_howl(self):
        self._tone(800, 1000, 350)
        rospy.loginfo("[LOST] HOWL emitted.")

    def play_confirmation(self):
        self._tone(500, 300, 400)
        rospy.loginfo("[HELPER] CONFIRMATION emitted.")

    def play_rumble(self):
        self._tone(200, 500, 250)
        rospy.loginfo("[LOST] RUMBLE beacon emitted.")

    # ==============================================================
    # MAIN STATE MACHINE
    # ==============================================================

    def loop(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():

            now = time.time()

            # ------------------------------------------------------
            # LOST MIRO STATE LOGIC
            # ------------------------------------------------------
            if self.physical_role == "lost":

                if self.state == "LOST_ACTIVE_HEARING":
                    if now - self.start_time >= self.distress_delay:
                        rospy.loginfo("[LOST] → DISTRESS_MODE")
                        self.state = "LOST_DISTRESS_MODE"
                        self.last_howl_time = now

                elif self.state == "LOST_DISTRESS_MODE":
                    # periodic howls
                    if now - self.last_howl_time >= self.howl_interval:
                        self.play_howl()
                        self.last_howl_time = now

                    # wait for confirmation
                    if self.detected_confirm:
                        rospy.loginfo("[LOST] Confirmation received → LOST_BEACON")
                        self.state = "LOST_BEACON"
                        self.phase1_complete = True
                        self.phase2_ready = True

                elif self.state == "LOST_BEACON":
                    # Lost emits continuous rumble until navigation engine takes over
                    self.play_rumble()

            # ------------------------------------------------------
            # HELPER (CANDIDATE + PROMOTED) LOGIC
            # ------------------------------------------------------
            else:

                if self.state == "HELPER_PASSIVE_LISTENING":
                    if self.detected_howl:
                        rospy.loginfo("[HELPER] PASSIVE → ACTIVE (helper state activated)")
                        self.state = "HELPER_ACTIVE_HEARING"
                        self.play_confirmation()
                        rospy.loginfo("[HELPER] → WAIT_FOR_PINGS")
                        self.state = "HELPER_WAIT_FOR_PINGS"

                elif self.state == "HELPER_WAIT_FOR_PINGS":
                    if self.detected_rumble:
                        rospy.loginfo("[HELPER] RUMBLE detected → PHASE1 COMPLETE")
                        self.state = "HELPER_PHASE1_COMPLETE"
                        self.phase1_complete = True
                        self.local_ping_detected = True
                        self.phase2_ready = True

                elif self.state == "HELPER_PHASE1_COMPLETE":
                    # Just keep hearing on for Phase 2
                    pass

            rate.sleep()


if __name__ == "__main__":
    try:
        BrainstemPhase1()
    except rospy.ROSInterruptException:
        pass
