#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   BioRescue – Phase 1 (ROS version)
#
#   Structured acoustic signalling between two MiRo robots over ROS.
#
#   Roles:
#       HELPER_CANDIDATE   – demo mode + passive hearing
#       LOST_DISTRESS      – distress loop (sweep + triple pulses)
#       LOST_WAIT          – rumble beacon after confirmation
#       HELPER_CONFIRM     – confirmation loop after distress detected
#       HELPER_READY       – ready for Phase 2 navigation
#

import threading
import numpy as np
import rospy

from biorescue_detect_audio_engine import DetectAudioEngine
from biorescue_spectrogram import SpectrogramEngine
from miro_ros_interface import MiRoROSInterface, BLOCK_SAMPLES

# Audio configuration (matches MiRo mic stream: 40 blocks/sec × 500 samples)
BLOCK_RATE = 40.0
SAMPLE_RATE = BLOCK_RATE * BLOCK_SAMPLES


class BioRescuePhase1(object):

    STATE_HELPER_CANDIDATE = "HELPER_CANDIDATE"
    STATE_LOST_DISTRESS    = "STATE_LOST_DISTRESS"
    STATE_LOST_WAIT        = "STATE_LOST_WAIT"
    STATE_HELPER_CONFIRM   = "STATE_HELPER_CONFIRM"
    STATE_HELPER_READY     = "STATE_HELPER_READY"

    #   Init
    def __init__(self):
        rospy.init_node("biorescue_phase1")

        # ROS interface (tone, cosmetic joints, mics)
        self.miro = MiRoROSInterface()

        # Parameters
        self.start_as_lost = rospy.get_param("~is_lost", False)
        self.distress_volume = rospy.get_param("~distress_volume", 200)
        self.confirm_volume  = rospy.get_param("~confirm_volume", 180)
        self.rumble_volume   = rospy.get_param("~rumble_volume", 180)
        self.distress_rate_hz = rospy.get_param("~distress_rate_hz", 0.3)

        # Audio engines
        self.audio_engine = DetectAudioEngine(
            sample_rate=SAMPLE_RATE,
            block_size=BLOCK_SAMPLES
        )
        self.spectro = SpectrogramEngine(
            sample_rate=SAMPLE_RATE,
            block_size=BLOCK_SAMPLES,
            history_sec=10.0
        )

        # Internal state
        self.state_lock = threading.Lock()
        self.state = self.STATE_HELPER_CANDIDATE
        self.shutdown_flag = False

        # Startup behaviour
        self.startup()

        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.shutdown_flag = True
            self.spectro.shutdown()

    #   State helpers
    def set_state(self, s):
        with self.state_lock:
            self.state = s

    def get_state(self):
        with self.state_lock:
            return self.state

    #   Startup
    def startup(self):
        if self.start_as_lost:
            rospy.loginfo("BioRescue Phase 1: starting as LOST robot")
            self.enter_lost_distress()
        else:
            rospy.loginfo("BioRescue Phase 1: starting as HELPER_CANDIDATE")
            self.enter_helper_candidate()

        # Passive hearing loop
        th = threading.Thread(target=self.hearing_loop)
        th.daemon = True
        th.start()

    #   HELPER_CANDIDATE
    def enter_helper_candidate(self):
        """Helper candidate – demo mode + passive hearing."""
        self.set_state(self.STATE_HELPER_CANDIDATE)
        self.miro.set_cosmetic_neutral()

    #   LOST – Distress mode
    def enter_lost_distress(self):
        """LOST robot enters distress loop."""
        self.set_state(self.STATE_LOST_DISTRESS)
        self.miro.set_cosmetic_lost()

        th = threading.Thread(target=self.distress_loop)
        th.daemon = True
        th.start()

    def distress_loop(self):
        """Repeating distress call."""
        rate = rospy.Rate(self.distress_rate_hz)

        while not rospy.is_shutdown() and not self.shutdown_flag:
            if self.get_state() != self.STATE_LOST_DISTRESS:
                return
            self.emit_distress_call()
            rate.sleep()

    def emit_distress_call(self):
        """
        Distress call:
            1. Sweep (400→900→500 Hz)
            2. Triple pulses (1500 Hz)
            3. Listen gap
        """
        vol = self.distress_volume

        sweep_freqs = [400, 550, 700, 850, 900, 750, 600, 500]
        dur = 0.05
        for f in sweep_freqs:
            self.miro.play_tone(f, vol)
            rospy.sleep(dur)

        rospy.sleep(0.04)

        for _ in range(3):
            self.miro.play_tone(1500, vol)
            rospy.sleep(0.04)
            rospy.sleep(0.05)

        rospy.sleep(0.12)

    #   LOST – Wait / rumble beacon
    def enter_lost_wait(self):
        """LOST robot has heard confirmation – switch to rumble."""
        self.set_state(self.STATE_LOST_WAIT)

        th = threading.Thread(target=self.rumble_loop)
        th.daemon = True
        th.start()

    def rumble_loop(self):
        """Continuous low-frequency rumble beacon (~120 Hz)."""
        freq = 120.0
        vol = self.rumble_volume
        dur = 0.12

        while not rospy.is_shutdown() and not self.shutdown_flag:
            if self.get_state() != self.STATE_LOST_WAIT:
                return
            self.miro.play_tone(freq, vol)
            rospy.sleep(dur)

    #   HELPER – Confirmation mode
    def enter_helper_confirm(self):
        """Helper emits confirmation calls until rumble is detected."""
        self.set_state(self.STATE_HELPER_CONFIRM)

        th = threading.Thread(target=self.confirmation_loop)
        th.daemon = True
        th.start()

    def confirmation_loop(self):
        """Repeating confirmation call."""
        rate = rospy.Rate(1.5)

        while not rospy.is_shutdown() and not self.shutdown_flag:
            if self.get_state() != self.STATE_HELPER_CONFIRM:
                return
            self.emit_confirmation_call()
            rate.sleep()

    def emit_confirmation_call(self):
        """
        Confirmation pattern:
            1. Descending chirp
            2. Double pulses (1400 Hz)
            3. Short gap
        """
        vol = self.confirm_volume

        chirp_freqs = [900, 750, 600, 450]
        dur = 0.04
        for f in chirp_freqs:
            self.miro.play_tone(f, vol)
            rospy.sleep(dur)

        rospy.sleep(0.03)

        for _ in range(2):
            self.miro.play_tone(1400, vol)
            rospy.sleep(0.03)
            rospy.sleep(0.04)

        rospy.sleep(0.08)

    #   HELPER_READY
    def enter_helper_ready(self):
        """Helper has locked onto rumble beacon and is ready for Phase 2."""
        self.set_state(self.STATE_HELPER_READY)
        rospy.loginfo("BioRescue Phase 1: HELPER_READY – navigation can begin")

    #   Passive hearing loop
    def hearing_loop(self):
        """
        Continuous passive hearing:
            HELPER_CANDIDATE → listen for distress
            LOST_DISTRESS    → listen for confirmation
            HELPER_CONFIRM   → listen for rumble
        """
        rate = rospy.Rate(BLOCK_RATE)

        while not rospy.is_shutdown() and not self.shutdown_flag:

            mics = self.miro.mics

            if mics is not None:
                block = np.vstack([
                    mics["left"],
                    mics["right"],
                    mics["centre"],
                    mics["tail"],
                ])

                if block.shape[1] != BLOCK_SAMPLES:
                    rate.sleep()
                    continue

                # Visualise + detect
                self.spectro.add_block(block)
                self.audio_engine.process_block(block)

                state = self.get_state()

                if state == self.STATE_HELPER_CANDIDATE:
                    if self.audio_engine.detected_distress():
                        rospy.loginfo("BioRescue Phase 1: distress detected → HELPER_CONFIRM")
                        self.enter_helper_confirm()

                elif state == self.STATE_LOST_DISTRESS:
                    if self.audio_engine.detected_confirmation():
                        rospy.loginfo("BioRescue Phase 1: confirmation detected → LOST_WAIT")
                        self.enter_lost_wait()

                elif state == self.STATE_HELPER_CONFIRM:
                    if self.audio_engine.detected_rumble():
                        rospy.loginfo("BioRescue Phase 1: rumble detected → HELPER_READY")
                        self.enter_helper_ready()

            rate.sleep()


#   Main
if __name__ == "__main__":
    BioRescuePhase1()
