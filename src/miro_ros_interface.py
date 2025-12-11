#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   MiRo ROS Interface (minimal)
#
#   Lightweight wrapper around MiRo ROS topics for BioRescue Phase 1.
#
#   Uses:
#       /miro/sensors/mics           (std_msgs/Int16MultiArray)
#       /miro/control/tone           (std_msgs/UInt16MultiArray)
#       /miro/control/cosmetic_joints (std_msgs/Float32MultiArray)
#       /miro/control/kinematic_joints (sensor_msgs/JointState, not used yet)
#

import numpy as np
import rospy

from std_msgs.msg import Int16MultiArray, Float32MultiArray, UInt16MultiArray
from sensor_msgs.msg import JointState

# MiRo mic config: 4 channels × 500 samples
BLOCK_SAMPLES = 500


class MiRoROSInterface(object):
    """
    Minimal ROS interface for MiRo used by BioRescue Phase 1.

    Provides:
        - self.mics: latest microphone block as dict of 4 × N int16 arrays
        - play_tone(freq, volume): publish to /miro/control/tone
        - set_cosmetic_neutral(), set_cosmetic_lost(): simple postures
    """

    def __init__(self):

        # Latest mic block (dict of numpy arrays or None)
        self.mics = None

        # Subscribers

        self.sub_mics = rospy.Subscriber(
            "/miro/sensors/mics",
            Int16MultiArray,
            self.callback_mics,
            queue_size=1
        )

        # Publishers

        self.pub_tone = rospy.Publisher(
            "/miro/control/tone",
            UInt16MultiArray,
            queue_size=0
        )

        self.pub_cos = rospy.Publisher(
            "/miro/control/cosmetic_joints",
            Float32MultiArray,
            queue_size=0
        )

        self.pub_kin = rospy.Publisher(
            "/miro/control/kinematic_joints",
            JointState,
            queue_size=0
        )

        # Cosmetic joint baseline (same length as client_demo: 6 floats)
        # This is just a neutral-looking pose.
        self.cosmetic_state = np.array(
            [0.0, 0.5, 0.5, 0.5, 0.2, 0.0], dtype=np.float32
        )

    #   Microphone callback
    def callback_mics(self, msg: Int16MultiArray):
        """
        Convert Int16MultiArray (flattened 4×N) into 4 separate channels.
        """

        data = np.array(msg.data, dtype=np.int16)

        if data.size % 4 != 0:
            return

        n = data.size // 4
        block = data.reshape((4, n))

        self.mics = {
            "left":   block[0, :],
            "right":  block[1, :],
            "centre": block[2, :],
            "tail":   block[3, :],
        }

    #   Tone helper
    def play_tone(self, freq_hz: float, volume: int):
        """
        Publish a tone command to /miro/control/tone.

        /miro/control/tone type: std_msgs/UInt16MultiArray
        Convention (from client_demo):
            data[0] = frequency (Hz)
            data[1] = volume    (0–255)
            data[2] = flags     (1 = simple tone)
        """

        msg = UInt16MultiArray()

        v = max(0, min(255, int(volume)))
        f = max(0, int(freq_hz))

        msg.data = [f, v, 1]
        self.pub_tone.publish(msg)

    #   Cosmetic joints helpers
    def set_cosmetic_neutral(self):
        """
        Neutral cosmetic pose – essentially whatever client_demo uses
        as baseline.
        """
        msg = Float32MultiArray()
        msg.data = self.cosmetic_state.tolist()
        self.pub_cos.publish(msg)

    def set_cosmetic_lost(self):
        """
        Slightly "drooped" body posture to indicate LOST state.
        You can tweak values later if you want a different look.
        """
        state = self.cosmetic_state.copy()
        # index 0 is often used for tail/body droop in demos – make it negative
        state[0] = -0.3

        msg = Float32MultiArray()
        msg.data = state.tolist()
        self.pub_cos.publish(msg)
