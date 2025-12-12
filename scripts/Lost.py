#!/usr/bin/python3

import os
from glob import glob
import rospy
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
from geometry_msgs.msg import TwistStamped, Pose2D
from sensor_msgs.msg import JointState, CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import math
import heapq
import miro2 as miro
import time
from math import pi
from math import radians
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry


from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

class lostMiro:
    def __init__(self):
        base1 = "/miro"
        self.interface = miro.lib.RobotInterface()
        self.velocity = TwistStamped()
        self.listening = True
        self.audio = rospy.Subscriber(base1 + "/sensors/mics", Int16MultiArray, self.audio_callback, queue_size=1)
        self.audio_data = None
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, math.radians(20.0), 0.0, 0.0]
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]

        self.timer = rospy.Timer(rospy.Duration(0.1), self.detect_audio)

    
    def bandpass(self, data, edges, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data
    
    def plot_band(self, band_signal, edges, label):
        # Number of samples
        samples = len(band_signal)

        spectrum = np.fft.rfft(band_signal)
        freqs = np.fft.rfftfreq(samples, d=1.0 / 20000.0)

        plt.figure(figsize=(8, 4))
        plt.plot(freqs, np.abs(spectrum))
        plt.title(f"Detected {label} ({edges[0]}-{edges[1]} Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(edges)  # zoom into the band of interest
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)

    def detect_audio(self, *args):
        self.interface.msg_cos_joints.set(self.cos_joints,10)
        self.interface.msg_kin_joints.set(self.kin_joints,10)
        if self.audio_data is None:
            return
        threshhold = 900
        audio = self.audio_data[:]

        ranges = [[400,560],
                  [720,880],
                  [1040,1200],
                  [1360,1520],
                  [1680,1840],
                  ]
        commands = [self.found,
                    self.turn_left,
                    self.turn_right,
                    self.forward,
                    self.stop,
                    ]
        channels = np.zeros((len(ranges),audio.shape[0]))
        for i,n in enumerate(ranges):
            channels[i] = self.bandpass(audio, n, 20000.0)

        if not self.listening:
            return
        amplitude = np.max(np.count_nonzero(abs(channels)>threshhold,axis=1))
        channel = np.argmax(np.count_nonzero(abs(channels)>threshhold,axis=1))
        rospy.loginfo(f"{amplitude} {channel} {np.max(channels[channel])}")


        if amplitude > 200:
            rospy.loginfo(f"{ranges[channel][0]} - {ranges[channel][1]} Hz frequency detected in audio input.")
            rospy.loginfo(commands[channel].__name__)
            self.listening = False
            commands[channel]()


    def turn_left(self): 
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = 1.0
        self.interface.msg_cmd_vel.set(self.velocity,0.4)
        rospy.sleep(0.1)
        self.listening = True  
    def turn_right(self):
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = -1.0
        self.interface.msg_cmd_vel.set(self.velocity,0.4)
        rospy.sleep(0.1)
        self.listening = True  
    def forward(self):
        self.velocity.twist.linear.x = 0.12
        self.velocity.twist.angular.z = 0.0
        self.interface.msg_cmd_vel.set(self.velocity,0.4)
        rospy.sleep(0.1)
        self.listening = True  

    def stop(self):
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = 0.0
        self.interface.msg_cmd_vel.set(self.velocity,0.4)
        rospy.sleep(0.1)
        self.listening = True  
    def found(self):
        self.listening = True  

    def audio_callback(self,msg):
        audio = np.asarray(msg.data)
        self.audio_data = audio



if __name__ == "__main__":
    robot = lostMiro()
    if(rospy.get_node_uri()):
        pass
    else:
        rospy.init_node("lost_nav")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("exiting...")


