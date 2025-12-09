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
        self.currentDirection = ((0,0),0)
        self.currentAngle = 0
        self.goalReached = False
        self.interface = miro.lib.RobotInterface
        self.velocity = TwistStamped()
        self.odom = Odometry()
        self.listening = True
        self.currentInstruction = ""
        self.posx = 1
        self.posy = 2
        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.odeometerySub = rospy.Subscriber(base1 + "/sensors/odom",
            Odometry, self.callback_odom, queue_size=1, tcp_nodelay=True)
        self.audio = rospy.Subscriber(base1 + "/sensors/mics", Int16MultiArray, self.audio_callback, queue_size=1)
        self.audio_data = None

        self.timer = rospy.Timer(rospy.Duration(0.1), self.detect_audio)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.execute_movement)

    
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

        if self.audio_data is None:
            return
        threshhold = 500
        audio = self.audio_data
        # if 3000hz - 3400hz:

        channels = np.array([
                self.bandpass(audio, [700, 900], 20000.0),
                self.bandpass(audio, [1100, 1300], 20000.0),
                self.bandpass(audio, [1500, 1700], 20000.0),
                self.bandpass(audio, [1900, 2100], 20000.0),
                self.bandpass(audio, [2300, 2500], 20000.0),
                self.bandpass(audio, [2700, 2900], 20000.0),
                self.bandpass(audio, [3100, 3300], 20000.0),
                self.bandpass(audio, [3500, 3700], 20000.0),
                self.bandpass(audio, [3900, 4100], 20000.0)
        ])

        if np.mean(abs(channels[-1])) > threshhold:
            rospy.loginfo("3900-4100 Hz frequency detected in audio input.")
            rospy.loginfo("stop")     
            self.currentDirection = ((0,0),radians(0)) # stop?
            rospy.sleep(1)
            self.listening = True
            return

        if not self.listening:
            return

        if np.mean(abs(channels[0])) > threshhold:
            rospy.loginfo("700 - 900 Hz frequency detected in audio input.")
            rospy.loginfo("north")
            self.currentDirection = ((0.5,0),radians(180))
            self.plot_band(fft(abs(channels[0])), [0, 1600], "700-900 Hz")
            print(self.listening)       
        # if 1000 - 1200hz:
        elif np.mean(abs(channels[1])) > threshhold:
            rospy.loginfo("1100-1300 Hz frequency detected in audio input.")
            rospy.loginfo("east")
            self.currentDirection = ((0,0.5),radians(90))
            print(self.listening)       
        # if 2000 - 2400hz:
        elif np.mean(abs(channels[2])) > threshhold:
            rospy.loginfo("1500 - 1700 Hz frequency detected in audio input.")
            rospy.loginfo("south")
            self.currentDirection = ((-0.5,0),radians(360))          
        # if 2500 - 2900hz: 
        elif np.mean(abs(channels[3])) > threshhold:
            rospy.loginfo("1900-2100 Hz frequency detected in audio input.")
            rospy.loginfo("west")
            self.currentDirection = ((0,-0.5),radians(270)) 
        # if 3000hz - 3400hz:
        elif np.mean(abs(channels[4])) > threshhold:
            rospy.loginfo("2300-2500 Hz frequency detected in audio input.")
            rospy.loginfo("north east")
            self.currentDirection = ((-0.5,-0.5),radians(45)) 
        # if 2000 - 2400hz:
        elif np.mean(abs(channels[5])) > threshhold:
            rospy.loginfo("2700-2900 Hz frequency detected in audio input.")
            rospy.loginfo("north west")
            self.currentDirection = ((-0.5,-0.5),radians(315)) 
        # if 2500 - 2900hz:     
        elif np.mean(abs(channels[6])) > threshhold:
            rospy.loginfo("3100-3300 Hz frequency detected in audio input.")
            rospy.loginfo("south east")
            self.currentDirection = ((0.5,0.5),radians(135)) 
        # if 3000hz - 3400hz:
        elif np.mean(abs(channels[7])) > threshhold:
            rospy.loginfo("3500-3700 Hz frequency detected in audio input.")
            rospy.loginfo("south west")
            self.currentDirection = (((0.5,-0.5),radians(225)))
 
        else: return
        self.listening = False
        # self.execute_movement()

    def callback_odom(self, odometry):
        if odometry != None:
            self.posx = odometry.pose.pose.position.x
            self.posy = odometry.pose.pose.position.y
            orientation_q = odometry.pose.pose.orientation
            orientation_list = [orientation_q.x,orientation_q.y,orientation_q.z,orientation_q.w]
            _,_,yaw = euler_from_quaternion(orientation_list)
            self.currentAngle = yaw

    def audio_callback(self,msg):
        audio = np.asarray(msg.data)
        self.audio_data = audio

    def execute_movement(self, *args):
        if self.listening: return
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        invert_move = -self.currentDirection[1]
        dists = [(self.currentAngle%(2*np.pi)-invert_move%(2*np.pi))%(2*np.pi),(invert_move%(2*np.pi)-self.currentAngle%(2*np.pi))%(2*np.pi)]
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = 0.0
        self.pub_cmd_vel.publish(self.velocity)
        newpos = np.array([self.posx - self.currentDirection[0][0],self.posy - self.currentDirection[0][1]])
        cur_pos = np.array([self.posx,self.posy])
        print(self.currentDirection)
        if np.linalg.norm(newpos-cur_pos) < 0.1:
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0
            self.pub_cmd_vel.publish(self.velocity)
            rospy.sleep(1)
            self.listening = True
        else:
            cur_pos = np.array([self.posx,self.posy])
            angle = np.arctan2(*(cur_pos-newpos))
            dists = [(self.currentAngle%(2*np.pi)-invert_move%(2*np.pi))%(2*np.pi),(invert_move%(2*np.pi)-self.currentAngle%(2*np.pi))%(2*np.pi)]
            print(newpos, cur_pos, angle)
            if min(dists) < 0.1:
                self.velocity.twist.linear.x = 0.1
                self.velocity.twist.angular.z = 0.0
                self.pub_cmd_vel.publish(self.velocity)                
            elif dists[0] > dists[1]:
                # print("iogs")
                self.velocity.twist.angular.z = 1.0
            else:
                # print("ioadhs")
                self.velocity.twist.angular.z = -1.0
            self.pub_cmd_vel.publish(self.velocity)


if __name__ == "__main__":
    rospy.init_node("lost_nav")
    robot = lostMiro()
    # mic = rospy.Subscriber("miro01/sensors/mics", Int16MultiArray, audio_callback, queue_size=1)
    try:
        # while robot.goalReached == False:
        #     if robot.listening == False:
        #         print("listening2 ", robot.listening)
        #         robot.execute_movement()
        #         robot.listening = True
        #     else:
        rospy.spin()
    except KeyboardInterrupt:
        print("exiting...")


