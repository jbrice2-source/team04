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
        base1 = "/miro01"
        self.currentDirection = ((0,0),0)
        self.goalReached = False
        self.interface = miro.lib.RobotInterface
        self.velocity = TwistStamped()
        self.pos = Pose2D()
<<<<<<< HEAD
        self.soundHeard = False
=======
        self.soundHeard = True
>>>>>>> main
        self.currentInstruction = ""
        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)
        self.mic = rospy.Subscriber(base1 + "/sensors/mics", Int16MultiArray, self.audio_callback,queue_size=1)

    def bandpass(data, edges, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data


    def callback_pose(self, pose):
        if pose != None:
            self.pos = pose
            self.currentAngle = self.pos.theta

    def audio_callback(self,msg):
        audio = np.asarray(msg.data)
        #if 1000 - 1400hz:
        if np.any((robot.bandpass(audio, [1000, 1400], 20000.0))):
            rospy.loginfo("1000-1400 Hz frequency detected in audio input.")
            rospy.loginfo("Turning left")
            self.soundHeard = True
        #if 1500 - 1900hz:
        if np.any((robot.bandpass(audio, [1500, 1900], 20000.0))):
            rospy.loginfo("1500-1900 Hz frequency detected in audio input.")
            rospy.loginfo("Turning right")
            self.soundHeard = True
        #if 2000 - 2400hz:
        if np.any((robot.bandpass(audio, [2000, 2400], 20000.0))):
            rospy.loginfo("2000-2400 Hz frequency detected in audio input.")
            rospy.loginfo("Moving forward")
            self.soundHeard = True
        #if 2500 - 2900hz:
        if np.any((robot.bandpass(audio, [2500, 2900], 20000.0))):
            rospy.loginfo("2500-2900 Hz frequency detected in audio input.")
            rospy.loginfo("Moving backwards")
            self.soundHeard = True
        #if 3000hz - 3400hz:
        if np.any((robot.bandpass(audio, [3000, 3400], 20000.0))):
            rospy.loginfo("3000-3400 Hz frequency detected in audio input.")
            rospy.loginfo("Stopping")
            self.soundHeard = True
    def interpret_sound(self):
        self.currentDirection = ((0,-0.1),radians(90))
    
    def execute_movement(self):
        print(self.currentDirection)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        invert_move = -self.currentDirection[1]
        dists = [(self.pos.theta%(2*np.pi)-invert_move%(2*np.pi))%(2*np.pi),(invert_move%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]
        self.velocity.twist.linear.x = 0.0
        self.velocity.twist.angular.z = 0.0
        self.pub_cmd_vel.publish(self.velocity)
        newpos = np.array([self.pos.x - self.currentDirection[0][0],self.pos.y - self.currentDirection[0][1]])
        cur_pos = np.array([self.pos.x,self.pos.y])
        while np.linalg.norm(newpos-cur_pos) > 0.01:
            cur_pos = np.array([self.pos.x,self.pos.y])
            angle = np.arctan2(*(cur_pos-newpos))
            dists = [(self.pos.theta%(2*np.pi)-invert_move%(2*np.pi))%(2*np.pi),(invert_move%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]
            # print(self.pos.theta,move)
            if min(dists) < 0.1:
                self.velocity.twist.linear.x = 0.1
                self.velocity.twist.angular.z = 0.0
                self.pub_cmd_vel.publish(self.velocity)                
            elif dists[0] > dists[1]:
                self.velocity.twist.angular.z = 1

            else:
                self.velocity.twist.angular.z = -1
            self.pub_cmd_vel.publish(self.velocity)
        self.velocity.twist.linear.x = 0
        self.velocity.twist.angular.z = 0
        self.pub_cmd_vel.publish(self.velocity)
        

if __name__ == "__main__":
    rospy.init_node("lost_nav")
    robot = lostMiro()
    try:
        while robot.goalReached == False:
            while robot.soundHeard == False:
                #listen
                print(f"Instruction {robot.currentInstruction} recieved")
                #move
            robot.interpret_sound()
            robot.execute_movement()
            robot.soundHeard = False

            
    except KeyboardInterrupt:
        print("exiting...")


