#!/usr/bin/python3

import os
import rospy
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
from geometry_msgs.msg import TwistStamped, Pose2D
from sensor_msgs.msg import JointState, CompressedImage

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import math
import miro2 as miro
import time


from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import uuid

droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)

class Explore:
    
    def __init__(self):
        base1 = "/miro01"
        # base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        # self.velocity# = TwistStamped()
        
        self.pos = Pose2D()
        self.cur_target = None
        self.time1 = time.time_ns()
        self.add_dist = 0.0
        self.camera_interval  = time.time_ns()
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(50.0), 0.0, 0.0]
        self.input_package = None
        self.map = np.zeros((100,100))
        # self.cos_joints = Float32MultiArray()
        # self.cos_joints.data = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0)
        # self.pub_cmd_vel2 = rospy.Publisher(base2 + "/control/cmd_vel", TwistStamped, queue_size=0)
        # self.pub_cos = rospy.Publisher(base1 + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        # self.pub_illum = rospy.Publisher(basename + "/control/illum", UInt32MultiArray, queue_size=0)
        self.pub_kin = rospy.Publisher(base1 + "/control/kinematic_joints", JointState, queue_size=0)
        # self.pub_tone = rospy.Publisher(basename + "/control/tone", UInt16MultiArray, queue_size=0)
        # self.pub_command = rospy.Publisher(basename + "/control/command", String, queue_size=0)

        # subscribers
        self.sub_package = rospy.Subscriber(base1 + "/sensors/package",
                    miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)
        # self.pose2 = rospy.Subscriber(base2 + "/sensors/body_pose",
        #     Pose2D, self.callback_pose2, queue_size=1, tcp_nodelay=True)
        # self.sub_mics = rospy.Subscriber(basename + "/sensors/mics",
        #             Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(base1 + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(base1 + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

        self.pub_kin.publish(self.kin)        


        self.timer = rospy.Timer(rospy.Duration(0.1), self.random_move)
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.camera_move)


    def callback_package(self, msg):
        self.input_package = msg
        
    def callback_pose(self, pose):
        if pose is not None:
            self.pos = pose
        
    def random_move(self, *args):
        self.velocity.twist.linear.x = 0.2
        target = self.cur_target
        target = np.array([0.5,-0.5])
        dists = [(self.pos.theta%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]
        if min(dists) < 0.5:
            self.velocity.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.velocity)
        elif dists[0] >= dists[1]:
            self.velocity.twist.angular.z = 1.2
            self.velocity.twist.linear.x = 0.01
            # print("turning left")
        else:
            self.velocity.twist.angular.z = -1.2
            self.velocity.twist.linear.x = 0.01

        self.pub_cmd_vel.publish(self.velocity)
        
if __name__ == "__main__":
    try:
        main = Explore()
        if(rospy.get_node_uri()):
            pass
        else:
            rospy.init_node("explore")
        while not rospy.core.is_shutdown():
            main.loop()
        main.interface.disconnect()
        exit()
    except Exception as e:
        print("exception",e)


