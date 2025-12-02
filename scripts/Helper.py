#!/usr/bin/python3

import os
import rospy
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, Int16MultiArray, String
from geometry_msgs.msg import TwistStamped, Pose2D
from sensor_msgs.msg import JointState, CompressedImage
from nav_msgs.msg import Odometry

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import math
import miro2 as miro
import time
from queue import Queue, LifoQueue

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import uuid
import heapq

droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)

MAP_SCALE = 10
MAP_SIZE = 80
OBSTACLE_SIZE=1
BODY_SIZE=2


#Grid Cell for the A-Star implementation
class Cell:
    def __init__(self):
        self.parent_x = 0
        self.parent_y = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

class Helper():
    
    def __init__(self):
        
        # Initilises variables
        self.pos = None
        self.orientation = None
        self.starting_pose = None
        self.prob_map = np.zeros((OBSTACLE_SIZE,OBSTACLE_SIZE),dtype=int)+127
        self.final_map = np.zeros_like(self.prob_map.shape, dtype=bool)
        self.path = None
        self.map_pos = None
        
        self.cameras = [None,None]
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(10.0), 0.0, math.radians(-10.0)]
        self.input_package = None
        
        robot_name = "/miro"

        # Calls publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher(robot_name + "/control/cmd_vel", TwistStamped, queue_size=10)
        self.pub_kin = rospy.Publisher(robot_name + "/control/kinematic_joints", JointState, queue_size=10)


        self.sub_mics = rospy.Subscriber(robot_name + "/sensors/mics",
                    Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_package = rospy.Subscriber(robot_name + "/sensors/package",
                    miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose_sub = rospy.Subscriber(robot_name + "/sensors/odom",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(robot_name + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(robot_name + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)
        
    
        
        
    def callback_pose(self, pose):
        if pose is not None:
            self.pos = np.array([pose.pose.position.x,pose.pose.position.y])
            qw = pose.pose.orientation.w
            qz = pose.pose.orientation.z
            self.orientation = np.arctan2(2*(qw*qz), 1-2*(qz**2))
            if self.starting_pose is None:
                self.starting_pose = self.pos
            
    
    def callback_package(self, package):
        if package is not None:
            self.input_package = package
            if not hasattr(self, 'obstacle_timer'):
                self.obstacle_timer = rospy.Timer(rospy.Duration(0.1), self.obstacle_detection)
    
    def callback_mics(self):
        pass
    
    def callback_caml(self, ros_image):
        self.callback_cam(ros_image,0)
    
    def callback_camr(self, ros_image):
        self.callback_cam(ros_image,1)
        
    def callback_cam(self, image, index):
        pass
    
    
    # function to increase the probability that a given cell is full
    def increase_prob(cells, is_full):
        new_cells = cells
        for i,cell in enumerate(cells.reshape(-1,3)):
            new_cell = 0
            # if a cell is very certainly full or empty it stays that way
            if (cell == 0).all() or (cell == 255).all():
                new_cells[i] = cell
                continue
            elif is_full:
                new_cell = cell/255*0.75 # due to risk aversion full cells are made more likely to register than empty ones
            else:
                new_cell = 1-((1-cell/255)*0.9)
            new_cells[i] = new_cell*255
            if (new_cells[i] == 127).all(): # 127 indicates an unexplored cell, if evaluated cannot ever be unexplored again
                new_cells[i] += 1
        return new_cells.reshape(cells.shape).astype(int)
    
    def obstacle_detection(self):
        """Converts sonar data into an obstacle map
        Inputs: Pose, sonar
        Outputs: map
        """
    
    def eploration_algorithm(self):
        """Performs bredth first search on the obstacle map
        Inputs: Pose, map
        Outputs: path
        """
    
    def explore_path(self):
        """Explores the path performed by the search algorithm
        Inputs: Pose, path
        Outputs: Movement
        """
        
    def detect_miro(self):
        """Uses computer vision to detect the miro
        Inputs: Camera
        Outputs: Bool
        """
    
    def pose_detection(self):
        """Uses computer vision to detect the pose of the miro
        Inputs: Pose, Camera
        Outputs: Pose
        """
    
    def Astar(self):
        """Uses A* to determine the miro's next path
        Inputs: Map
        Outputs: Path
        """
        
    def send_audio(self):
        """Send audio signal
        Inputs: Command
        Outputs: Audio
        """
    
    
if __name__ == '__main__':
    main = Helper()
    rospy.init_node("Helper_miro")
    rospy.spin()