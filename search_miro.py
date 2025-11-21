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

class Cell:
    def __init__(self):
        self.parent_x = 0
        self.parent_y = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0


class searchMiro:
    def __init__(self):
        base1 = "/miro01"
        base2 = "/miro02"
        self.interface = miro.lib.RobotInterface
        self.velocity = TwistStamped()
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pos = Pose2D()
        self.add_dist = 0.0
        self.camera_interval  = time.time_ns()
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(50.0), 0.0, 0.0]
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        #Publisherz
        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.pub_cos = rospy.Publisher(base1 + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)

        #Subscribers
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)
        self.currentAngle = None
        self.currentPos = None
    #Adds values to the 2D array for a cylinder obstical
    def addCylinder(self,grid,x,y):
        for i in range (x - 1, x + 3):
            for j in range (y - 1, y + 3):
                grid[i][j] = 'C'
            
        return grid

    #Adds values to the 2D array for the lost and seeking miros
    def addMiros(self,grid,lostx,losty,searchx,searchy):
        grid[lostx][losty] = 'L'
        grid[lostx-1][losty] = 'L'
        grid[searchx][searchy] = 'M'
        grid[searchx-1][searchy] = 'M'
        return grid

    #Generates a grid of the world
    def generateGrid(self,start,goal,rows,cols):  
            
            grid = [['.' for _ in range(cols)] for _ in range(rows)]
            grid = self.addMiros(grid,goal[0],goal[1],start[0],start[1])
            grid = self.addCylinder(grid,18,13)#
            grid = self.addCylinder(grid,29,13)
            for row in grid:
                print(' '.join(row))
            return grid
 
    #Returns the estimated remaining cost between two points
    def findHeuristic(self,row,col,goal):
        return ((row - goal[0])**2 + (col - goal[1])**2)**0.5
    
    #Checks if the robot is in the goal state
    def isGoal(self,row,col,goal):
        return row == goal[0] and col == goal[1]
    
    #Check if the move is possible
    def isUnblocked(self,row,col,grid):
        return not grid[row][col] == 'C'
    
    #Check if it is valid
    def isValid(self,row, col,width,height):
        return (row >= 0) and (row < width) and (col >= 0) and (col < height)

    #Displays the path to take
    def tracePath(self, cellDetails,goal):
        print("The path is ")
        path = []
        row = goal[0]
        col = goal[1]
        dirMap = {
            (-1,0): ((-0.1,0),radians(360)),
            (-1,1): ((-0.1,-0.1),radians(45)),
            (0,1): ((0,0.1),radians(90)),
            (1,1): ((0.1,0.1),radians(135)),
            (1,0): ((0.1,0),radians(180)),
            (1,-1): ((0.1,-0.1),radians(225)),
            (0,-1): ((0,-0.1),radians(270)),
            (-1,-1): ((-0.1,-0.1),radians(315))
        }
        while not (cellDetails[row][col].parent_x == row and cellDetails[row][col].parent_y == col):      
            tempRow = cellDetails[row][col].parent_x
            tempCol = cellDetails[row][col].parent_y
            changeMatrix = (tempRow - row, tempCol - col)
            dir = dirMap.get(changeMatrix)
            row = tempRow
            col = tempCol
            path.append(dir)
        path.reverse()
        
        for i in path:
            print("->", i, end=" ")
        print()
        return path

    #A-Star search implementation
    def aStarSearch(self,grid,start,goal,width,height):

        if not self.isValid(start[0],start[1],width,height) or not self.isValid(goal[0],goal[1],width,height):
            return "Source or destinaton is invalid"
        
        if self.isGoal(start[0],start[1],goal):
            return "Already at destination"
        
        closedList = [[False for _ in range(height)] for _ in range(width)]
        cellDetails = [[Cell() for _ in range(height)] for _ in range(width)]
        x = start[0]
        y = start[1]
        cellDetails[x][y].f = 0
        cellDetails[x][y].g = 0
        cellDetails[x][y].h = 0
        cellDetails[x][y].parent_x = x
        cellDetails[x][y].parent_y = y

        openList = []
        heapq.heappush(openList,(0.0,x,y))
        foundGoal = False

        while len(openList) > 0:
            p = heapq.heappop(openList)

            x = p[1]
            y = p[2]
            closedList[x][y] = True

            directions = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
            for dir in directions:
                xNew = x + dir[0]
                yNew = y + dir[1]

                if self.isUnblocked(xNew,yNew,grid) and self.isValid(xNew,yNew,width,height) and not closedList[xNew][yNew]:
                    if self.isGoal(xNew,yNew,goal):
                        cellDetails[xNew][yNew].parent_x = x
                        cellDetails[xNew][yNew].parent_y = y
                        print("Destination Found")
                        foundGoal = True
                        return self.tracePath(cellDetails, goal)
                    else:
                        gNew = cellDetails[x][y].g +1.0
                        hNew = self.findHeuristic(xNew,yNew,goal)
                        fNew = gNew + hNew

                        if cellDetails[xNew][yNew].f == float('inf') or cellDetails[xNew][yNew].f > fNew:
                            heapq.heappush(openList,(fNew,xNew,yNew))
                            cellDetails[xNew][yNew].f = fNew   
                            cellDetails[xNew][yNew].g = gNew   
                            cellDetails[xNew][yNew].h = hNew   
                            cellDetails[xNew][yNew].parent_x = x
                            cellDetails[xNew][yNew].parent_y = y   
        if not foundGoal:       
            print("Failed to find destination")
            return []
    #Callbacks
    def callback_pose(self, pose):
        if pose != None:
            self.pos = pose
            self.currentAngle = self.pos.theta

    def callback_cam(self, ros_image, index):
            
        # ignore empty frames which occur sometimes during parameter changes
        if len(ros_image.data) == 0:
            print("dropped empty camera frame")
            return
        try:

            # convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # orb = cv2.ORB_create()
            # kp , des= orb.detectAndCompute(image,None)
            # image = cv2.drawKeypoints(image,kp,None,color=(0,0,255),flags=0)

            # store image for display
            self.camera[index] = image


        except CvBridgeError as e:

            # swallow error, silently
            #print(e)
            pass
    def callback_caml(self, ros_image):
        self.callback_cam(ros_image,0)
    
    def callback_camr(self, ros_image):
        self.callback_cam(ros_image,1)

    def move(self,path):
        rate = rospy.Rate(10)
        for move in path:
            currentAngle = self.currentAngle
            print(move)
            print("waiting")
            self.velocity.twist.linear.x = 0
            self.velocity.twist.angular.z = 0
            self.pub_cmd_vel.publish(self.velocity)
            invert_move = -move[1]
            # time.sleep(1)
            dists = [(self.pos.theta%(2*np.pi)-invert_move%(2*np.pi))%(2*np.pi),(invert_move%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]

            self.velocity.twist.linear.x = 0.0
            self.velocity.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.velocity)
            # time.sleep(0.1)
            #move forward
            newpos = np.array([self.pos.x - move[0][0],self.pos.y - move[0][1]])
            cur_pos = np.array([self.pos.x,self.pos.y])
            while np.linalg.norm(newpos-cur_pos) > 0.01:
                print(f"{np.linalg.norm(newpos-cur_pos)}")
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

#check for lost miro periodically

#if found navigate there

#if not get moving

#if move until too close to surface then turn

#if it sees other miro remember where it is 

if __name__ == "__main__":
    try:
        rospy.init_node("search_miro")
        robot = searchMiro()
        while robot.currentAngle is None:
            print("Waiting for first angle reading…")

        rospy.spin
        width  = 39
        height = 26
        start = (10,13)
        goal = (37,18)
        grid = robot.generateGrid(start,goal,width,height)
        path = robot.aStarSearch(grid,start,goal,width,height)
        robot.move(path)
        robot.interface.disconnect()
    except KeyboardInterrupt:
        print("\nExecution ending (Ctrl+C)=")
        # try:
        #     robot.interface.disconnect()
        # except:
        #     pass
    except Exception as e:
        print("exception",e)
    robot.interface.disconnect()