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

MAP_SCALE = 20
MAP_SIZE = 200
OBSTACLE_SIZE=2
BODY_SIZE=1


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
        self.prob_map = np.zeros((MAP_SIZE,MAP_SIZE),dtype=np.uint8)+127
        self.final_map = np.zeros_like(self.prob_map.shape, dtype=bool)
        self.path = []
        self.prob_map_pos = None
        self.interface = miro.lib.RobotInterface()

        self.cameras = [None,None]
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(30.0), 0.0, math.radians(-10.0)]
        self.velocity = TwistStamped()
        self.input_package = None
        self.move_head = True
        self.head_direction = 20
        self.map_start = np.array([MAP_SIZE//2,MAP_SIZE//2])
        self.map_pos = np.copy(self.map_start)
        robot_name = "/miro"
        self.miro_found = False

        # Calls publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher(robot_name + "/control/cmd_vel", TwistStamped, queue_size=10)
        self.pub_kin = rospy.Publisher(robot_name + "/control/kinematic_joints", JointState, queue_size=10)


        self.sub_mics = rospy.Subscriber(robot_name + "/sensors/mics",
                    Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_package = rospy.Subscriber(robot_name + "/sensors/package",
                    miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose_sub = rospy.Subscriber(robot_name + "/sensors/odom",
            Odometry, self.callback_pose, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(robot_name + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(robot_name + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)
        
        # self.timer1 = rospy.Timer(rospy.Duration(0.1), self.obstacle_detection)
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.head_move)


        # creating plots for visualisation
        plt.subplot(221)
        self.display = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.subplot(222)
        self.display2 = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.subplot(223)
        self.display3 = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.show()
        
        
    def callback_pose(self, pose):
        if pose is not None:
            self.pos = np.array([pose.pose.pose.position.x,pose.pose.pose.position.y])
            qw = pose.pose.pose.orientation.w
            qz = pose.pose.pose.orientation.z
            self.orientation = np.arctan2(2*(qw*qz), 1-2*(qz**2))
            if self.starting_pose is None:
                self.starting_pose = self.pos.copy()
            self.map_pos = self.pos2map(self.pos[0],self.pos[1])
            
    
    def callback_package(self, package):
        if package is not None:
            self.input_package = package
            if not hasattr(self, 'obstacle_timer'):
                self.obstacle_timer = rospy.Timer(rospy.Duration(0.1), self.obstacle_detection)
                self.exploration_timer = rospy.Timer(rospy.Duration(1.0), self.exploration_algorithm)
    
    def callback_mics(self, data):
        pass
    
    def callback_caml(self, ros_image):
        self.callback_cam(ros_image,0)
    
    def callback_camr(self, ros_image):
        self.callback_cam(ros_image,1)
        
    def callback_cam(self, image, index):
        pass
    
    # continuously moves the head around to better guage it's surroundings
    def head_move(self, *args):
        if self.move_head:
            self.kin.position[2] = self.input_package.kinematic_joints.position[2]+np.radians(self.head_direction)
        if abs(self.kin.position[2]) > np.radians(45):
            self.head_direction *= -1 # reverses the direction after hitting a limit
        # self.pub_kin.publish(self.kin)     
        self.interface.msg_kin_joints.set(self.kin,0.1)
    
    # function to increase the probability that a given cell is full
    def increase_prob(cells, is_full,dist):
        new_cells = np.zeros_like(cells).flatten()
        for i,cell in enumerate(cells.flatten()):
            new_cell = 0
            # if a cell is very certainly full or empty it stays that way
            if cell == 0 or cell == 255:
                new_cells[i] = cell
                continue
            elif is_full:
                new_cell = cell/255*0.6 # due to risk aversion full cells are made more likely to register than empty ones
            else:
                new_cell = 1-((1-cell/255)*0.95)
            new_cells[i] = new_cell*255
            if new_cells[i] == 127: # 127 indicates an unexplored cell, if evaluated cannot ever be unexplored again
                new_cells[i] += 1
        return new_cells.reshape(cells.shape).astype(int)
    
    # function to calculate map coordinates from pose coordinates
    def pos2map(self, posx, posy):
        return np.clip(np.array([self.map_start[0]-round((posx-self.starting_pose[0])*MAP_SCALE),
                self.map_start[1]-round((posy-self.starting_pose[1])*MAP_SCALE)],dtype=int),0,MAP_SIZE)
        
    # function to translate map coordinates into pose coordinates
    def map2pos(self, mapx, mapy):
        return np.array([(self.map_start[0]-mapx)/MAP_SCALE+self.starting_pose[0],
                (self.map_start[1]-mapy)/MAP_SCALE+self.starting_pose[1]])

    def obstacle_detection(self, *args):
        """Converts sonar data into an obstacle map
        Inputs: Pose, sonar
        Outputs: map
        """
        try:
            dist = self.input_package.sonar.range
            if self.starting_pose is None or self.pos is None:
                return
            if dist == float('inf'): # filtering out infinity
                dist = 1.0
            dist += 0.05 # adds distance to account for head offset, probably useless
            
            # makes the robot reverse and recalculate it's path if it's in front of an obstacle 

                # rospy.sleep(1)
            print(dist)

            # calculates the objects relative position
            obj_vec = dist*np.array([np.cos(self.orientation+self.input_package.kinematic_joints.position[2]),
                                        np.sin(self.orientation+self.input_package.kinematic_joints.position[2])])
            
            # translates the object position into map coordinates
            map_coords = self.pos2map(*(obj_vec+self.pos))
            
            if dist > 0.2:
                maxarg = np.argmax(abs(obj_vec))
                scan_range = np.arange(0.0,abs(obj_vec[maxarg]),1/MAP_SCALE)
                # for every cell between the miro and the object, set as empty
                for i in scan_range:
                    cur_coord = i*obj_vec/abs(obj_vec[maxarg])
                    cur_map = self.pos2map(*(cur_coord+self.pos)).astype(int)
                    coords = [max(cur_map[0]-OBSTACLE_SIZE,0),
                            min(cur_map[0]+OBSTACLE_SIZE+1,self.prob_map.shape[0]),
                            max(cur_map[1]-OBSTACLE_SIZE,0),
                            min(cur_map[1]+OBSTACLE_SIZE+1,self.prob_map.shape[1])]
                    self.prob_map[coords[0]:coords[1],coords[2]:coords[3]] = Helper.increase_prob(self.prob_map[coords[0]:coords[1],coords[2]:coords[3]],False,dist)
            if dist < 0.6: 
                selected_map = self.prob_map[max(map_coords[0]-OBSTACLE_SIZE,0):min(map_coords[0]+OBSTACLE_SIZE+1,self.prob_map.shape[0]),
                                        max(map_coords[1]-OBSTACLE_SIZE,0):min(map_coords[1]+OBSTACLE_SIZE+1,self.prob_map.shape[1])]
                self.prob_map[max(map_coords[0]-OBSTACLE_SIZE,0):min(map_coords[0]+OBSTACLE_SIZE+1,self.prob_map.shape[0]),
                        max(map_coords[1]-OBSTACLE_SIZE,0):min(map_coords[1]+OBSTACLE_SIZE+1,self.prob_map.shape[1])] = Helper.increase_prob(selected_map,True,0)
                
            # draws in the graph
            if self.display is not None:
                self.display.set_data(self.prob_map)
                plt.draw()
        except Exception as e:
            print(e)
        


    def exploration_algorithm(self, *args):
        """Performs bredth first search on the obstacle map
        Inputs: Pose, map
        Outputs: path
        """
        try:
            if self.miro_found: return
            if len(self.path) > 3:
                return
            map_copy = self.prob_map.copy()
            node_queue = Queue()
            visited = set()
            node_queue.put(self.map_pos)
            dist_map = np.zeros(map_copy.shape[:2],dtype=int)-1
            visited.add(tuple(self.map_pos))
            end_pos = None
            # finds the distances from the nearby nodes
            while not node_queue.empty():
                cur_pos = node_queue.get()
                cur_val = dist_map[cur_pos[0],cur_pos[1]]
                
                # uses von-newman neighboourhood for dijkstra's
                adjacent_list = np.array([[-1,0],[0,1],[1,0],[0,-1]])
                adjacent = cur_pos+adjacent_list
                
                # itterates through all the adjacent nodes
                for i in adjacent:
                    # skips if the index is out of bounds or already visited
                    if (i<0).any() or (i>=MAP_SIZE).any():
                        continue
                    if tuple(i) in visited:
                        continue
                    cell = map_copy[i[0],i[1]]
                    # adds the indes to the set of visited nodes
                    visited.add(tuple(i))
                    
                    # checks if the cell is not a miro
                    value = cell
                    # checks if the cell is unexplored, if so it terminates the search
                    if value == 127:
                        end_pos = i
                        dist_map[i[0],i[1]] = cur_val+1
                        break
                    # chicks if the cell is likely to be full, very high threshold due ot uncertainty
                    elif value < 120:
                        continue
                    node_queue.put(i)
                    # increases the distance of the node on the graph
                    if dist_map[i[0],i[1]] == -1 or dist_map[i[0],i[1]] > cur_val+1:
                        dist_map[i[0],i[1]] = cur_val+1
                else:
                    continue
                break
            # returns nothing if no path is found, this occurs when there is no unexplored space left on the map
            if end_pos is None:
                self.path = []
                print("No path left")
                return
            # displays the distance map in a plot
            self.display2.set_data((dist_map+1)*255//(np.max(dist_map)+2))
            plt.draw()
            next_val = (dist_map[end_pos[0],end_pos[1]],end_pos)
            path = []
            # backtracks until it finds the starting position again
            while next_val[0] > 0:
                cur_pos = next_val[1]
                
                # uses moores distance this time as it yeilds more natural movement
                min_pos = np.clip(cur_pos-1,0,None)
                max_pos = np.clip(cur_pos+1,None,MAP_SIZE-1)
                x_axis = np.arange(min_pos[0],max_pos[0]+1,1)
                y_axis = np.arange(min_pos[1],max_pos[1]+1,1)
                xv, yv = np.meshgrid(x_axis,y_axis)
                adjacent = np.append([xv],[yv],axis=0).reshape(2,-1).T
                for i in adjacent:
                    # finds the nearest adjacent value
                    value = dist_map[i[0],i[1]]
                    if value != -1 and value < next_val[0]:
                        next_val = (value, i)
                # appends the best value to the path
                path.append(next_val[1])
            self.path = path[:-5]
            if not hasattr(self, 'movement_timer'):
                self.movement_timer = rospy.Timer(rospy.Duration(0.2), self.explore_path)

            # displays the path
            path_map = np.zeros(dist_map.shape,dtype=int)
            for i in path:
                path_map[i[0],i[1]] = 255
            self.display3.set_data(path_map)
            plt.draw()
            print(path)         
        except Exception as e:
            print(e)

    
    def explore_path(self, *args):
        """Explores the path performed by the search algorithm
        Inputs: Pose, path
        Outputs: Movement
        """
        if self.miro_found:
            return
        try:
            if len(self.path) == 0:
                self.velocity.twist.linear.x = 0.05
                self.velocity.twist.angular.z = 1.8
                self.interface.msg_cmd_vel.set(self.velocity, 0.2)
                # rospy.sleep(6)
                print("path finished")
                return
            # print(self.path[-1])
            # calculates the target angle for the position
            target_pos = self.map2pos(*self.path[-1])
            target = np.arctan2(target_pos[1]-self.pos[1],target_pos[0]-self.pos[0])
            print(self.path[-1], target_pos-self.pos)

            # calculates the differance in angle between current and target
            dists = [(self.orientation%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.orientation%(2*np.pi))%(2*np.pi)]

            dist = self.input_package.sonar.range
            if self.starting_pose is None or self.pos is None:
                return
            if dist == float('inf'): # filtering out infinity
                dist = 1.0
            # dist += 0.01 # adds distance to account for head offset, probably useless

            # checks if the robot is close to the next node in the path 
            # self.velocity.twist.linear.x = 0.00

            if np.linalg.norm(target_pos-self.pos) < 0.3:
                # self.velocity.twist.linear.x = 0.20
                # self.velocity.twist.angular.z = 0.0
                self.path = self.path[:-1] # removes the last element from the path
                print("path explored")

            # checks if the robot is looking in the right direction
            if min(dists) < 0.3:
                # self.velocity.twist.angular.z = 0.0
                if dist < 0.4 and abs(self.input_package.kinematic_joints.position[2]) < 0.6:
                    self.path = []
                    self.velocity.twist.linear.x = -0.2
                    self.move_head = False
                    # self.interface.msg_cmd_vel.set(self.velocity, 1)
                    print("reversing")
                else:
                    self.velocity.twist.linear.x = 0.2*(1-min(dists)/np.pi)
                    self.move_head = True
                    print("moving forwards")
            else:
                self.velocity.twist.linear.x = 0.0



                # self.interface.msg_cmd_vel.set(self.velocity, 0.5)
            if min(dists) < 0.2:
                self.velocity.twist.angular.z = 0.0
                print("stop turning", min(dists))
            # moves clockwise if the right angle is lower
            elif dists[0] >= dists[1]:
                self.velocity.twist.angular.z = 1.8
                print("moving right", min(dists))
                # self.velocity.twist.linear.x = 0.02
                # print("turning left")
            # moves counter clockwise otherwise
            else:
                self.velocity.twist.angular.z = -1.8
                print("moving left", min(dists))
                # self.velocity.twist.linear.x = 0.02

            self.interface.msg_cmd_vel.set(self.velocity, 0.3)
        except Exception as e:
            print("exception",e)
        
        
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
    if(rospy.get_node_uri()):
        pass
    else:
        rospy.init_node("Helper_miro")
    rospy.spin()