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
from queue import Queue, LifoQueue

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import uuid


droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)

MAP_SCALE = 10
MAP_SIZE = 80
OBSTACLE_SIZE=1
BODY_SIZE=0


class Explore:
    
    def __init__(self):
        base1 = "/miro01"
        # base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        # self.velocity# = TwistStamped()
        
        self.pos = Pose2D()
        self.start_pos = None
        self.cur_target = None
        self.time1 = time.time_ns()
        self.add_dist = 0.0
        self.camera_interval  = time.time_ns()
        self.kin = JointState()
        # defines the joint positions
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(10.0), 0.0, math.radians(-10.0)]
        self.input_package = None
        self.map = np.zeros((MAP_SIZE,MAP_SIZE,3),np.uint8)+127
        self.display = None
        self.map_start = np.array([MAP_SIZE//2,MAP_SIZE//2])
        self.map_pos = np.copy(self.map_start)
        self.head_direction = 5
        self.path = []
        self.starting_scan = True

        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=10)
        self.pub_kin = rospy.Publisher(base1 + "/control/kinematic_joints", JointState, queue_size=10)

        # subscribers
        self.sub_package = rospy.Subscriber(base1 + "/sensors/package",
                    miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)


        # performs a simple rotation to scan the surrounding area
        scan_timer = time.time_ns()
        while time.time_ns()-scan_timer < 5e9:
            self.velocity.twist.angular.z = 1.2
            self.pub_cmd_vel.publish(self.velocity)
        self.starting_scan = False


        # timers to call the various functions
        self.timer = rospy.Timer(rospy.Duration(0.1), self.move_to_point)
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.head_move)
        self.timer3 = rospy.Timer(rospy.Duration(1.0), self.search_map)

        # creating plots for visualisation
        plt.subplot(221)
        self.display = plt.imshow(self.map, vmin=0,vmax=255)
        plt.subplot(222)
        self.display2 = plt.imshow(self.map, vmin=0,vmax=255)
        plt.subplot(223)
        self.display3 = plt.imshow(self.map, vmin=0,vmax=255)
        plt.show()

    # function to increase the probability that a given cell is full
    def increase_prob(cells, is_full,dist):
        new_cells = np.zeros(cells.reshape(-1,3).shape)
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
            # print(new_cells[i],cell)
        return new_cells.reshape(cells.shape).astype(int)
    
    # callback for sonar
    def callback_package(self, msg):
        self.input_package = msg
        dist = self.input_package.sonar.range
        if self.start_pos is None or self.pos is None:
            return
        if dist == float('inf'): # filtering out infinity
            dist = 1.0
        dist += 0.01 # adds distance to account for head offset, probably useless
        
        # makes the robot reverse and recalculate it's path if it's in front of an obstacle 
        if dist < 0.2 and abs(self.kin.position[2]) < 0.6:
            self.path = []
            self.velocity.twist.linear.x = -0.1
            self.pub_cmd_vel.publish(self.velocity)
        
        # calculates the objects relative position
        obj_vec = dist*np.array([np.cos(self.pos.theta+self.kin.position[2]),
                                    np.sin(self.pos.theta+self.kin.position[2])])
        # position vector for faster calculations
        pos_vec = np.array([self.pos.x,self.pos.y])
        
        # translates the object position into map coordinates
        map_coords = self.pos2map(*(obj_vec+pos_vec))
        
        if dist > 0.2:
            maxarg = np.argmax(abs(obj_vec))
            scan_range = np.arange(0.0,abs(obj_vec[maxarg]),1/MAP_SCALE)
            # for every cell between the miro and the object, set as empty
            for i in scan_range:
                cur_coord = i*obj_vec/abs(obj_vec[maxarg])
                cur_map = self.pos2map(*(cur_coord+pos_vec)).astype(int)
                coords = [max(cur_map[0]-OBSTACLE_SIZE,0),
                          min(cur_map[0]+OBSTACLE_SIZE+1,self.map.shape[0]),
                          max(cur_map[1]-OBSTACLE_SIZE,0),
                          min(cur_map[1]+OBSTACLE_SIZE+1,self.map.shape[1])]
                self.map[coords[0]:coords[1],coords[2]:coords[3]] = Explore.increase_prob(self.map[coords[0]:coords[1],coords[2]:coords[3]],False,dist)
        if dist < 0.8: 
            selected_map = self.map[max(map_coords[0]-OBSTACLE_SIZE,0):min(map_coords[0]+OBSTACLE_SIZE+1,self.map.shape[0]),
                                    max(map_coords[1]-OBSTACLE_SIZE,0):min(map_coords[1]+OBSTACLE_SIZE+1,self.map.shape[1])]
            self.map[max(map_coords[0]-OBSTACLE_SIZE,0):min(map_coords[0]+OBSTACLE_SIZE+1,self.map.shape[0]),
                     max(map_coords[1]-OBSTACLE_SIZE,0):min(map_coords[1]+OBSTACLE_SIZE+1,self.map.shape[1])] = Explore.increase_prob(selected_map,True,0)
            
        # draws in the graph
        if self.display is not None:
            self.display.set_data(self.map)
            plt.draw()
        
    # function to calculate map coordinates from pose coordinates
    def pos2map(self, posx, posy):
        return np.clip(np.array([self.map_start[0]-round((posx-self.start_pos.x)*MAP_SCALE),
                self.map_start[1]-round((posy-self.start_pos.y)*MAP_SCALE)],dtype=int),0,MAP_SIZE)
        
    # function to translate map coordinates into pose coordinates
    def map2pos(self, mapx, mapy):
        return np.array([(self.map_start[0]-mapx)/MAP_SCALE+self.start_pos.x,
                (self.map_start[1]-mapy)/MAP_SCALE+self.start_pos.y])

    # callback for the odometery
    def callback_pose(self, pose):
        if pose is not None:
            if self.start_pos is None:
                self.start_pos = pose
            self.pos.x = pose.x
            self.pos.y = pose.y
            self.pos.theta = pose.theta
            # sets positions of previous robot as empty 
            self.map[max(self.map_pos[0]-BODY_SIZE,0):min(self.map_pos[0]+BODY_SIZE+1,self.map.shape[0]-1),
                     max(self.map_pos[1]-BODY_SIZE,0):min(self.map_pos[1]+BODY_SIZE+1,self.map.shape[1]-1)] = 255
            # sets the new position of the robot to full
            self.map_pos = self.pos2map(self.pos.x, self.pos.y)
            self.map[max(self.map_pos[0]-BODY_SIZE,0):min(self.map_pos[0]+BODY_SIZE+1,self.map.shape[0]-1),
                     max(self.map_pos[1]-BODY_SIZE,0):min(self.map_pos[1]+BODY_SIZE+1,self.map.shape[1]-1)] = 0
            self.map[max(self.map_pos[0]-BODY_SIZE,0):min(self.map_pos[0]+BODY_SIZE+1,self.map.shape[0]-1),
                     max(self.map_pos[1]-BODY_SIZE,0):min(self.map_pos[1]+BODY_SIZE+1,self.map.shape[1]-1)][:,:,0] = 255
            # displays the map
            if self.display is not None:
                self.display.set_data(self.map)
                plt.draw()
       
    # continuously moves the head around to better guage it's surroundings
    def head_move(self, *args):
        self.kin.position[2] += np.radians(self.head_direction)
        if abs(self.kin.position[2]) > np.radians(50):
            self.head_direction *= -1 # reverses the direction after hitting a limit
        self.pub_kin.publish(self.kin)        

    # dijkstra's algorithm to search for unexplored space
    def search_map(self, *args):
        if len(self.path) != 0:
            return
        map_copy = self.map.copy()
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
                if (cell == cell[0]).all():
                    value = cell[0]
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
        self.path = path
        # displays the path
        path_map = np.zeros(dist_map.shape,dtype=int)
        for i in path:
            path_map[i[0],i[1]] = 255
        self.display3.set_data(path_map)
        plt.draw()
        print(path)
       
    # function to move the robot along the target path
    def move_to_point(self, *args):
        try:
            if len(self.path) == 0:
                self.velocity.twist.linear.x = 0.0
                self.velocity.twist.angular.z = 0.0
                self.pub_cmd_vel.publish(self.velocity)
                print("path finished")
                return
            print(self.path[-1])
            # calculates the target angle for the position
            target_pos = self.map2pos(*self.path[-1])
            self.velocity.twist.linear.x = 0.2
            target = np.arctan2(target_pos[1]-self.pos.y,target_pos[0]-self.pos.x)
            
            # calculates the differance in angle between current and target
            dists = [(self.pos.theta%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]

            # checks if the robot is close to the next node in the path 
            if np.linalg.norm(target_pos-np.array([self.pos.x,self.pos.y])) < 0.3:
                self.velocity.twist.linear.x = 0.0
                self.velocity.twist.angular.z = 0.0
                self.path = self.path[:-1] # removes the last element from the path
            
            # checks if the robot is looking in the right direction
            elif min(dists) < 0.1:
                self.velocity.twist.angular.z = 0.0
                self.pub_cmd_vel.publish(self.velocity)
            # moves clockwise if the right angle is lower
            elif dists[0] >= dists[1]:
                self.velocity.twist.angular.z = 1.2
                self.velocity.twist.linear.x = 0.01
                # print("turning left")
            # moves counter clockwise otherwise
            else:
                self.velocity.twist.angular.z = -1.2
                self.velocity.twist.linear.x = 0.01

            self.pub_cmd_vel.publish(self.velocity)
        except Exception as e:
            print("exception",e)
        
    def loop(self):
        pass
        
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


