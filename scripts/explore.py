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

MAP_SCALE = 40

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
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(15.0), 0.0, math.radians(-10.0)]
        self.input_package = None
        self.map = np.zeros((200,200,3),np.uint8)+127
        self.display = None
        # self.cos_joints = Float32MultiArray()
        # self.cos_joints.data = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        self.map_start = np.array([100,100])
        self.map_pos = np.copy(self.map_start)

        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=10)
        self.pub_kin = rospy.Publisher(base1 + "/control/kinematic_joints", JointState, queue_size=10)

        # subscribers
        self.sub_package = rospy.Subscriber(base1 + "/sensors/package",
                    miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)


        self.timer = rospy.Timer(rospy.Duration(0.1), self.move_to_point)
        # self.timer2 = rospy.Timer(rospy.Duration(0.1), self.camera_move)
        self.pub_kin.publish(self.kin)        

        # plt.subplot(121)
        self.display = plt.imshow(self.map, vmin=0,vmax=255)
        plt.show()

    def increase_prob(cells, is_full):
        new_cells = np.zeros(cells.reshape(-1,3).shape)
        for i,cell in enumerate(cells.reshape(-1,3)):
            new_cell = 0
            if is_full:
                new_cell = cell/255*0.8
            else:
                new_cell = 1-((1-cell/255)*0.8)
            
            new_cells[i] = new_cell*255
            print(new_cells[i],cell)
        return new_cells.reshape(cells.shape).astype(int)

    def callback_package(self, msg):
        self.input_package = msg
        dist = self.input_package.sonar.range
        if dist < 0.8 and dist > 0.1:
            # dist += 0.1
            obj_vec = dist*np.array([np.sin(self.pos.theta),np.cos(self.pos.theta)])
            pos_vec = np.array([self.pos.x,self.pos.y])
            # print(obj_vec,pos_vec-obj_vec, self.pos2map(*(pos_vec-obj_vec)))
            map_coords = self.pos2map(*(obj_vec+pos_vec))
            # print(map_coords, self.map_pos,dist)
            maxarg = np.argmax(abs(obj_vec))
            # print(obj_vec,obj_vec[maxarg])
            scan_range = np.arange(0.0,abs(obj_vec[maxarg]),1/MAP_SCALE)
            for i in scan_range:
                cur_coord = i*obj_vec/abs(obj_vec[maxarg])
                cur_map = self.pos2map(*(cur_coord+pos_vec)).astype(int)
                selected_map = self.map[max(cur_map[0]-1,0):min(cur_map[0]+1,self.map.shape[0]),max(cur_map[1]-1,0):min(cur_map[1]+1,self.map.shape[1])]
                self.map[max(cur_map[0]-1,0):min(cur_map[0]+1,self.map.shape[0]),max(cur_map[1]-1,0):min(cur_map[1]+1,self.map.shape[1])] = Explore.increase_prob(selected_map,False)
            selected_map = self.map[max(map_coords[0]-1,0):min(map_coords[0]+1,self.map.shape[0]),max(map_coords[1]-1,0):min(map_coords[1]+1,self.map.shape[1])]
            self.map[max(map_coords[0]-1,0):min(map_coords[0]+1,self.map.shape[0]),max(map_coords[1]-1,0):min(map_coords[1]+1,self.map.shape[1])] = Explore.increase_prob(selected_map,True)
            # self.map[max(map_coords[0]-1,0):min(map_coords[0]+1,self.map.shape[0]),max(map_coords[1]-1,0):min(map_coords[1]+1,self.map.shape[1])][:,:,2] = 255
            if self.display is not None:
                self.display.set_data(self.map)
                plt.draw()
        
    def pos2map(self, posx, posy):
        return np.array([self.map_start[0]+round((posx-self.start_pos.x)*MAP_SCALE),
                self.map_start[1]+round((posy-self.start_pos.y)*MAP_SCALE)],dtype=int)
        
    def map2pos(self, mapx, mapy):
        return np.array([(mapx-self.map_start[0])/MAP_SCALE+self.start_pos.x,
                (mapy-self.map_start[1])/MAP_SCALE]+self.start_pos.y)
        
    def callback_pose(self, pose):
        if pose is not None:
            if self.start_pos is None:
                self.start_pos = pose
            self.pos.x = pose.x
            self.pos.y = pose.y
            self.pos.theta = pose.theta
            self.map[max(self.map_pos[0]-4,0):min(self.map_pos[0]+4,self.map.shape[0]),max(self.map_pos[1]-4,0):min(self.map_pos[1]+4,self.map.shape[1])] = 255
            self.map_pos = self.pos2map(self.pos.x, self.pos.y)
            self.map[max(self.map_pos[0]-4,0):min(self.map_pos[0]+4,self.map.shape[0]),max(self.map_pos[1]-4,0):min(self.map_pos[1]+4,self.map.shape[1])] = 0
            self.map[max(self.map_pos[0]-4,0):min(self.map_pos[0]+4,self.map.shape[0]),max(self.map_pos[1]-4,0):min(self.map_pos[1]+4,self.map.shape[1])][:,:,0] = 255
            # self.map[*self.map_pos][2] = 0
            # self.map[*self.map_pos][0] = 0
            if self.display is not None:
                self.display.set_data(self.map)
                plt.draw()
       
    def move_to_point(self, *args):
        self.velocity.twist.linear.x = 0.2
        target = self.cur_target
        # target_pos = np.array([0.5,-0.7])
        self.velocity.twist.angular.z = 1.2
        self.velocity.twist.linear.x = 0.00
        # self.pub_cmd_vel.publish(self.velocity)
        return
        target_pos = self.map2pos(150,90)
        target = np.arctan2(target_pos[1]-self.pos.y,target_pos[0]-self.pos.x)
        dists = [(self.pos.theta%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]
        # print(dists, self.pos.x, self.pos.y)
        if np.linalg.norm(target_pos-np.array([self.pos.x,self.pos.y])) < 0.1:
            self.velocity.twist.angular.z = 0.0
            self.velocity.twist.linear.x = 0.0
        elif min(dists) < 0.1:
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


