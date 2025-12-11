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
import onnxruntime as ort

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
MAP_SIZE = 50
OBSTACLE_SIZE=0
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
        self.helper_path = []
        self.helper_moving = False
        self.prob_map_pos = None
        self.interface = miro.lib.RobotInterface()
        self.image_converter = CvBridge()
        mode_path = "best_new.onnx"
        self.onnx_model = ort.InferenceSession(mode_path)

        self.camera = [None,None]
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(35.0), 0.0, math.radians(-10.0)]
        self.sens_kin = JointState()
        self.sens_kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.sens_kin.position = [0.0, math.radians(35.0), 0.0, math.radians(-10.0)]
        self.velocity = TwistStamped()
        self.input_package = None
        self.move_head = True
        self.head_direction = 20
        self.map_start = np.array([MAP_SIZE//2,MAP_SIZE//2])
        self.map_pos = np.copy(self.map_start)
        robot_name = "/miro"
        self.miro_found = False
        self.pred_dist = [None, None]
        self.midpoints = [None, None]
        self.angle_queue = []
        self.distance_queue = []
        self.pred_pos = None
        self.pred_map_pos = None



        # Calls publishers and subscribers
        self.pub_cmd_vel = rospy.Publisher(robot_name + "/control/cmd_vel", TwistStamped, queue_size=10)
        self.pub_tone = rospy.Publisher(robot_name + "control/tone", UInt16MultiArray,queue_size=1)



        self.sub_kin = rospy.Subscriber(robot_name + "/sensors/kinematic_joints", JointState, self.callback_kin, queue_size=10)
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
        
        self.timer1 = rospy.Timer(rospy.Duration(0.1), self.obstacle_detection)
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.head_move)
        self.timer3 = rospy.Timer(rospy.Duration(3), self.detect_miro)

        # creating plots for visualisation
        plt.figure()
        plt.subplot(321)
        self.display = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.subplot(322)
        self.display2 = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.subplot(323)
        self.display3 = plt.imshow(self.prob_map, vmin=0,vmax=255)
        plt.subplot(324)
        self.display4 = plt.imshow(self.prob_map, vmin=0,vmax=255)
        # plt.show(block=False)

        # plt.figure()
        plt.subplot(325)
        self.camera_display1 = plt.imshow(np.array([[0.0]]), 'gray')
        plt.subplot(326)
        self.camera_display2 = plt.imshow(np.array([[0.0]]), 'gray')
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
    
    def callback_kin(self, kin):
        if kin is not None:
            self.sens_kin = kin

    def callback_mics(self, data):
        pass
    
    def callback_caml(self, ros_image):
        self.callback_cam(ros_image,0)
    
    def callback_camr(self, ros_image):
        self.callback_cam(ros_image,1)

    def callback_cam(self, ros_image, index):
        # ignore empty frames which occur sometimes during parameter changes
        if len(ros_image.data) == 0:
            print("dropped empty camera frame")
            return
        try:
            # convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # store image for display
            self.camera[index] = image
        except CvBridgeError as e:
            pass
    
    # continuously moves the head around to better guage it's surroundings
    def head_move(self, *args):
        if self.miro_found: return
        if self.move_head:
            self.kin.position[2] = self.sens_kin.position[2]+np.radians(self.head_direction)
            if abs(self.kin.position[2]) > np.radians(30):
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
                self.map_start[1]-round((posy-self.starting_pose[1])*MAP_SCALE)],dtype=int),0,MAP_SIZE-1)
        
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
            if self.miro_found: return
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
            obj_vec = dist*np.array([np.cos(self.orientation+self.sens_kin.position[2]),
                                        np.sin(self.orientation+self.sens_kin.position[2])])
            
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
                selected_map = self.prob_map[max(map_coords[0]-OBSTACLE_SIZE-1,0):min(map_coords[0]+OBSTACLE_SIZE+2,self.prob_map.shape[0]),
                                        max(map_coords[1]-OBSTACLE_SIZE-1,0):min(map_coords[1]+OBSTACLE_SIZE+2,self.prob_map.shape[1])]
                self.prob_map[max(map_coords[0]-OBSTACLE_SIZE-1,0):min(map_coords[0]+OBSTACLE_SIZE+2,self.prob_map.shape[0]),
                        max(map_coords[1]-OBSTACLE_SIZE-1,0):min(map_coords[1]+OBSTACLE_SIZE+2,self.prob_map.shape[1])] = Helper.increase_prob(selected_map,True,0)
                
            # draws in the graph
            if self.display is not None:
                display_map = cv2.cvtColor(self.prob_map,cv2.COLOR_GRAY2RGB)
                display_map[self.map_pos[0],self.map_pos[1]] = np.array([255,0,0])
                self.display.set_data(display_map)
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
                if dist < 0.4 and abs(self.sens_kin.position[2]) < 0.6:
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
                self.move_head = True


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
    

    def letterbox(img, new_shape):
        shape = img.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # Compute padding
        new_unpad = round(shape[1] * r), round(shape[0] * r)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

        
    def detect_miro(self, *args):
        """Uses computer vision to detect the miro
        Inputs: Camera
        Outputs: Bool
        """
        self.miro_found = True
        self.move_head = False
        self.kin.position = [0.0, math.radians(50.0), math.radians(0), math.radians(4.0)]
        self.interface.msg_kin_joints.set(self.kin,1)
        rospy.sleep(0.6)
        camera_images = self.camera.copy()
        for index, img in enumerate(camera_images):
            classes = ["0","45","90","135","180","225","270","315"]
            image = img.copy()
            img_height, img_width = image.shape[:2]
            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img, pad = Helper.letterbox(img, (640, 640))

            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0

            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
            outputs = self.onnx_model.run(None, {"images": image_data})
            results = outputs[0]
            results = results.transpose()
            res1 = np.argmax(results[:,4:])//8
            result = results[res1]
            conf = np.max(result[4:])
            if conf > 0.5:
                print("found miro")
                self.timer3.shutdown()
                self.final_map = self.prob_map<150
                self.timer4 = rospy.Timer(rospy.Duration(0.5), self.pose_detection)
                self.timer5 = rospy.Timer(rospy.Duration(0.1), self.look_miro)
                self.timer5 = rospy.Timer(rospy.Duration(0.5), self.move_miro)
                self.timer6 = rospy.Timer(rospy.Duration(1), self.Astar)
                self.timer7 = rospy.Timer(rospy.Duration(0.5), self.lead_miro)
                self.timer8 = rospy.Timer(rospy.Duration(1), self.helperAstar)
                self.send_audio("found")
                break
            if index == 0:
                self.camera_display1.set_data(image)#
            else:
                self.camera_display2.set_data(image)#
        else:
            self.miro_found = False
            self.kin.position = [0.0, math.radians(30.0), 0.0, math.radians(-10.0)]
            self.move_head = True


    
    def pose_detection(self, *args):
        """Uses computer vision to detect the pose of the miro
        Inputs: Pose, Camera
        Outputs: Pose
        """
        if type(None) in map(type,self.camera):
            return
        try:
            for index, img in enumerate(self.camera):
                classes = ["0","45","90","135","180","225","270","315"]
                image = img.copy()
                img_height, img_width = image.shape[:2]
                # Convert the image color space from BGR to RGB
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img, pad = Helper.letterbox(img, (640, 640))

                # Normalize the image data by dividing it by 255.0
                image_data = np.array(img) / 255.0

                # Transpose the image to have the channel dimension as the first dimension
                image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

                # Expand the dimensions of the image data to match the expected input shape
                image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
                outputs = self.onnx_model.run(None, {"images": image_data})
                results = outputs[0]
                results = results.transpose()
                res1 = np.argmax(results[:,4:])//8
                result = results[res1]
                class_id = np.argmax(result[4:])
                conf = np.max(result[4:])
                dimentions = np.array([img_width,img_height])
                padding = 640-dimentions/dimentions.max()*640
                bbox = np.array([[result[0]-result[2]/2-padding[0]/2,result[0]+result[2]/2-padding[0]/2],
                                [result[1]-result[3]/2-padding[1]/2,result[1]+result[3]/2-padding[1]/2]])
                bbox = bbox.reshape(-1,2)*np.array([img_width/(640-padding[0]),img_height/(640-padding[1])]).reshape(1,2)
                bbox = bbox.astype(int).T.reshape(-1)*640//img_width
                # self.pos = np.array([self.pos.x,self.pos.y])
                # other_vec = np.array([self.pos2.x,self.pos2.y])
                pred_dist = np.sqrt(120/((bbox[3]-bbox[1])))/np.cos(self.sens_kin.position[2])

                image = cv2.resize(image.copy(),(640,360))
                cv2.rectangle(image, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0, 0),2)
                cv2.putText(image,classes[class_id]+' '+f"{conf:.2f} {pred_dist:.2f}",(bbox[0],bbox[1]+20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)            
                if conf > 0.45:
                    other_angle = (self.orientation+np.radians(int(classes[class_id]))-np.pi+self.sens_kin.position[2])%(np.pi*2)
                    self.distance_queue.append(pred_dist)
                    self.angle_queue.append(other_angle)
                    if len(self.distance_queue) > 5:
                        self.distance_queue = self.distance_queue[1:]
                    if len(self.angle_queue) > 5:
                        self.angle_queue = self.angle_queue[1:]
                    # print(self.sens_kin.position[2])
                    # print(bbox[3]-bbox[1],np.round(pred_dist,2) , np.round(np.linalg.norm(self.pos-other_vec),2), np.mean(self.distance_queue))
                    avg_dist = np.mean(self.distance_queue)
                    # print(np.round(self.pos,2))
                    # print(classes[class_id], other_angle, np.round(self.pos.theta%(np.pi*2),2))
                    # print(np.round(np.median(self.angle_queue),2), np.round(np.median(self.distance_queue),2))
                    # print(np.round(self.pos.theta%(2*np.pi),2),np.round(np.linalg.norm(self.pos-other_vec),2))
                    self.pred_dist[index] = np.mean(self.distance_queue)
                    self.midpoints[index] = np.array([bbox[2]+bbox[0],bbox[3]+bbox[1]])/2
                else:
                    self.pred_dist[index] = None
                    self.midpoints[index] = None
                if index == 0:
                    self.camera_display1.set_data(image)#
                else:
                    self.camera_display2.set_data(image)#
        except Exception as e:
            print("unexpected error occured", e)
        plt.draw()
    
    def look_miro(self, *args):
        if type(self.camera[0]) == type(None):
            return
        
        h,w,_ = self.camera[0].shape        
        cdist = 0
        cdisty = 0
        # print(self.midpoints)
        if type(self.midpoints[0]) != type(None) and type(self.midpoints[1]) != type(None):
            cdist = ((3*w/4 - self.midpoints[0][0])+(w/4 - self.midpoints[1][0]))/2
            cdisty = ((h - self.midpoints[0][1]- self.midpoints[1][1]))/2
        elif type(self.midpoints[0]) != type(None):
            cdist = 3*w/4 - self.midpoints[0][0]
            cdisty = h/2 - self.midpoints[0][1]
        elif type(self.midpoints[1]) != type(None):
            cdist = w/4 - self.midpoints[1][0]
            cdisty = h/2 - self.midpoints[1][1]
        # print(cdist,cdisty)
        
        pred_dist = None
        if abs(cdist) > 70:
            self.pred_pos = None
        if abs(cdist) < 50:
            self.velocity.twist.angular.z = 0.0
            if self.pred_dist[0] is not None and self.pred_dist[1] is not None:
                pred_dist = np.mean(self.pred_dist)
            elif self.pred_dist[0] is not None:
                pred_dist = self.pred_dist[0]
            elif self.pred_dist[1] is not None:
                pred_dist = self.pred_dist[1]
            
            if len(self.angle_queue) > 0 and pred_dist is not None:
                # calculate other miro's position if looking at it
                self.pred_angle = np.median(self.angle_queue)
                displacement = pred_dist*np.array([np.cos(self.orientation+self.sens_kin.position[2]),
                                                            np.sin(self.orientation+self.sens_kin.position[2])])
                self.pred_pos = self.pos+displacement

                self.pred_map_pos = self.pos2map(self.pred_pos[0],self.pred_pos[1])

                maxarg = np.argmax(abs(displacement))
                scan_range = np.arange(0.0,abs(displacement[maxarg])+1/MAP_SCALE,1/MAP_SCALE)
                # for every cell between the miro and the object, set as empty
                for i in scan_range:
                    cur_coord = i*displacement/abs(displacement[maxarg])
                    cur_map = self.pos2map(*(cur_coord+self.pos)).astype(int)
                    coords = [max(cur_map[0]-BODY_SIZE,0),
                            min(cur_map[0]+BODY_SIZE+1,self.final_map.shape[0]),
                            max(cur_map[1]-BODY_SIZE,0),
                            min(cur_map[1]+BODY_SIZE+1,self.final_map.shape[1])]
                    self.final_map[coords[0]:coords[1],coords[2]:coords[3]] = False

                new_map = cv2.cvtColor((1-self.final_map.astype(np.uint8))*255,cv2.COLOR_GRAY2RGB)
                new_map[self.map_pos[0],self.map_pos[1]] = np.array([255,0,0])
                new_map[self.pred_map_pos[0],self.pred_map_pos[1]] = np.array([0,0,255])
                self.display.set_data(new_map)
                # print("pos", self.pred_pos.round(2))
                # print("real pos", np.round([self.pos.x,self.pos.y],2))
        elif self.sens_kin.position[2] < math.radians(25) and cdist > 0:
            self.kin.position[2] = self.sens_kin.position[2]+math.radians(15)
            # print("turning left",self.sens_kin.position[2])
        elif self.sens_kin.position[2] > -math.radians(25) and cdist < 0:
            # print(cdist,self.sens_kin.position[2])
            self.kin.position[2] = self.sens_kin.position[2]+math.radians(15)*np.sign(cdist)
            # print("turning right", self.kin.position[2], self.sens_kin.position[2])
            # self.velocity.twist.angular.z = 0.6
        else:
            self.velocity.twist.angular.z = 1.0*np.sign(cdist)
            # print("moving body")
            self.kin.position[2] = 0.0#self.sens_kin.position[2]-math.radians(1)*np.sign(cdist)


            # self.kin.position[2] = self.sens_kin.position[2]-math.radians(1)
            # self.velocity.twist.angular.z = -0.6
        # else:
        #     self.velocity.twist.angular.z = 0.0
        #     pred_dist = None
        
        # print("pred dist: ", pred_dist)
        if pred_dist is None:
            self.velocity.twist.linear.x = 0.0
        elif pred_dist > 0.9:
            # print("moving forwards")
            self.velocity.twist.linear.x = 0.1
        elif pred_dist < 0.7:
            # print("moving backwards")
            self.velocity.twist.linear.x = -0.1
        
        else: 
            self.velocity.twist.linear.x = 0.0
        

        self.interface.msg_cmd_vel.set(self.velocity,0.1)
        if abs(cdisty) < 20:
            pass
        elif cdisty < 0:
            # print("looking down", self.sens_kin.position[3], math.radians(4))
            if self.sens_kin.position[3] < math.radians(5):
                self.kin.position[3] = np.clip(self.sens_kin.position[3]+math.radians(5), math.radians(-8), math.radians(5))
            else:
                self.kin.position[1] = np.clip(self.sens_kin.position[1]+math.radians(10), math.radians(45), math.radians(55))
        else:
            # print("looking up")
            if self.sens_kin.position[3] > -math.radians(5):
                self.kin.position[3] = np.clip(self.sens_kin.position[3]-math.radians(5), math.radians(-8), math.radians(5))
            else:
                self.kin.position[1] = np.clip(self.sens_kin.position[1]-math.radians(10), math.radians(45), math.radians(55))
        # self.kin.position[3] = np.clip(self.kin.position[3],math.radians(-15),math.radians(15))                   
        self.interface.msg_kin_joints.set(self.kin,0.1)
        # self.pub_cmd_vel2.publish(self.velocity2)

    def heuristic(self,pos1,pos2):
        return np.max(abs(pos1-pos2))

    def Astar(self, *args):
        """Uses A* to determine the miro's next path
        Inputs: Map
        Outputs: Path
        """
        if self.pred_map_pos is None:
            return

        start = self.pred_map_pos
        goal = self.map_start
        width, height = self.final_map.shape
        grid = self.final_map

        # if not self.isValid(start[0],start[1],width,height) or not self.isValid(goal[0],goal[1],width,height):
        #     return "Source or destinaton is invalid"
        
        # checks if the miro is near it's destination
        if (start-goal<2).all():
            return "Already at destination"
        
        # closedList = [[False for _ in range(height)] for _ in range(width)]
        closeList = set()
        # cellDetails = [[Cell() for _ in range(height)] for _ in range(width)]
        cellDetails = np.zeros((self.final_map.shape[0],self.final_map.shape[1],5), dtype=int)-1

        f,g,h,parentx,parenty = range(5)
        # initilises the starting node
        x = start[0]
        y = start[1]
        cellDetails[x,y,g] = 0
        cellDetails[x,y,h] = self.heuristic(start,goal)
        cellDetails[x,y,f] = cellDetails[x,y,h]+cellDetails[x,y,g]
        cellDetails[x,y,parentx] = x
        cellDetails[x,y,parenty] = y

        openList = []
        heapq.heappush(openList,(0.0,x,y))
        foundGoal = False

        while len(openList) > 0 and not foundGoal:
            p = heapq.heappop(openList)

            x = p[1]
            y = p[2]
            closeList.update((x,y))

            directions = [(0,1),(1,0),(0,-1),(-1,0)]
            for dir in directions:
                xNew = x + dir[0]
                yNew = y + dir[1]
                if xNew < 0 or yNew < 0: continue
                if xNew >= width or yNew >= height: continue
                    
                if (xNew,yNew) in closeList:
                    continue

                if not grid[xNew,yNew]:
                    # checks if it is the goal cell
                    if (np.array([xNew,yNew])==goal).all():
                        cellDetails[xNew,yNew,parentx] = x
                        cellDetails[xNew,yNew,parenty] = y
                        print("Destination Found")
                        foundGoal = True
                        break
                        #return self.tracePath(cellDetails, goal)
                    # adds cell to the list
                    else:
                        gNew = cellDetails[x,y,g] +1
                        hNew = self.heuristic(np.array([xNew,yNew]),goal)
                        fNew = gNew + hNew

                        if cellDetails[xNew,yNew,f] == -1 or cellDetails[xNew,yNew,f] > fNew:
                            heapq.heappush(openList,(fNew,xNew,yNew))
                            cellDetails[xNew,yNew,f] = fNew   
                            cellDetails[xNew,yNew,g] = gNew   
                            cellDetails[xNew,yNew,h] = hNew   
                            cellDetails[xNew,yNew,parentx] = x
                            cellDetails[xNew,yNew,parenty] = y
            
        if not foundGoal:       
            print("Failed to find destination")
            return []
        else:
            print(cellDetails[goal[0],goal[1],parentx:parenty+1])
            path = [cellDetails[goal[0],goal[1],parentx:parenty+1].reshape(2)]
            while cellDetails[path[-1][0],path[-1][1],g] > 0:
                print(path,np.array([cellDetails[path[-1][0],path[-1][1],parentx:parenty+1]]))
                path.append(cellDetails[path[-1][0],path[-1][1],parentx:parenty+1])
        self.path = path
        display_map = cv2.cvtColor((1-self.final_map.astype(np.uint8))*255, cv2.COLOR_GRAY2RGB)
        for i in path:
            display_map[i[0],i[1]] = np.array([255,0,0])
        self.display4.set_data(display_map)
        plt.draw()
    
    def helperAstar(self, *args):
        if self.pred_map_pos is None:
            return

        start = self.path[10]
        goal = self.path[-1]
        width, height = self.final_map.shape
        grid = self.final_map

        # if not self.isValid(start[0],start[1],width,height) or not self.isValid(goal[0],goal[1],width,height):
        #     return "Source or destinaton is invalid"
        
        # checks if the miro is near it's destination
        if (start-goal<2).all():
            return "Already at destination"
        
        # closedList = [[False for _ in range(height)] for _ in range(width)]
        closeList = set()
        # cellDetails = [[Cell() for _ in range(height)] for _ in range(width)]
        cellDetails = np.zeros((self.final_map.shape[0],self.final_map.shape[1],5), dtype=int)-1

        f,g,h,parentx,parenty = range(5)
        # initilises the starting node
        x = start[0]
        y = start[1]
        cellDetails[x,y,g] = 0
        cellDetails[x,y,h] = self.heuristic(start,goal)
        cellDetails[x,y,f] = cellDetails[x,y,h]+cellDetails[x,y,g]
        cellDetails[x,y,parentx] = x
        cellDetails[x,y,parenty] = y

        openList = []
        heapq.heappush(openList,(0.0,x,y))
        foundGoal = False

        while len(openList) > 0 and not foundGoal:
            p = heapq.heappop(openList)

            x = p[1]
            y = p[2]
            closeList.update((x,y))

            directions = [(0,1),(1,0),(0,-1),(-1,0)]
            for dir in directions:
                xNew = x + dir[0]
                yNew = y + dir[1]
                if xNew < 0 or yNew < 0: continue
                if xNew >= width or yNew >= height: continue
                    
                if (xNew,yNew) in closeList:
                    continue

                if not grid[xNew,yNew]:
                    # checks if it is the goal cell
                    if (np.array([xNew,yNew])==goal).all():
                        cellDetails[xNew,yNew,parentx] = x
                        cellDetails[xNew,yNew,parenty] = y
                        print("Destination Found")
                        foundGoal = True
                        break
                        #return self.tracePath(cellDetails, goal)
                    # adds cell to the list
                    else:
                        gNew = cellDetails[x,y,g] +1
                        hNew = self.heuristic(np.array([xNew,yNew]),goal)
                        fNew = gNew + hNew

                        if cellDetails[xNew,yNew,f] == -1 or cellDetails[xNew,yNew,f] > fNew:
                            heapq.heappush(openList,(fNew,xNew,yNew))
                            cellDetails[xNew,yNew,f] = fNew   
                            cellDetails[xNew,yNew,g] = gNew   
                            cellDetails[xNew,yNew,h] = hNew   
                            cellDetails[xNew,yNew,parentx] = x
                            cellDetails[xNew,yNew,parenty] = y
            
        if not foundGoal:       
            print("Failed to find destination")
            return []
        else:
            print(cellDetails[goal[0],goal[1],parentx:parenty+1])
            path = [cellDetails[goal[0],goal[1],parentx:parenty+1].reshape(2)]
            while cellDetails[path[-1][0],path[-1][1],g] > 0:
                print(path,np.array([cellDetails[path[-1][0],path[-1][1],parentx:parenty+1]]))
                path.append(cellDetails[path[-1][0],path[-1][1],parentx:parenty+1])
        self.helper_path = path        
        display_map = cv2.cvtColor((1-self.final_map.astype(np.uint8))*255, cv2.COLOR_GRAY2RGB)
        for i in path:
            display_map[i[0],i[1]] = np.array([255,0,0])
        self.display4.set_data(display_map)
        plt.draw()


    def lead_miro(self, *args):
        """Moves ahead of the lost miro and follows the path generates one step ahead
        """
        next_position = self.helper_path.reverse[-1]
        angle_difference = np.arctan2(next_position[1]-self.pred_map_pos[1],next_position[0]-self.pred_map_pos[0])
        if angle_difference - self.pred_angle > 3:
            self.velocity2.twist.angular.z = 0.5
            return
        self.velocity2.twist.angular.z = 0.0
        euclidean_difference = np.linalg.norm(next_position - self.pred_map_pos)
        if euclidean_difference > 0.1:
            self.velocity2.twist.linear.x = 0.15
            return
        self.velocity2.twist.linear.x = 0.0
        self.helper_moving = False


        
    def move_miro(self, *args):
        """Determines what signals to give the miro
        """
        if self.pred_pos is None: #or len(self.other_path)==0:
            # self.velocity2.twist.linear.x = 0.0
            # self.velocity2.twist.angular.z = 0.0
            # self.pub_cmd_vel2.publish(self.velocity2)
            self.send_audio("stop")
            return
        target_pos = self.map2pos(self.map_start[0],self.map_start[1])#self.other_path[0]+np.array([-1.0,0.0])
        # self.velocity2.twist.linear.x = 0.15
        target = np.arctan2(target_pos[1]-self.pred_pos[1],target_pos[0]-self.pred_pos[0])
        
        # calculates the differance in angle between current and target
        dists = [(self.pred_angle%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pred_angle%(2*np.pi))%(2*np.pi)]

        # checks if the robot is close to the next node in the path 
        if np.linalg.norm(target_pos-self.pred_pos) < 0.3:
            # self.velocity2.twist.linear.x = 0.0
            # self.velocity2.twist.angular.z = 0.0
            # self.other_path = self.other_path[1:]
            self.send_audio("stop")
        
        # checks if the robot is looking in the right direction
        elif min(dists) < 0.5:
            # self.velocity2.twist.angular.z = 0.0
            self.send_audio("forwards")
            # self.pub_cmd_vel2.publish(self.velocity2)
        # moves clockwise if the right angle is lower
        elif dists[0] >= dists[1]:
            # self.velocity2.twist.angular.z = 0.5
            # self.velocity2.twist.linear.x = 0.01
            self.send_audio("turn left")
        # moves counter clockwise otherwise
        else:
            # self.velocity2.twist.angular.z = -0.5
            # self.velocity2.twist.linear.x = 0.01
            # print("turning right")
            self.send_audio("turn right")

    def send_audio(self, command):
        """Send audio signal
        Inputs: Command
        Outputs: Audio
        """
        return
        msg = UInt16MultiArray()
        if command == "found":
            # msg.data = [1200, 128, 1000]
            self.interface.post_tone(1200, 25, 10)
        elif command == "forwards":
            # msg.data = [1600, 128, 1000]
            self.interface.post_tone(1600, 25, 10)
        elif command == "stop":
            # msg.data = [2000, 128, 1000]
            self.interface.post_tone(2000, 25, 10)
        elif command == "turn left":
            # msg.data = [2400, 128, 1000]
            self.interface.post_tone(2400, 25, 10)
        elif command == "turn right":
            # msg.data = [2800, 128, 1000]
            self.interface.post_tone(2800, 25, 10)
        # self.interface.post_tone(2800, 128, 1000)
        print(command)
        # self.pub_tone.publish(msg)


if __name__ == '__main__':
    main = Helper()
    if(rospy.get_node_uri()):
        pass
    else:
        rospy.init_node("Helper_miro")
    rospy.spin()