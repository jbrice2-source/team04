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

class TakeImages:
    
    def __init__(self):
        base1 = "/miro01"
        # base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        # self.velocity# = TwistStamped()
        
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pos = Pose2D()
        self.pos = Pose2D()
        self.targets = [0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]
        self.cur_target = None
        self.time1 = time.time_ns()
        self.add_dist = 0.0
        self.camera_interval  = time.time_ns()
        self.kin = JointState()
        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(50.0), 0.0, 0.0]
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
        # self.sub_package = rospy.Subscriber(base2 + "/sensors/package",
        #             miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
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

        


        self.timer = rospy.Timer(rospy.Duration(0.1), self.random_move)
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.camera_move)



        
    def callback_pose(self, pose):
        if pose != None:
            self.pos = pose
            # print(self.pos.theta%(2*np.pi))
    # def callback_pose2(self, pose):
    #     if pose != None:
    #         self.pos = pose

        
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
        
    def random_move(self, *args):
        self.velocity.twist.linear.x = 0.2
        
        if np.linalg.norm([self.pos.x,self.pos.y]) > 0.9-self.add_dist:
            self.velocity.twist.linear.x = -0.18
            self.velocity.twist.angular.z = 0.0
            self.add_dist = 0.1
            self.cur_target = None
        # elif np.linalg.norm([self.pos.x-self.pos.x,self.pos.y-self.pos.y]) < 0.35+self.add_dist:
        #     self.velocity.twist.linear.x = -0.18
        #     self.velocity.twist.angular.z = 0.0
        #     self.add_dist = 0.1
        #     self.cur_target = None
        elif time.time_ns()-self.time1 > 1e10 or self.cur_target == None:
            self.add_dist = 0.0
            self.cur_target = np.pi*2*(np.random.rand(1))
            print("changing target",time.time_ns()-self.time1, self.cur_target)
            self.time1 = time.time_ns()
        else:
            self.add_dist = 0.0
            np.arctan2(self.pos.x-xcord,self.pos.y-ycord)
            target = self.cur_target
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
        
    def camera_move(self, *args):
        self.pub_kin.publish(self.kin)
        # self.pub_cos.publish(self.cos_joints)
        # target = np.arctan2(self.pos.y-self.pos.y,self.pos.x-self.pos.x)%(2*np.pi)#self.targets[self.cur_target]*np.pi/180
          
        # dists = [(self.pos.theta%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pos.theta%(2*np.pi))%(2*np.pi)]
        # print(self.pos.theta, np.round(dists,2), np.argmin(dists), target, self.velocity#.twist.angular.z)

        # if min(dists) < 0.2:
        #     self.velocity#.twist.angular.z = 0.0
        #     self.pub_cmd_vel2.publish(self.velocity#)
            # rospy.sleep(2)
        if time.time_ns()-self.camera_interval > 5e9:
            print("taking picture", )
            self.camera_interval = time.time_ns()
            self.take_picture()
        # elif dists[0] >= dists[1]:
        #     self.velocity#.twist.angular.z = 0.4
        #     # print("turning left")
        # else:
        #     self.velocity#.twist.angular.z = -0.4
        #     # print("turning right")
    
        # self.pub_cmd_vel2.publish(self.velocity#)
        
    def take_picture(self):
        if not os.path.isdir("neg_dataset"):
            os.mkdir("neg_dataset")  
        images = self.camera.copy()
        
        if not images[0] is None and not images[1] is None:
            for i in range(len(images)):
                # print(images[i])
                img_id = uuid.uuid4().hex
                # id, camera index, pos1x, pos1y, pos1theta, posx, posy, postheta
                # params = [img_id,i,self.pos.x,self.pos.y,self.pos.theta,self.pos.x,self.pos.y,self.pos.theta]
                cv2.imwrite(f"neg_dataset/{img_id}.png", images[i])
                # with open("dataset/labels.csv","a") as file:
                #     file.write(",".join(str(e) for e in params)+"\n")
        
    def loop(self):
        pass


        
if __name__ == "__main__":
    try:
        main = TakeImages()
        if(rospy.get_node_uri()):
            pass
        else:
            rospy.init_node("take_images")
        while not rospy.core.is_shutdown():
            main.loop()
        main.interface.disconnect()
        exit()
    except Exception as e:
        print("exception",e)


