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


from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation



class TakeImages:
    
    def __init__(self):
        base1 = "/miro01"
        base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pose = Pose2D()
        self.target = 90
        self.targets = [0,90,180,270]
        self.cur_target = 0
        
        

        self.pub_cmd_vel = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0)
        # self.pub_cos = rospy.Publisher(basename + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        # self.pub_illum = rospy.Publisher(basename + "/control/illum", UInt32MultiArray, queue_size=0)
        # self.pub_kin = rospy.Publisher(basename + "/control/kinematic_joints", JointState, queue_size=0)
        # self.pub_tone = rospy.Publisher(basename + "/control/tone", UInt16MultiArray, queue_size=0)
        # self.pub_command = rospy.Publisher(basename + "/control/command", String, queue_size=0)

        # subscribers
        # self.sub_package = rospy.Subscriber(base2 + "/sensors/package",
        #             miro.msg.sensors_package, self.callback_package, queue_size=1, tcp_nodelay=True)
        self.pose = rospy.Subscriber(base1 + "/sensors/body_pose",
            Pose2D, self.callback_pose, queue_size=1, tcp_nodelay=True)
        # self.sub_mics = rospy.Subscriber(basename + "/sensors/mics",
        #             Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(base2 + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(base2 + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

        

        
    def callback_pose(self, pose):
        if pose != None:
            self.pose = pose
            target = self.targets[self.cur_target]*np.pi/180
            if pose.theta < target-0.1:
                self.velocity.twist.angular.z = 0.5
            elif pose.theta > target+0.1:
                self.velocity.twist.angular.z = -0.5
            else: 
                self.velocity.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.velocity)

        
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
            orb = cv2.ORB_create()
            kp , des= orb.detectAndCompute(image,None)
            image = cv2.drawKeypoints(image,kp,None,color=(0,0,255),flags=0)

            # store image for display
            self.camera[index] = image


        except CvBridgeError as e:

            # swallow error, silently
            #print(e)
            pass
        
    def loop(self):
        if self.pose.theta == self.targets[self.cur_target]*np.pi/180:
            if not os.path.isdir("pictures"):
                os.mkdir("pictures")
            
            img1 = self.camera[0]
            if type(img1) != None:
                print(type(img1))
                cv2.imwrite("pictures/{targets[self.cur_target]}.png", img1)
                self.cur_target += 1
        
        
if __name__ == "__main__":
    main = TakeImages()
    if(rospy.get_node_uri()):
        pass
    else:
        rospy.init_node("take_images")
    rospy.sleep(200)
    while not rospy.core.is_shutdown():
        main.loop()
    self.interface.disconnect()


