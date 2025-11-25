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
import miro2 as miro
import time

import onnxruntime

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
class LookMiro:
    
    def __init__(self):
        base1 = "/miro01"
        base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pos = Pose2D()
        self.cur_target = 0
        self.timer1 = time.time_ns()
        self.timer2 = time.time_ns()

        self.midpoints = [None,None]
        self.kin = JointState()
        self.onnx = onnxruntime.InferenceSession("best.onnx")

        self.pub_cmd_vel = rospy.Publisher(base2 + "/control/cmd_vel", TwistStamped, queue_size=0)
        # self.pub_cos = rospy.Publisher(basename + "/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        # self.pub_illum = rospy.Publisher(basename + "/control/illum", UInt32MultiArray, queue_size=0)
        self.pub_kin = rospy.Publisher(base2 + "/control/kinematic_joints", JointState, queue_size=0)
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

        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(40.0), 0.0, 0.0]
        self.pub_kin.publish(self.kin)

        
        self.timer = rospy.Timer(rospy.Duration(0.1), self.match_image)
        # self.timer2 = rospy.Timer(rospy.Duration(0.1), self.look_miro)


        plt.subplot(121)
        self.display1 = plt.imshow(np.array([[0.0]]), 'gray')
        plt.subplot(122)
        self.display2 = plt.imshow(np.array([[0.0]]), 'gray')
        plt.show()

        
    def callback_pose(self, pose):
        if pose != None:
            self.pos = pose
            # print(self.pos.theta%(2*np.pi))

        
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
            # swallow error, silently
            #print(e)
            pass

    def match_image(self, *args):
        if type(None) in map(type,self.camera):
            return
        for index, img in enumerate(self.camera):
            # img2 = cv2.imread('pictures/135_1.png', cv2.IMREAD_GRAYSCALE) # trainImage
            
            img_w, img_h = img.shape[1], img.shape[0]

            processed_image = cv2.resize(img, (640, 640))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_image = processed_image.transpose(2, 0, 1)
            processed_image = processed_image.reshape(1, 3, 640, 640)

            processed_image = processed_image/255.0
            processed_image = processed_image.astype(np.float32)
            classes = ["135","15","165","195","225","255","315","345","45","75", "background"]
            outputs = np.array(self.onnx.run(None, {"images": processed_image})).flatten()
            arg = np.argmax(outputs)
            print(outputs)
            print(index, arg, classes[arg], outputs[arg], len(outputs))
            text_img = cv2.putText(img.copy(), classes[arg], (100,100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3, color = (250,0,100),thickness=3)
            
            if index == 0:
                self.display1.set_data(text_img)#
            else:
                self.display2.set_data(text_img)#
            plt.draw()
        

        
        
    def loop(self):
        pass

        
if __name__ == "__main__":
    try:
        main = LookMiro()
        if(rospy.get_node_uri()):
            pass
        else:
            rospy.init_node("look_miro")
        while not rospy.core.is_shutdown():
            main.loop()
        main.interface.disconnect()
        exit()
    except Exception as e:
        print("exception",e)
        exit()


