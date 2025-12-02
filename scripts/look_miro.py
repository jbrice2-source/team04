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

import onnxruntime as ort

from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
MIN_MATCH_COUNT = 10
images = glob("cropped_pictures/*.png")

image_data = np.empty_like(images,dtype=tuple)

# Initiate SIFT detector
sift = cv2.SIFT_create()
for i,n in enumerate(images):
    img1 = cv2.imread(n, cv2.IMREAD_GRAYSCALE)          # queryImage
    kp1, des1 = sift.detectAndCompute(img1,None)
    image_data[i] = (img1,kp1,des1)


class LookMiro:
    
    def __init__(self):
        base1 = "/miro01"
        base2 = "/miro02"
        self.interface = miro.lib.RobotInterface()

        self.velocity = TwistStamped()
        self.velocity2 = TwistStamped()

        
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pos = Pose2D()
        self.pos2 = Pose2D()
        self.cur_target = 0
        # self.timer1 = time.time_ns()
        # self.timer2 = time.time_ns()
        self.pred_dist = [None, None]
        self.midpoints = [None,None]
        self.kin = JointState()
        self.distance_queue = []
        self.angle_queue = []
        mode_path = "best_new.onnx"
        self.onnx_model = ort.InferenceSession(mode_path)
        self.package = None
        self.other_path = [np.array([-1.0,0.0]),np.array([0.0,0.5]),np.array([1.0,-0.5]),np.array([0.5,0.5]),np.array([-0.5,-0.5])]
        
        self.pred_pos = None
        self.pred_angle = None
        

        self.pub_cmd_vel = rospy.Publisher(base2 + "/control/cmd_vel", TwistStamped, queue_size=0)
        self.pub_cmd_vel2 = rospy.Publisher(base1 + "/control/cmd_vel", TwistStamped, queue_size=0, latch=True)
        
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
        self.pose2 = rospy.Subscriber(base2 + "/sensors/body_pose",
            Pose2D, self.callback_pose2, queue_size=1, tcp_nodelay=True)
        self.package = rospy.Subscriber(base2 + "/sensors/package",
            miro.msg.sensors_package, self.package_callback, queue_size=1, tcp_nodelay=True)
        # self.sub_mics = rospy.Subscriber(basename + "/sensors/mics",
        #             Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(base2 + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(base2 + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(50.0), math.radians(0), math.radians(-10.0)]
        self.pub_kin.publish(self.kin)
        
        self.timer = rospy.Timer(rospy.Duration(0.5), self.match_image)
        
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.look_miro)
        self.timer3 = rospy.Timer(rospy.Duration(1.0), self.move_miro)
        self.timer4 = rospy.Timer(rospy.Duration(0.1), self.vel_publish)


        plt.subplot(121)
        self.display1 = plt.imshow(np.array([[0.0]]), 'gray')
        plt.subplot(122)
        self.display2 = plt.imshow(np.array([[0.0]]), 'gray')
        plt.show()

    def vel_publish(self, *args):
        self.pub_cmd_vel2.publish(self.velocity2)
        
        
    def package_callback(self, package):
        self.package = package
        
    def callback_pose(self, pose):
        if pose != None:
            self.pos = pose
            
    def callback_pose2(self, pose):
        if pose != None:
            self.pos2 = pose
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

            classes = ["0","45","90","135","180","225","270","315"]
            image = img.copy()
            
            img_height, img_width = image.shape[:2]

            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img, pad = LookMiro.letterbox(img, (640, 640))

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

            pos_vec = np.array([self.pos.x,self.pos.y])
            other_vec = np.array([self.pos2.x,self.pos2.y])
            pred_dist = np.sqrt(90/((bbox[3]-bbox[1])))/np.cos(self.package.kinematic_joints.position[2])

            image = cv2.resize(image.copy(),(640,360))
            cv2.rectangle(image, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0, 0),2)
            cv2.putText(image,classes[class_id]+' '+f"{conf:.2f} {pred_dist:.2f}",(bbox[0],bbox[1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
            # print(conf, index)

            
            if conf > 0.3:
                other_angle = (self.pos2.theta+np.radians(int(classes[class_id]))-np.pi+self.package.kinematic_joints.position[2])%(np.pi*2)
                self.distance_queue.append(pred_dist)
                self.angle_queue.append(other_angle)
                if len(self.distance_queue) > 5:
                    self.distance_queue = self.distance_queue[1:]
                if len(self.angle_queue) > 5:
                    self.angle_queue = self.angle_queue[1:]
                # print(self.package.kinematic_joints.position[2])
                # print(bbox[3]-bbox[1],np.round(pred_dist,2) , np.round(np.linalg.norm(pos_vec-other_vec),2), np.mean(self.distance_queue))
                avg_dist = np.mean(self.distance_queue)
                # print(np.round(pos_vec,2))
                # print(classes[class_id], other_angle, np.round(self.pos.theta%(np.pi*2),2))
                # print(np.round(np.median(self.angle_queue),2), np.round(np.median(self.distance_queue),2))
                # print(np.round(self.pos.theta%(2*np.pi),2),np.round(np.linalg.norm(pos_vec-other_vec),2))
                self.pred_dist[index] = np.mean(self.distance_queue)
                self.midpoints[index] = np.array([bbox[2]+bbox[0],bbox[3]+bbox[1]])/2
            else:
                self.pred_dist[index] = None
                self.midpoints[index] = None
            
            if index == 0:
                self.display1.set_data(image)#
            else:
                self.display2.set_data(image)#
        # except:
        #     print("unexpected error occured")
        plt.draw()
        
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
        if abs(cdist) > 100:
            self.pred_pos = None
        if abs(cdist) < 70:
            self.velocity.twist.angular.z = 0.0
            if self.pred_dist[0] is not None and self.pred_dist[1] is not None:
                pred_dist = np.mean(self.pred_dist)
            elif self.pred_dist[0] is not None:
                pred_dist = self.pred_dist[0]
            elif self.pred_dist[1] is not None:
                pred_dist = self.pred_dist[1]
            
            if len(self.angle_queue) > 0 and pred_dist is not None:
                self.pred_angle = np.median(self.angle_queue)
                pos_vec = np.array([self.pos2.x,self.pos2.y])
                self.pred_pos = pos_vec+pred_dist*np.array([np.cos(self.pos2.theta+ self.package.kinematic_joints.position[2]),
                                                            np.sin(self.pos2.theta+self.package.kinematic_joints.position[2])])
                print("pos", self.pred_pos.round(2))
                # print("real pos", np.round([self.pos.x,self.pos.y],2))
        elif self.package.kinematic_joints.position[2] < math.radians(25) and cdist > 0:
            self.kin.position[2] = self.package.kinematic_joints.position[2]+math.radians(5)
        elif self.package.kinematic_joints.position[2] > -math.radians(25) and cdist < 0:
            # print(cdist,self.package.kinematic_joints.position[2])
            self.kin.position[2] = self.package.kinematic_joints.position[2]+math.radians(5)*np.sign(cdist)
            # self.velocity.twist.angular.z = 0.6
        else:
            self.velocity.twist.angular.z = 0.8*np.sign(cdist)
            self.kin.position[2] = 0.0#self.package.kinematic_joints.position[2]-math.radians(1)*np.sign(cdist)


            # self.kin.position[2] = self.package.kinematic_joints.position[2]-math.radians(1)
            # self.velocity.twist.angular.z = -0.6
        # else:
        #     self.velocity.twist.angular.z = 0.0
        #     pred_dist = None
        
        if pred_dist is None:
            self.velocity.twist.linear.x = 0.0
        elif pred_dist > 1.1:
            self.velocity.twist.linear.x = 0.15
        elif pred_dist < 0.8:
            self.velocity.twist.linear.x = -0.15
            
        else: 
            self.velocity.twist.linear.x = 0.0
        

        self.pub_cmd_vel.publish(self.velocity)
        if abs(cdisty) < 20:
            pass
        elif cdisty > 0:
            if abs(self.package.kinematic_joints.position[3]) < math.radians(7):
                self.kin.position[3] = np.clip(self.package.kinematic_joints.position[3]+math.radians(0.3), math.radians(-15), math.radians(30))
            else:
                self.kin.position[1] = np.clip(self.package.kinematic_joints.position[1]+math.radians(0.3), math.radians(30), math.radians(50))
        else:
            if abs(self.package.kinematic_joints.position[3]) < math.radians(25):
                self.kin.position[3] = np.clip(self.package.kinematic_joints.position[3]-math.radians(0.3), math.radians(-15), math.radians(30))
            else:
                self.kin.position[1] = np.clip(self.package.kinematic_joints.position[1]-math.radians(0.3), math.radians(30), math.radians(50))
        # self.kin.position[3] = np.clip(self.kin.position[3],math.radians(-15),math.radians(15))                    
        self.pub_kin.publish(self.kin)
        # self.pub_cmd_vel2.publish(self.velocity2)
        
    def move_miro(self, *args):
        if self.pred_pos is None or len(self.other_path)==0:
            self.velocity2.twist.linear.x = 0.0
            self.velocity2.twist.angular.z = 0.0
            self.pub_cmd_vel2.publish(self.velocity2)
            print("Path finished")
            return
        target_pos = self.other_path[0]#np.array([1.0,0.0])
        self.velocity2.twist.linear.x = 0.1
        target = np.arctan2(target_pos[1]-self.pred_pos[1],target_pos[0]-self.pred_pos[0])
        
        # calculates the differance in angle between current and target
        dists = [(self.pred_angle%(2*np.pi)-target%(2*np.pi))%(2*np.pi),(target%(2*np.pi)-self.pred_angle%(2*np.pi))%(2*np.pi)]

        # checks if the robot is close to the next node in the path 
        if np.linalg.norm(target_pos-self.pred_pos) < 0.3:
            self.velocity2.twist.linear.x = 0.0
            self.velocity2.twist.angular.z = 0.0
            self.other_path = self.other_path[1:]
            print("goal found")
        
        # checks if the robot is looking in the right direction
        elif min(dists) < 0.5:
            self.velocity2.twist.angular.z = 0.0
            self.pub_cmd_vel2.publish(self.velocity2)
        # moves clockwise if the right angle is lower
        elif dists[0] >= dists[1]:
            self.velocity2.twist.angular.z = 0.5
            self.velocity2.twist.linear.x = 0.01
            # print("turning left")
        # moves counter clockwise otherwise
        else:
            self.velocity2.twist.angular.z = -0.5
            self.velocity2.twist.linear.x = 0.01

        self.pub_cmd_vel2.publish(self.velocity2)
        
        
    def loop(self):
        pass
        # target = self.cur_target*np.pi/180
        # dists = [(self.pos.theta%(2*np.pi)-target)%(2*np.pi),(target-self.pos.theta%(2*np.pi))%(2*np.pi)]

        # print(self.midpoints)
        # print(self.pos.theta, dists[0],dists[1], np.argmin(dists), target, self.velocity.twist.angular.z)
        # if min(dists) < 0.01:
        #     self.velocity.twist.angular.z = 0.0
        #     self.pub_cmd_vel.publish(self.velocity)
            
        #     # rospy.sleep(2)
        # elif dists[0] >= dists[1]:
        #     self.velocity.twist.angular.z = 0.6
        #     # print("turning left")
        # else:
        #     self.velocity.twist.angular.z = -0.6
        #     # print("turning right")
    
        # self.pub_cmd_vel.publish(self.velocity)


        
if __name__ == "__main__":
    try:
        main = LookMiro()
        if(rospy.get_node_uri()):
            pass
        else:
            rospy.init_node("look_miro", anonymous=True)
        # while not rospy.core.is_shutdown():
        #     main.loop()
        # main.interface.disconnect()
        rospy.spin()
    except Exception as e:
        print("exception",e)
        exit()


