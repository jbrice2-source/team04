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
        
        self.image_converter = CvBridge()
        self.camera = [None, None]
        self.pos = Pose2D()
        self.cur_target = 0
        self.timer1 = time.time_ns()
        self.timer2 = time.time_ns()

        self.midpoints = [None,None]
        self.kin = JointState()

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
        self.timer2 = rospy.Timer(rospy.Duration(0.1), self.look_miro)


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
            # orb = cv2.ORB_create()
            # kp , des= orb.detectAndCompute(image,None)
            # image = cv2.drawKeypoints(image,kp,None,color=(0,0,255),flags=0)

            # store image for display
            self.camera[index] = image
            # if time.time_ns()-self.timer1 > 1e8 and index == 0:
            #     self.timer1 = time.time_ns()
            #     self.match_image(self.camera[index],index)
            # elif time.time_ns()-self.timer2 > 1e8 and index == 1:
            #     self.timer2 = time.time_ns()
            #     self.match_image(self.camera[index],index)
        except CvBridgeError as e:
            # swallow error, silently
            #print(e)
            pass

    def match_image(self, *args):
        if type(None) in map(type,self.camera):
            return
        for index, img in enumerate(self.camera):
            # img2 = cv2.imread('pictures/135_1.png', cv2.IMREAD_GRAYSCALE) # trainImage
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            best = ([],None,None,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            try:
                for i in image_data[index::2]:
                    # find the keypoints and descriptors with SIFT
                    
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                    search_params = dict(checks = 50)
                    
                    flann = cv2.FlannBasedMatcher(index_params,search_params)
                    matches = flann.knnMatch(i[2],des2,k=2)
                    
                    # store all the good matches as per Lowe's ratio test.
                    good = []
                    for m,n in matches:
                        if m.distance < 0.75*n.distance:
                            good.append(m)
                    if len(good) > len(best[0]):
                        best = (good,i[0],i[1],kp2)
                if len(best[0])>MIN_MATCH_COUNT:
                    src_pts = np.float32([ best[2][m.queryIdx].pt for m in best[0] ]).reshape(-1,1,2)
                    dst_pts = np.float32([ best[3][m.trainIdx].pt for m in best[0] ]).reshape(-1,1,2)
                
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    matchesMask = mask.ravel().tolist()
                
                    h,w = best[1].shape
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts,M)
                    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                    # img2 = cv2.polylines(img2,[np.int32([[[100,100]],[[100,200]],[[200,100]],[[200,100]]])],True,255,3, cv2.LINE_AA)
                    centroid = (np.sum(dst, axis=0)/dst.shape[0]).astype(int)
                    self.midpoints[index] = centroid
                    img2 = cv2.circle(img2, (round(centroid[0,0]),round(centroid[0,1])), 10, (128,0,0), -1)
                else:
                    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                    matchesMask = None
                    self.midpoints[index] = None
                    
                draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

                img3 = cv2.drawMatches(best[1],best[2],img2,best[3],best[0],None,**draw_params)
                
                if index == 0:
                    self.display1.set_data(img3)#
                else:
                    self.display2.set_data(img3)#
            except:
                print("unexpected error occured")
        plt.draw()
        
    def look_miro(self, *args):
        if type(self.camera[0]) == type(None):
            return
        h,w,_ = self.camera[0].shape
        cdist = 0
        cdisty = 0
        if type(self.midpoints[0]) != type(None) and type(self.midpoints[1]) != type(None):
            cdist = ((3*w/4 - self.midpoints[0][0,0])+(w/4 - self.midpoints[1][0,0]))/2
            cdisty = ((h - self.midpoints[0][0,1]- self.midpoints[1][0,1]))/2
        elif type(self.midpoints[0]) != type(None):
            cdist = 3*w/4 - self.midpoints[0][0,0]
            cdisty = h/2 - self.midpoints[0][0,1]
        elif type(self.midpoints[1]) != type(None):
            cdist = w/4 - self.midpoints[1][0,0]
            cdisty = h/2 - self.midpoints[1][0,1]
        if abs(cdist) < 50:
            self.velocity.twist.angular.z = 0.0
        elif cdist > 0:
            self.velocity.twist.angular.z = 0.6
        else:
            self.velocity.twist.angular.z = -0.6
        self.pub_cmd_vel.publish(self.velocity)
        if abs(cdisty) < 50:
            pass
        elif cdisty > 0:
            self.kin.position[3] -= math.radians(1)
        else:
            self.kin.position[3] += math.radians(1)
        self.pub_kin.publish(self.kin)
        
        
        
    def loop(self):
        # target = self.cur_target*np.pi/180
        # dists = [(self.pos.theta%(2*np.pi)-target)%(2*np.pi),(target-self.pos.theta%(2*np.pi))%(2*np.pi)]

        # print(self.midpoints)
        # print(self.pos.theta, dists[0],dists[1], np.argmin(dists), target, self.velocity.twist.angular.z)
        if min(dists) < 0.01:
            self.velocity.twist.angular.z = 0.0
            self.pub_cmd_vel.publish(self.velocity)
            
            # rospy.sleep(2)
        elif dists[0] >= dists[1]:
            self.velocity.twist.angular.z = 0.6
            # print("turning left")
        else:
            self.velocity.twist.angular.z = -0.6
            # print("turning right")
    
        self.pub_cmd_vel.publish(self.velocity)


        
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


