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
        self.pose2 = rospy.Subscriber(base2 + "/sensors/body_pose",
            Pose2D, self.callback_pose2, queue_size=1, tcp_nodelay=True)
        # self.sub_mics = rospy.Subscriber(basename + "/sensors/mics",
        #             Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        self.sub_caml = rospy.Subscriber(base2 + "/sensors/caml/compressed",
                    CompressedImage, self.callback_caml, queue_size=1, tcp_nodelay=True)
        self.sub_camr = rospy.Subscriber(base2 + "/sensors/camr/compressed",
                CompressedImage, self.callback_camr, queue_size=1, tcp_nodelay=True)

        self.kin.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin.position = [0.0, math.radians(50.0), 0.0, math.radians(-10)]
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

    def filter_Detections(results, thresh = 0.4):
        A = []
        for detection in results:
            class_id = detection[4:].argmax()
            confidence_score = np.sum(detection[4:])
            # if confidence_score > thresh:
                # print(detection[4:])
            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)
        # print(considerable_detections[:,-1])
        return considerable_detections

    def match_image(self, *args):
        if type(None) in map(type,self.camera):
            return
        for index, img in enumerate(self.camera):
            mode_path = "best_new.onnx"
            onnx_model = ort.InferenceSession(mode_path)

            classes = ["0","45","90","135","180","225","270","315"]
            image = img.copy()
            
            img_height, img_width = image.shape[:2]

            # img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img2, pad = LookMiro.letterbox(image, (640, 640))
            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img2) / 255.0
            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
            outputs = onnx_model.run(None, {"images": image_data})
            results = outputs[0]
            results = results.transpose()
            # print(results.shape)
            
            
            results = LookMiro.filter_Detections(results)
            print(results.shape)
            
            pos1_vec = np.array([self.pos.x,self.pos.y])
            pos2_vec = np.array([self.pos2.x,self.pos2.y])
            real_dist = np.linalg.norm(pos1_vec-pos2_vec)

            if len(results) != 0:
                rescaled_results, confidences = LookMiro.rescale_back(results, img_width, img_height)

                for res, conf in zip(rescaled_results, confidences):
                    x1,y1,x2,y2, cls_id = res
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = "{:.2f}".format(conf)
                    # draw the bounding boxes
                    cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),2)
                    cv2.putText(image,classes[cls_id]+' '+conf,(x1,y1-17),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),3)
                    pred_dist = np.sqrt(70/(y2-y1))
                    print("class", index,classes[cls_id], conf, y2-y1, pred_dist, pred_dist-real_dist)
                    self.pred_dist[index] = pred_dist
                    self.midpoints[index] = np.array([x2+x1,y2+y1])/2
            else:
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
        print(self.midpoints)
        if type(self.midpoints[0]) != type(None) and type(self.midpoints[1]) != type(None):
            cdist = ((3*w/4 - self.midpoints[0][0])+(w/4 - self.midpoints[1][0]))/2
            cdisty = ((h - self.midpoints[0][1]- self.midpoints[1][1]))/2
        elif type(self.midpoints[0]) != type(None):
            cdist = 3*w/4 - self.midpoints[0][0]
            cdisty = h/2 - self.midpoints[0][1]
        elif type(self.midpoints[1]) != type(None):
            cdist = w/4 - self.midpoints[1][0]
            cdisty = h/2 - self.midpoints[1][1]
        print(cdist,cdisty)
        
        pred_dist = None
        if self.pred_dist[0] is not None and self.pred_dist[1] is not None:
            pred_dist = np.mean(self.pred_dist)
        elif self.pred_dist[0] is not None:
            pred_dist = self.pred_dist[0]
        elif self.pred_dist[1] is not None:
            pred_dist = 0
        if pred_dist is None:
            self.velocity.twist.linear.x = 0.0
        elif pred_dist > 1.1:
            self.velocity.twist.linear.x = 0.1
        elif pred_dist < 1.0:
            self.velocity.twist.linear.x = -0.1
        else: 
            self.velocity.twist.linear.x = 0.0
        
        if abs(cdist) < 50:
            self.velocity.twist.angular.z = 0.0
        elif cdist > 0:
            self.velocity.twist.angular.z = 0.3
        elif cdist < 0:
            self.velocity.twist.angular.z = -0.3
        else:
            self.velocity.twist.angular.z = 0.0

            
        self.pub_cmd_vel.publish(self.velocity)
        if abs(cdisty) < 50:
            pass
        elif cdisty > 0:
            self.kin.position[3] -= math.radians(1)
        else:
            self.kin.position[3] += math.radians(1)
                    
        self.pub_kin.publish(self.kin)
        
        
        
    def NMS(boxes, conf_scores, iou_thresh = 0.55):

        #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        areas = (x2-x1)*(y2-y1)

        order = conf_scores.argsort()

        keep = []
        keep_confidences = []

        while len(order) > 0:
            idx = order[-1]
            A = boxes[idx]
            conf = conf_scores[idx]

            order = order[:-1]

            xx1 = np.take(x1, indices= order)
            yy1 = np.take(y1, indices= order)
            xx2 = np.take(x2, indices= order)
            yy2 = np.take(y2, indices= order)

            keep.append(A)
            keep_confidences.append(conf)

            # iou = inter/union

            xx1 = np.maximum(x1[idx], xx1)
            yy1 = np.maximum(y1[idx], yy1)
            xx2 = np.minimum(x2[idx], xx2)
            yy2 = np.minimum(y2[idx], yy2)

            w = np.maximum(xx2-xx1, 0)
            h = np.maximum(yy2-yy1, 0)

            intersection = w*h

            # union = areaA + other_areas - intesection
            other_areas = np.take(areas, indices= order)
            union = areas[idx] + other_areas - intersection

            iou = intersection/union

            boleans = iou < iou_thresh

            order = order[boleans]

            # order = [2,0,1]  boleans = [True, False, True]
            # order = [2,1]

        return keep, keep_confidences
    
    def rescale_back(results,img_w,img_h):
        cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
        cx = cx/640.0 * img_w
        cy = cy/640.0 * img_h
        w = w/640.0 * img_w
        h = h/640.0 * img_h
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        boxes = np.column_stack((x1, y1, x2, y2, class_id))
        keep, keep_confidences = LookMiro.NMS(boxes,confidence)
        print(np.array(keep).shape)
        return keep, keep_confidences
        
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


