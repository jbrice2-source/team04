#!/usr/bin/env python3
import os
import rospy
import numpy as np
import scipy.signal
from std_msgs.msg import UInt16MultiArray, Float32, Float32MultiArray, Int16MultiArray

def bandpass(data, edges, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def audio_callback(msg):    
    audio = np.asarray(msg.data)
    
    # if 400 - 600hz:
    if np.any((bandpass(audio, [400, 600], 20000.0))):
        rospy.loginfo("400-600 Hz frequency detected in audio input.")
        rospy.loginfo("Turning left")
        
    # if 700 - 900hz:
    if np.any((bandpass(audio, [700, 900], 20000.0))):
        rospy.loginfo("400-600 Hz frequency detected in audio input.")
        rospy.loginfo("Turning right")
        
    # if 1000 - 1200hz:
    if np.any((bandpass(audio, [1000, 1200], 20000.0))):
        rospy.loginfo("1000-1200 Hz frequency detected in audio input.")
        rospy.loginfo("Moving forward")
        
    # if 1300 - 1500hz:
    if np.any((bandpass(audio, [1300, 1500], 20000.0))):
        rospy.loginfo("1300-1500 Hz frequency detected in audio input.")
        rospy.loginfo("Moving backwards")
        
    # if 1600hz - 1800hz:
    if np.any((bandpass(audio, [1600, 1800], 20000.0))):
        rospy.loginfo("1600-1800 Hz frequency detected in audio input.")
        rospy.loginfo("Stopping")
    rospy.sleep(0.6)

def make_sound():
    # get robot name
    robot_name = rospy.get_param("~robot", os.getenv("MIRO_ROBOT_NAME", "rob01"))

    # publisher for tone control topic 
    # from miro-e documentation: The three elements are [frequency, volume, duration] of 
    # a tone that will be produced by the on-board speaker. Frequency is in Hertz 
    # (values between 50 and 2000 are accepted), volume is in 0 to 255, and duration is in platform ticks (20ms periods).
    topic = f"/{robot_name}/control/tone"
    pub = rospy.Publisher(topic, UInt16MultiArray, queue_size=10)
    rospy.loginfo(f"Publishing to tone topic: {topic}")
    rospy.sleep(1)  # give publisher time to connect

    # create and send tone command
    msg = UInt16MultiArray()
    # [frequency (Hz), volume (0–255), duration (ms)]
    msg.data = [900, 128, 1000]   
    rospy.loginfo("playing 900 Hz tone for 1 second.")
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('frequency_instructions')
    # get robot name 
    robot_name = rospy.get_param("~robot", os.getenv("MIRO_ROBOT_NAME", "rob01"))
    # subscribe to microphone audio data
    mic_topic = f"/{robot_name}/sensors/mics"
    rospy.Subscriber(mic_topic, Int16MultiArray, audio_callback, queue_size=1)
    rospy.loginfo("audio localisation node started")
    rospy.spin()
