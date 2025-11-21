#!/usr/bin/env python3
import os
import rospy
import numpy as np
import scipy.signal
from std_msgs.msg import Int16MultiArray, Float32, Float32MultiArray

def bandpass(data, edges, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def audio_callback(msg):    
    rospy.sleep(0.4)
    audio = np.asarray(msg.data)
    print(audio.shape)
    # separate left and right channels

    print(f"400-600 MAX {np.max(bandpass(audio, [400, 600], 20000.0))}")
    print(f"900-1100 MAX {np.max(bandpass(audio, [900, 1100], 20000.0))}")
    print(f"1400-1600 MAX {np.max(bandpass(audio, [1400, 1600], 20000.0))}")
    print(f"1900-2100 MAX {np.max(bandpass(audio, [1900, 2100], 20000.0))}")
    print(f"2400-2600 MAX {np.max(bandpass(audio, [3000, 3400], 20000.0))}")
    print(f"2900-3100 MAX {np.max(bandpass(audio, [3000, 3400], 20000.0))}")
    print(f"3400-3600 MAX {np.max(bandpass(audio, [3000, 3400], 20000.0))}")
    print(f"3900-4100 MAX {np.max(bandpass(audio, [3000, 3400], 20000.0))}")

    # if 1000 - 1400hz:
    if np.max(bandpass(audio, [400, 600], 20000.0)) > 700:
        rospy.loginfo("400-600 Hz frequency detected in audio input.")
        rospy.loginfo("Turning left")
        
    # if 1500 - 1900hz:
    if np.max(bandpass(audio, [900, 1100], 20000.0)) > 700:
        rospy.loginfo("900-1100 Hz frequency detected in audio input.")
        rospy.loginfo("Turning right")
        
    # if 2000 - 2400hz:
    if np.max(bandpass(audio, [1400, 1600], 20000.0)) > 700:
        rospy.loginfo("1400 - 1600 Hz frequency detected in audio input.")
        rospy.loginfo("Moving forward")
        
    # if 2500 - 2900hz: 
    if np.max(bandpass(audio, [1900, 2100], 20000.0)) > 700:
        rospy.loginfo("1900-2100 Hz frequency detected in audio input.")
        rospy.loginfo("Moving backwards")
        
    # if 3000hz - 3400hz:
    if np.max(bandpass(audio, [2400, 2600], 20000.0)) > 700:
        rospy.loginfo("2400-2600 Hz frequency detected in audio input.")
        rospy.loginfo("Stopping")

    # if 2000 - 2400hz:
    if np.max(bandpass(audio, [2900, 3100], 20000.0)) > 700:
        rospy.loginfo("2900-3100 Hz frequency detected in audio input.")
        rospy.loginfo("Moving forward")
        
    # if 2500 - 2900hz:     
    if np.max(bandpass(audio, [3400, 3600], 20000.0)) > 700:
        rospy.loginfo("3400-3600 Hz frequency detected in audio input.")
        rospy.loginfo("Moving backwards")
        
    # if 3000hz - 3400hz:
    if np.max(bandpass(audio, [3900, 4100], 20000.0)) > 700:
        rospy.loginfo("3900-4100 Hz frequency detected in audio input.")
        rospy.loginfo("Stopping")


    else: print("no audio found")
    

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
    rospy.loginfo("audio node started")
    rospy.spin()
