#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import UInt16MultiArray

def make_sound():
    rospy.init_node('make_sound')

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
    make_sound()
