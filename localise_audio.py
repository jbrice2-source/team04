
#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Int16MultiArray, Float32

# GCC phat algorithm to compute difference in arrival time of sound for localisation
# takes left and right audio signals as input
def gcc_phat(sig, refsig, fs=20000, max_tau=None, interp=1):
    # sig (left signal), refsig (right signal)
    # fs: sampling frequency
    # tau: estimated time delay between signals
    # interp: interpolation factor
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    R /= np.abs(R) + 1e-9
    cc = np.fft.irfft(R, n=(interp * n))
    max_shift = int(interp * n / 2)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau

def audio_callback(msg):
    # miros audio - to np array
    data = np.array(msg.data, dtype=np.int16)
    # separate left and right channels
    left  = data[0::2].astype(np.float32)
    right = data[1::2].astype(np.float32)

    fs = 20000 # sampling frequency
    mic_distance = 0.14 # distance between microphones
    speed_sound = 343.0 # speed of sound

    # tau - estimated time delay
    tau = gcc_phat(left, right, fs)
    # calculate direction (angle) of sound source
    angle = np.arcsin(np.clip(speed_sound * tau / mic_distance, -1.0, 1.0))
    angle_deg = np.degrees(angle)
    # log for debugging / confirmation
    rospy.loginfo(f"Sound angle: {angle_deg:.2f}°")
    pub_angle.publish(angle_deg)

if __name__ == '__main__':
    rospy.init_node('localise_audio')
    pub_angle = rospy.Publisher('/sound_direction', Float32, queue_size=10)
    rospy.Subscriber('/miro/sensors/mics', Int16MultiArray, audio_callback)
    rospy.loginfo("GCC-PHAT localisation node started.")
    rospy.spin()
