#!/usr/bin/env python3
import os
import rospy
import numpy as np
from std_msgs.msg import Int16MultiArray, Float32, Float32MultiArray
import miro2
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

last_update_time = 0.0


# notes
# test with other miro, make other miro make combination of sounds, low>med>high boop boop boop

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
past_angles = np.zeros(30,dtype=float)

# data -> input audio data, edges -> freq edges e.g. [50, 100](hz)
def bandpass(data: np.ndarray, edges, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

cur_angle = 0.0
def audio_callback(msg):
    global past_angles, cos_joints, pub_cos, cur_angle
    # miros audio - to np array
    # data = np.array(msg.data, dtype=np.int16)
    # miro uses 4 microphones so reshape to n_frames and 4 channels
    # frames = data.reshape(-1, 4)
    
    data = np.asarray(msg.data)
    data_bandpassed = bandpass(data, [900, 1000], 20000.0)
    
    global last_update_time

    # --- RAW AND FILTERED SIGNALS ---
    raw = data
    filtered = data_bandpassed
    fs = 20000.0

    # --- UPDATE PLOT ONCE PER SECOND ---
    current_time = time.time()
    if current_time - last_update_time > 1.0:

        # FFT
        N = len(raw)
        freqs = np.fft.rfftfreq(N, 1/fs)
        raw_fft = np.abs(np.fft.rfft(raw))
        filtered_fft = np.abs(np.fft.rfft(filtered))

        # --- MAKE NEW PLOT ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, raw_fft, label="Raw FFT", alpha=0.6)
        ax.plot(freqs, filtered_fft, label="Filtered FFT (800–1000 Hz)", alpha=0.8)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Frequency Spectrum (Raw vs Bandpassed)")
        ax.legend()
        ax.grid(True)

        # Save to file (no GUI)
        fig.savefig("/tmp/audio_fft.png")
        plt.close(fig)

        last_update_time = current_time

    # data_bandpassed = np.transpose(data_bandpassed.reshape((4, 500)))
    frames = data_bandpassed.reshape(4,500)
    print(frames.shape)
    # separate left and right channels
    left = frames[:,0]  # left mic
    right = frames[:, 1] # right mic

    mic_distance = 0.11 # distance between microphones
    speed_sound = 343.0 # speed of sound

    # tau - estimated time delay using gcc function
    tau = gcc_phat(left, right, fs)
    # calculate direction (as an angle) of sound source
    angle = np.arcsin(np.clip(speed_sound * tau / mic_distance, -1.0, 1.0)) 
    angle_deg = np.degrees(angle)
    cur_angle = angle_deg
    if angle_deg != 0:
        past_angles = np.append(np.array([angle_deg]),past_angles[1:])
    mean = np.mean(past_angles[past_angles != 0])
    if np.isnan(mean):
        mean = 0
    else:
        cos_joints.data[left_ear] = np.round(np.clip(np.radians(mean+90)/(np.pi),0.01,0.99),1)
        cos_joints.data[right_ear] = np.round(np.clip(1-np.radians(mean+90)/(np.pi),0.01,0.99),1)
        # pub_cos.publish(cos_joints)
        print(np.clip(np.radians(mean+90)/(np.pi),0.01,0.99))
    # log for debugging / confirmation and send to publisher
    rospy.loginfo(f"Sound angle: {angle_deg:.2f}° {mean:.2f}")
    # pub_angle.publish(Float32(data=angle_deg))

# def move_ear(*args):
#     pub_cos.publish(cos_joints)

droop, wag, left_eye, right_eye, left_ear, right_ear = range(6)
cos_joints = Float32MultiArray()
cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

if __name__ == '__main__':
    rospy.init_node('localise_audio')
     # get robot name 
    robot_name = rospy.get_param("~robot", os.getenv("MIRO_ROBOT_NAME", "rob01"))
    # publish the angle of the sound source to sound_direction topic
    pub_angle = rospy.Publisher('/sound_direction', Float32, queue_size=5)
    # pub_cos = rospy.Publisher(f"/miro/control/cosmetic_joints", Float32MultiArray, queue_size=5)
    # timer = rospy.Timer(rospy.Duration(0.5),move_ear)
    # pub_cos.publish(cos_joints)
    print(robot_name)
    # subscribe to microphone audio data
    mic_topic = f"/{robot_name}/sensors/mics"
    print("subscribe to mic topic")
    rospy.Subscriber(mic_topic, Int16MultiArray, audio_callback, queue_size=1)
    rospy.loginfo("audio localisation node started")
    rospy.spin()
