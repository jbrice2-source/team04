#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   BioRescue – Spectrogram Viewer
#
#   Displays a scrolling spectrogram of the last 10 seconds of audio.
#   Audio blocks are supplied externally (from Phase-1 node) to avoid
#   multiple subscriptions to /miro/sensors/mics.
#
#   This file contains no ROS dependencies. It is a utility module
#   providing real-time visualisation via matplotlib.
#

import numpy as np
import matplotlib.pyplot as plt
import threading
import time


class SpectrogramEngine(object):

    #   Initialise
    def __init__(self, sample_rate=20000.0, block_size=500, history_sec=10.0):

        self.fs = sample_rate
        self.n = block_size

        # Number of blocks to store for a 10-second window
        self.max_blocks = int((history_sec * self.fs) // self.n)

        # Rolling buffer
        self.buffer = []

        # Shutdown flag
        self.shutdown_flag = False

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("MiRo Audio – Last 10s Spectrogram")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

        # Thread for refreshing plot
        thread = threading.Thread(target=self.update_loop)
        thread.daemon = True
        thread.start()

    #   Add a new audio block
    def add_block(self, block):
        """
        Add one microphone block for spectrogram display.
        block: ndarray shape (4, n)
        """

        # Combine L/R/centre channels
        mono = (block[0] + block[1] + block[2]) / 3.0

        # Normalise from 16-bit
        mono = mono.astype(np.float32) / 32768.0

        # Add to buffer
        self.buffer.append(mono)

        # Trim to history size
        if len(self.buffer) > self.max_blocks:
            self.buffer = self.buffer[-self.max_blocks:]

    #   Update loop (runs in separate thread)
    def update_loop(self):
        """
        Periodically refresh the spectrogram (~5 Hz).
        """
        while not self.shutdown_flag:
            self.update_plot()
            time.sleep(0.2)

    #   Update plot
    def update_plot(self):
        if len(self.buffer) < 3:
            return

        audio = np.concatenate(self.buffer)

        self.ax.clear()
        self.ax.set_title("MiRo Audio – Last 10s Spectrogram")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")

        # Spectrogram
        self.ax.specgram(
            audio,
            NFFT=256,
            Fs=self.fs,
            noverlap=128,
            cmap="viridis",
        )

        plt.pause(0.001)

    #   Shutdown cleanly
    def shutdown(self):
        """
        Shut down spectrogram thread and close window.
        """
        self.shutdown_flag = True
        plt.close(self.fig)
