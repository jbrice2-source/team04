#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#   BioRescue – Tone-based DetectAudioEngine
#
#   Detects three structured tone patterns:
#       (1) Distress call    – sweep + triple pulses
#       (2) Confirmation call – chirp + double pulses
#       (3) Rumble beacon    – sustained low tone
#
#   Uses:
#       - FFT band-energy detection
#       - Timing windows
#       - Debounce logic
#       - Light smoothing for stability
#

import numpy as np
import rospy


class DetectAudioEngine(object):

    #   Initialise
    def __init__(self, sample_rate=20000.0, block_size=500):

        self.fs = sample_rate
        self.n = block_size

        # Frequency bands for pattern isolation
        self.band_low  = (80.0, 200.0)         # rumble beacon
        self.band_mid  = (300.0, 1000.0)       # sweeps/chirps
        self.band_high = (1200.0, 2000.0)      # pulses

        # Thresholds (to be tuned on real MiRo)
        self.thresh_distress_mid = 0.02
        self.thresh_distress_high = 0.02

        self.thresh_confirm_mid = 0.02
        self.thresh_confirm_high = 0.02

        self.thresh_rumble_low = 0.015

        # Timing windows
        self.max_distress_window = 0.6
        self.max_confirm_window  = 0.6
        self.min_rumble_duration = 0.5

        # Smoothing (optional but ideal for MiRo’s noisy mics)
        self.alpha = 0.65
        self.e_low_prev = 0.0
        self.e_mid_prev = 0.0
        self.e_high_prev = 0.0

        # Internal pattern state
        self.last_mid_burst_distress = None
        self.high_pulse_times_distress = []

        self.last_mid_burst_confirm = None
        self.high_pulse_times_confirm = []

        self.rumble_start_time = None

        # Debounce timers
        self.last_trigger_distress = -999
        self.last_trigger_confirm  = -999
        self.last_trigger_rumble   = -999

        self.debounce_distress = 1.0     # sec
        self.debounce_confirm  = 1.0
        self.debounce_rumble   = 1.0

        # Output flags (edge-triggered each block)
        self._distress_flag = False
        self._confirm_flag  = False
        self._rumble_flag   = False

        # Precompute FFT frequency bins
        self.freq_bins = np.fft.rfftfreq(self.n, d=1.0/self.fs)

    #   Public API
    def process_block(self, block):
        """
        block: ndarray shape (4, n)
        """
        # Reset flags for this block
        self._distress_flag = False
        self._confirm_flag = False
        self._rumble_flag = False

        # Validate audio block
        if block.shape[0] != 4 or block.shape[1] != self.n:
            return

        # Use L/R/centre channels
        left   = block[0, :].astype(np.float32)
        right  = block[1, :].astype(np.float32)
        centre = block[2, :].astype(np.float32)

        # Normalise 16-bit PCM → [-1, 1]
        scale = 1.0 / 32768.0
        mono = (left + right + centre) * (scale / 3.0)

        # FFT
        spec = np.abs(np.fft.rfft(mono))

        # Band energies
        e_low  = self.band_energy(spec, self.band_low)
        e_mid  = self.band_energy(spec, self.band_mid)
        e_high = self.band_energy(spec, self.band_high)

        # Light smoothing
        e_low  = self.alpha * self.e_low_prev  + (1-self.alpha) * e_low
        e_mid  = self.alpha * self.e_mid_prev  + (1-self.alpha) * e_mid
        e_high = self.alpha * self.e_high_prev + (1-self.alpha) * e_high

        self.e_low_prev  = e_low
        self.e_mid_prev  = e_mid
        self.e_high_prev = e_high

        # Time
        t = rospy.get_time()

        # Update pattern detectors
        self.update_distress_detector(e_mid, e_high, t)
        self.update_confirm_detector(e_mid, e_high, t)
        self.update_rumble_detector(e_low, t)

    def detected_distress(self):
        return self._distress_flag

    def detected_confirmation(self):
        return self._confirm_flag

    def detected_rumble(self):
        return self._rumble_flag

    #   Band energy helper
    def band_energy(self, spec, band):
        lo, hi = band
        idx = np.where((self.freq_bins >= lo) & (self.freq_bins <= hi))[0]
        if idx.size == 0:
            return 0.0
        return float(np.mean(spec[idx]))

    #   Distress detector (sweep + 3 pulses)
    def update_distress_detector(self, e_mid, e_high, t):

        # Mid-band → indicates sweep start
        if e_mid > self.thresh_distress_mid:
            self.last_mid_burst_distress = t
            self.high_pulse_times_distress = []

        # High-band → pulses
        if e_high > self.thresh_distress_high:
            if self.last_mid_burst_distress is not None:
                if t - self.last_mid_burst_distress < self.max_distress_window:
                    self.high_pulse_times_distress.append(t)

        # Evaluate pattern
        if self.last_mid_burst_distress is not None:
            if (t - self.last_mid_burst_distress) >= 0.15:

                if len(self.high_pulse_times_distress) >= 3:
                    if t - self.last_trigger_distress > self.debounce_distress:
                        self._distress_flag = True
                        self.last_trigger_distress = t

                # Reset
                self.last_mid_burst_distress = None
                self.high_pulse_times_distress = []

    #   Confirmation detector (chirp + 2 pulses)
    def update_confirm_detector(self, e_mid, e_high, t):

        if e_mid > self.thresh_confirm_mid:
            self.last_mid_burst_confirm = t
            self.high_pulse_times_confirm = []

        if e_high > self.thresh_confirm_high:
            if self.last_mid_burst_confirm is not None:
                if t - self.last.last_mid_burst_confirm < self.max_confirm_window:
                    self.high_pulse_times_confirm.append(t)

        if self.last_mid_burst_confirm is not None:
            if (t - self.last_mid_burst_confirm) >= 0.15:

                if len(self.high_pulse_times_confirm) >= 2:
                    if t - self.last_trigger_confirm > self.debounce_confirm:
                        self._confirm_flag = True
                        self.last_trigger_confirm = t

                self.last_mid_burst_confirm = None
                self.high_pulse_times_confirm = []

    #   Rumble detector (low band > thresh for duration)
    def update_rumble_detector(self, e_low, t):

        if e_low > self.thresh_rumble_low:
            if self.rumble_start_time is None:
                self.rumble_start_time = t
            else:
                if (t - self.rumble_start_time) >= self.min_rumble_duration:
                    if t - self.last_trigger_rumble > self.debounce_rumble:
                        self._rumble_flag = True
                        self.last_trigger_rumble = t
        else:
            self.rumble_start_time = None
