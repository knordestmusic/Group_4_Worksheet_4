import numpy as np
import soundfile as sf


class SchroederReverb:
    def __init__(self, fs, comb_delays, comb_gains, ap_delays, ap_gain):
        self.fs = fs
        self.comb_delays = comb_delays
        self.comb_gains = comb_gains
        self.ap_delays = ap_delays
        self.ap_gain = ap_gain

    def comb_filter(self, x, delay, g):
        ### Feedback comb: y[n] = x[n] + g * y[n - delay] ###
        y = np.zeros_like(x)
        for n in range(len(x)):
            y[n] = x[n]
            if n >= delay:
                y[n] += g * y[n - delay]
        return y

    def allpass_filter(self, x, delay, g):
        y = np.zeros_like(x)
        for n in range(len(x)):
            x_d = x[n - delay] if n >= delay else 0
            y_d = y[n - delay] if n >= delay else 0
            y[n] = -g * x[n] + x_d + g * y_d
        return y

    def process(self, x):
        # 1) Series all-pass
        ap_out = x
        for d in self.ap_delays:
            ap_out = self.allpass_filter(ap_out, d, self.ap_gain)
        # 2) parallel comb bank
        comb_sum = sum(
            self.comb_filter(ap_out, d, g)
            for d, g in zip(self.comb_delays, self.comb_gains)
        )
        return comb_sum
