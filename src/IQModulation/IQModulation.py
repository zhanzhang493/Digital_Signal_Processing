import numpy as np


class IQModule:
    name = 'IQ Modulation and De-Modulation Module'

    def __init__(self):
        pass

    @staticmethod
    def IQ_modulator(ipt_i, ipt_q, fc, fs):
        pass

    @staticmethod
    def IQ_demodulator(ipt, fc, fs):
        ts = 1 / fs
        num_point = len(ipt)
        t_axis = np.arange(num_point) * ts
        LO_phase = 2 * np.pi * fc * t_axis
        LO_I = np.cos(LO_phase)
        LO_Q = np.sin(LO_phase)
        opt_i = ipt * LO_I
        opt_q = ipt * (-LO_Q)
        return opt_i, opt_q
