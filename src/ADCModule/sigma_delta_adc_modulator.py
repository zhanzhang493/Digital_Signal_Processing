import numpy as np
import os
import sys

current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, '..', 'FilterDesign')
sys.path.append(src_path)  # add module path

import FilterModule


class ADCModule:
    name = 'ADC Module'

    def __init__(self):
        pass

    @staticmethod
    def decimation_filter(v, OSR):
        z = ADCModule.moving_average(v, OSR)
        return z[::OSR]

    @staticmethod
    def moving_average(v, OSR):
        num = np.ones(OSR)
        num[0] = 0.5
        num[OSR - 1] = 0.5
        den = [OSR]
        z = FilterModule.FilterModule.digital_filter(num, den, v)
        return z

    """ Sigma Delta Modulator"""
    @staticmethod
    def first_order(u, half_spread, num_bit):
        # x = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            # x[id] = u[id] - v[id-1]
            # y[id] = x[id] + y[id-1]
            y[id] = u[id] - v[id-1] + y[id-1]
            v[id] = ADCModule.quantizer_add_noise(half_spread, num_bit, y[id])
            # v[id] = (1.0 if y[id] > 0.0 else -1.0)

        return u, y, v    # u - input, y - quantizer input, v - quantizer output

    @staticmethod
    def second_order(u, half_spread, num_bit):
        # x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        # x2 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            # x0[id] = u[id] - v[id-1]
            # x1[id] = x0[id] + x1[id-1]
            x1[id] = u[id] - v[id-1] + x1[id-1]
            # x2[id] = x1[id] - v[id-1]
            # y[id] = x2[id] + y[id-1]
            y[id] = x1[id] - v[id-1] + y[id-1]
            v[id] = ADCModule.quantizer_add_noise(half_spread, num_bit, y[id])

        return u, x1, y, v

    @staticmethod
    def third_order(u, half_spread, num_bit):
        x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            x0[id] = u[id] - v[id - 1] + x0[id - 1]
            x1[id] = x0[id] - v[id - 1] + x1[id - 1]
            y[id] = x1[id] - v[id - 1] + y[id - 1]
            v[id] = ADCModule.quantizer_add_noise(half_spread, num_bit, y[id])

        return u, x0, x1, y, v

    """ Quantizer """
    @staticmethod
    def quantizer_mid_rise(half_spread, num_bit, y):
        num_step = 2 ** num_bit
        step = 2 * half_spread / num_step
        assert (y <= half_spread + step/2) and (y >= - (half_spread + step/2))

        e = abs((y/step) - int(y/step))
        if e > 0:
            if y >= 0:
                step_id = int(y/step) + 1
            else:
                step_id = int(y/step) - 1
        else:
            if y >= 0:
                step_id = int(y / step)
            else:
                step_id = int(y / step)

        if abs(step_id) > (num_step//2):
            step_id = (num_step//2) * (step_id / abs(step_id))

        v = step_id * step

        return v

    @staticmethod
    def quantizer_mid_tread(half_spread, num_bit, y):
        assert num_bit > 1
        num_step = 2 ** num_bit
        step = 2 * half_spread / num_step
        assert (y <= half_spread + step/2) and (y >= - (half_spread + step/2))

        e = abs((y/step) - int(y/step))
        if e > 0.5:
            if y >= 0:
                step_id = int(y / step) + 1
            else:
                step_id = int(y / step) - 1
        else:
            if y >= 0:
                step_id = int(y / step)
            else:
                step_id = int(y / step)

        if abs(step_id) > (num_step//2 - 1):
            step_id = (num_step//2-1) * (step_id / abs(step_id))

        v = step_id * step

        return v

    @staticmethod
    def quantizer_add_noise(half_spread, num_bit, y):
        num_step = 2 ** num_bit
        step = 2 * half_spread / num_step
        # assert (y <= half_spread + step / 2) and (y >= - (half_spread + step / 2))

        v = y + step * np.random.uniform(-1, 1, np.shape(y)) / 2

        return v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """ Quantizer """
    # half_spread = 1
    # num_bit = 2
    # num_step = 2 ** num_bit
    # step = 2 * half_spread / num_step
    # y = -1 * step
    # v1 = ADCModule.quantizer_mid_rise(half_spread, num_bit, y)
    # v2 = ADCModule.quantizer_mid_tread(half_spread, num_bit, y)
    # v3 = ADCModule.quantizer_add_noise(half_spread, num_bit, y)
    # print(step, y, v1, v2, v3)
    # print(step, y-v1, y-v2, y-v3)
    #
    # y = 0.25
    # v1 = ADCModule.quantizer_mid_rise(1, 2, y)
    # v2 = ADCModule.quantizer_mid_tread(1, 2, y)
    # print(v1, v2)

    """ Sigma Delta Modulator Test"""
    half_spread = 1
    num_bit = 1
    N = 2048
    freq = 30
    fn = 100
    OSR = 10
    num_OSR = N * OSR
    fs = fn * OSR

    t_osr = np.arange(num_OSR) / fs
    tn = t_osr[::OSR]

    amp = 0.8
    # input signal
    s_analog = amp * np.sin(2 * freq * np.pi * t_osr)

    """ DS Quantization """
    s_sample = s_analog[::OSR]
    s_sample_q = ADCModule.quantizer_add_noise(half_spread, num_bit, s_sample)

    u, y, v = ADCModule.first_order(s_analog, half_spread, num_bit)
    u2, x2, y2, v2 = ADCModule.second_order(s_analog, half_spread, num_bit)
    u3, x3_0, x3_1, y3, v3 = ADCModule.third_order(s_analog, half_spread, num_bit)

    V = 20 * np.log10(np.abs(np.fft.fft(v)))
    max_value = np.max(V)
    V = V - max_value

    V2 = 20 * np.log10(np.abs(np.fft.fft(v2)))
    max_value = np.max(V2)
    V2 = V2 - max_value

    V3 = 20 * np.log10(np.abs(np.fft.fft(v3)))
    max_value = np.max(V3)
    V3 = V3 - max_value

    fig_SD = plt.figure(figsize=(7, 4), dpi=100)
    ax_SD = fig_SD.add_subplot()
    ax_SD.semilogx(V[0:len(V) // 2 - 1], '--', label='SD')
    ax_SD.semilogx(V2[0:len(V2) // 2 - 1], label='SD2')
    ax_SD.semilogx(V3[0:len(V3) // 2 - 1], label='SD3')
    ax_SD.set_ylabel('dB', fontsize=10)
    plt.legend()
    
    """ Decimation Filter """
    v1_dec = ADCModule.decimation_filter(v, OSR)
    v2_dec = ADCModule.decimation_filter(v2, OSR)
    v3_dec = ADCModule.decimation_filter(v3, OSR)

    fig_time = plt.figure(figsize=(7, 4), dpi=100)
    ax_time = fig_time.add_subplot()
    ax_time.plot(tn, s_sample, label='Ideal')
    ax_time.plot(tn, s_sample_q, label='DS Quantization')
    ax_time.plot(tn, v1_dec, label='Oversamp-SD')
    ax_time.plot(tn, v2_dec, label='Oversamp-SD2')
    ax_time.plot(tn, v3_dec, label='Oversamp-SD3')
    ax_time.set_title('Time domain', fontsize=12)
    ax_time.set_xlabel('Time', fontsize=10)
    ax_time.set_ylabel('Volts', fontsize=10)
    plt.legend()

    S_sample = 20 * np.log10(np.abs(np.fft.fft(s_sample)))
    max_value = np.max(S_sample)
    S_sample = S_sample - max_value

    S_sample_q = 20 * np.log10(np.abs(np.fft.fft(s_sample_q)))
    max_value = np.max(S_sample_q)
    S_sample_q = S_sample_q - max_value

    v1_dec = 20 * np.log10(np.abs(np.fft.fft(v1_dec)))
    max_value = np.max(v1_dec)
    v1_dec = v1_dec - max_value

    v2_dec = 20 * np.log10(np.abs(np.fft.fft(v2_dec)))
    max_value = np.max(v2_dec)
    v2_dec = v2_dec - max_value

    v3_dec = 20 * np.log10(np.abs(np.fft.fft(v3_dec)))
    max_value = np.max(v3_dec)
    v3_dec = v3_dec - max_value

    fig_spec = plt.figure(figsize=(7, 4), dpi=100)
    ax_spec = fig_spec.add_subplot()
    ax_spec.plot(S_sample, label='Ideal')
    ax_spec.plot(S_sample_q, label='DS quantization')
    ax_spec.plot(v1_dec, label='Oversamp-SD')
    ax_spec.plot(v2_dec, label='Oversamp-SD2')
    ax_spec.plot(v3_dec, label='Oversamp-SD3')
    ax_spec.set_ylabel('dB', fontsize=10)
    plt.legend()

    plt.show()


