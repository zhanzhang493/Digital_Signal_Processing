import matplotlib.font_manager as fm
import os
import sys

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)

C = 3e8
GHz = 1e9
MHz = 1e6
us = 1e-6


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    #############################################################
    """ Simple """
    #############################################################
    fs = 100 * MHz
    ts = 1 / fs
    N = 2 ** 10
    fc = 75 * MHz
    t_axis = np.arange(0, N - 1) * ts

    s = np.exp(1j * 2 * np.pi * fc * t_axis)
    S_f = np.fft.fft(s)
    S_f_dB = 20 * np.log10(np.abs(S_f))

    s_r = np.real(s)
    S_r_f = np.fft.fft(s_r)
    S_r_f_dB = 20 * np.log10(np.abs(S_r_f))

    s_i = np.imag(s)
    S_i_f = np.fft.fft(s_i)
    S_i_f_dB = 20 * np.log10(np.abs(S_i_f))

    """ Figure """
    fig_spec = plt.figure(figsize=(5, 5), dpi=100)
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title('Spectrum', fontsize=10, fontproperties=font_times)
    ax_spec.set_xlabel('freq - point', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('magnitude - dB', fontsize=8, fontproperties=font_times)
    ax_spec.plot(S_f_dB, label='complex - 75 MHz (fs = 100 MHz)')
    ax_spec.plot(S_r_f_dB, label='real - 75 MHz (fs = 100 MHz)')
    ax_spec.plot(S_r_f_dB, label='imag - 75 MHz (fs = 100 MHz)')
    plt.legend()

    #############################################################
    """ Complicated """
    #############################################################
    fs = 120 * MHz
    ts = 1 / fs

    B = 10 * MHz
    fc = 20 * MHz
    PRI = 10.24 * us
    num_point = int(fs * PRI)
    t_axis = np.arange(num_point) * ts
    alpha = B / PRI

    f_axis = np.arange(num_point) / num_point * fs

    ideal_fmcw_phase = 2 * np.pi * (fc * t_axis + alpha * (t_axis ** 2) / 2)
    s = np.exp(1j * ideal_fmcw_phase)

    s_r = np.real(s)
    S_r_f = np.fft.fft(s_r)
    S_r_f_dB = 20 * np.log10(np.abs(S_r_f))

    """ Figure """
    fig_time = plt.figure(figsize=(5, 5), dpi=100)
    ax_time = fig_time.add_subplot()
    ax_time.set_title('FMCW', fontsize=10, fontproperties=font_times)
    ax_time.set_xlabel('time', fontsize=8, fontproperties=font_times)
    ax_time.set_ylabel('magnitude', fontsize=8, fontproperties=font_times)
    ax_time.plot(np.real(s), label='ideal fmcw')
    plt.legend()

    fig_spec = plt.figure(figsize=(5, 5), dpi=100)
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title('Spectrum', fontsize=10, fontproperties=font_times)
    ax_spec.set_xlabel('freq - MHz', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('magnitude - dB', fontsize=8, fontproperties=font_times)
    ax_spec.plot(f_axis/MHz, S_r_f_dB, label='real - 20-30 MHz (fs = 100 MHz)')
    plt.legend()

    LO_phase = 2 * np.pi * (fc * t_axis)
    LO_I = np.cos(LO_phase)
    LO_Q = np.sin(LO_phase)
    y_I = s_r * LO_I
    y_Q = s_r * -LO_Q

    Y_I_f = np.fft.fft(y_I)
    Y_I_f_dB = 20 * np.log10(np.abs(Y_I_f))

    Y_Q_f = np.fft.fft(y_Q)
    Y_Q_f_dB = 20 * np.log10(np.abs(Y_Q_f))

    y = y_I + 1j * y_Q
    Y_f = np.fft.fft(y)
    Y_f_dB = 20 * np.log10(np.abs(Y_f))

    fig_spec = plt.figure(figsize=(7, 5), dpi=100)
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title('Spectrum', fontsize=10, fontproperties=font_times)
    ax_spec.set_xlabel('freq - MHz', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('magnitude - dB', fontsize=8, fontproperties=font_times)
    ax_spec.plot(f_axis / MHz, Y_I_f_dB, label='I (fs = 100 MHz)')
    ax_spec.plot(f_axis / MHz, Y_Q_f_dB, label='Q (fs = 100 MHz)')
    ax_spec.plot(f_axis / MHz, Y_f_dB, label='IQ Demoulation (fs = 100 MHz)')
    plt.legend()

    plt.show()
