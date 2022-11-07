import matplotlib.font_manager as fm
import numpy as np
import os
import sys

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'IQModulation'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'PhaseModule'))
import IQModulation
import FilterModule
import PhaseModule


C = 3e8
GHz = 1e9
MHz = 1e6
us = 1e-6


def hamming(num_order):
    n = np.arange(num_order)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_order - 1))
    return win


def plot_spectrum(freq_axis, spectrum, s_label, title):
    fig_spec = plt.figure(figsize=(7, 5), dpi=100)
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title(title, fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=10, fontproperties=font_times)
    ax_spec.plot(freq_axis / MHz, spectrum, label=s_label)
    plt.tick_params(labelsize=10)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.grid()
    plt.legend(fontsize=8, loc='lower right')

    return fig_spec


def plot_spectrum_sub(ax_sub, freq_sub, spectrum_sub, title):
    ax_sub.set_title(title, fontsize=14, fontproperties=font_times)
    ax_sub.set_xlabel('Freq - MHz', fontsize=12, fontproperties=font_times)
    ax_sub.set_ylabel('Magnitude - dB', fontsize=12, fontproperties=font_times)
    ax_sub.plot(freq_sub / MHz, spectrum_sub)
    plt.tick_params(labelsize=12)
    labels = ax_sub.get_xticklabels() + ax_sub.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.ylim(0, 51)
    plt.grid()
    # plt.legend(fontsize=8, loc='lower right')


def plot_spectrum_mult(freq_axis, spectrum_np, save_name):
    fig_spec = plt.figure(figsize=(45, 12), dpi=50)
    num_col = 4
    num_row = 2
    title_list = ['Original spectrum', 'In-phase component', 'In-phase component after LPF', 'IQ output',
                  'Filter response', 'Quadrature-phase component', 'Quadrature-phase component after LPF',
                  'IQ output after LPF']
    for k in range(8):
        spectrum = spectrum_np[k]
        ax_spec = fig_spec.add_subplot(num_row, num_col, k+1)
        plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
        plot_spectrum_sub(ax_spec, freq_axis, spectrum, title_list[k])
        if k == 4:
            plt.ylim(-60, 0)
        else:
            plt.ylim(0, 51)

    plt.savefig(save_name+'.pdf')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """======================================================================="""
    """ input signal """
    fs = 200 * MHz
    ts = 1 / fs

    B = 10 * MHz
    fc = 30 * MHz
    LO_fc = 35 * MHz
    PRI = 10.24 * us
    num_point = int(fs * PRI)
    t_axis = np.arange(num_point) * ts
    alpha = B / PRI
    amp = 2
    beta = amp / PRI
    save_name = 'IQ_center_freq'

    f_axis = np.arange(num_point) / num_point * fs
    f_axis_shift = np.arange(-int(num_point/2), int(num_point/2)) / num_point * fs

    """======================================================================="""
    """ input signal """
    ideal_fmcw_phase = 2 * np.pi * (fc * t_axis + alpha * (t_axis ** 2) / 2)
    s = (1 + beta * t_axis) * np.exp(1j * ideal_fmcw_phase)

    s_r = np.real(s)
    S_r_f = np.fft.fft(s_r)
    S_r_f_dB = 20 * np.log10(np.abs(S_r_f))

    """======================================================================="""
    """ IQ Demodulation  """
    y_I, y_Q = IQModulation.IQModule.IQ_demodulator(s_r, LO_fc, fs)

    Y_I_f = np.fft.fft(y_I)
    Y_I_f_dB = 20 * np.log10(np.abs(Y_I_f))

    Y_Q_f = np.fft.fft(y_Q)
    Y_Q_f_dB = 20 * np.log10(np.abs(Y_Q_f))

    y = y_I + 1j * y_Q
    Y_f = np.fft.fft(y)
    Y_f_dB = 20 * np.log10(np.abs(Y_f))

    """======================================================================="""
    """ Filtering """
    win = hamming(16)
    b = win
    a = [16]

    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, num_point)))
    freq_response_dB = 20 * np.log10(np.abs(freq_response))
    freq_response_dB = freq_response_dB - np.amax(freq_response_dB)

    y_I_filter = FilterModule.FilterModule.digital_filter(b, a, y_I)
    Y_I_filter = np.fft.fft(y_I_filter)
    Y_I_filter_dB = 20 * np.log10(np.abs(Y_I_filter))

    y_Q_filter = FilterModule.FilterModule.digital_filter(b, a, y_Q)
    Y_Q_filter = np.fft.fft(y_Q_filter)
    Y_Q_filter_dB = 20 * np.log10(np.abs(Y_Q_filter))

    y_filter = y_I_filter + 1j * y_Q_filter
    Y_filter = np.fft.fft(y_filter)
    Y_filter_dB = 20 * np.log10(np.abs(Y_filter))

    """======================================================================="""
    """ Figure """
    specturm_numpy = np.zeros((8, num_point), dtype=float)
    specturm_numpy[0] = np.fft.fftshift(S_r_f_dB)
    specturm_numpy[1] = np.fft.fftshift(Y_I_f_dB)
    specturm_numpy[2] = np.fft.fftshift(Y_I_filter_dB)
    specturm_numpy[3] = np.fft.fftshift(Y_f_dB)
    specturm_numpy[4] = freq_response_dB
    specturm_numpy[5] = np.fft.fftshift(Y_Q_f_dB)
    specturm_numpy[6] = np.fft.fftshift(Y_Q_filter_dB)
    specturm_numpy[7] = np.fft.fftshift(Y_filter_dB)

    plot_spectrum_mult(f_axis_shift, specturm_numpy, save_name)

    real_phase_est = PhaseModule.PhaseModule.phase_estimator(y_I_filter, y_Q_filter)
    real_phase_est_unwrap = PhaseModule.PhaseModule.phase_unwrapping(real_phase_est)
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Unwrapping estimated phase of PLL',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Unwrapping Phase', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, real_phase_est_unwrap, label='unwrapping estimated phase')
    plt.legend(fontsize=8)

    plt.show()
    # plot_spectrum(f_axis_shift, np.fft.fftshift(S_r_f_dB), 'real - 20-30 MHz (fs = 100 MHz)', 'Spectrum')

    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_I_f_dB), 'In-phase component', 'Spectrum')
    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_Q_f_dB), 'Quadrature-phase component', 'Spectrum')
    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_f_dB), 'IQ Demoulation (fs = 100 MHz)', 'Spectrum')

    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_I_filter_dB), 'In-phase component', 'Spectrum')
    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_Q_filter_dB), 'Quadrature-phase component', 'Spectrum')
    # plot_spectrum(f_axis_shift, np.fft.fftshift(Y_filter_dB), 'IQ Demoulation (fs = 100 MHz)', 'Spectrum')

    # fig_mag = plt.figure(figsize=(7, 4), dpi=100)
    # ax_mag = fig_mag.add_subplot()
    # ax_mag.set_title('Magnitude of Frequency Response', fontsize=12)
    # ax_mag.set_xlabel('log w', fontsize=10)
    # ax_mag.set_ylabel('Magnitude - dB', fontsize=10)
    # ax_mag.plot(np.linspace(-0.5, 0.5, num_point), 20 * np.log10(np.abs(freq_response)), label='filter module')
    # plt.tick_params(labelsize=10)
    # labels = ax_mag.get_xticklabels() + ax_mag.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    # plt.ylim(-60, 0)
    # plt.grid()
    # plt.legend(fontsize=8, loc='lower right')



