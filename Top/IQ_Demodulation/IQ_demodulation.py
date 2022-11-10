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
    ax_sub.set_title(title, fontsize=22, fontproperties=font_times)
    ax_sub.set_xlabel('Freq - MHz', fontsize=18, fontproperties=font_times)
    ax_sub.set_ylabel('Magnitude - dB', fontsize=18, fontproperties=font_times)
    ax_sub.plot(freq_sub / MHz, spectrum_sub)
    plt.tick_params(labelsize=18)
    labels = ax_sub.get_xticklabels() + ax_sub.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.ylim(0, 51)
    plt.grid()


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
        plot_spectrum_sub(ax_spec, freq_axis, spectrum, 'Fig.'+str(k+1)+ ' - '+ title_list[k])
        if k == 4:
            plt.ylim(-70, 1)
        else:
            plt.ylim(0, 51)

    plt.savefig(save_name+'.pdf')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """======================================================================="""
    """ ============================= Case 1 ================================="""
    """======================================================================="""
    """ input signal """
    FS = 200 * MHz
    TS = 1 / FS

    B = 10 * MHz
    FC = 30 * MHz
    LO_FC = 30 * MHz
    PRI = 10.24 * us
    NUM_POINT = int(FS * PRI)
    T_AXIS = np.arange(NUM_POINT) * TS
    ALPHA = B / PRI
    AMP = 2
    BETA = AMP / PRI
    SAVE_NAME = 'IQ_begin_freq'

    F_AXIS = np.arange(NUM_POINT) / NUM_POINT * FS
    F_AXIS_SHIFT = np.arange(-int(NUM_POINT/2), int(NUM_POINT/2)) / NUM_POINT * FS

    """======================================================================="""
    """ input signal """
    IDEAL_FMCW_PHASE = 2 * np.pi * (FC * T_AXIS + ALPHA * (T_AXIS ** 2) / 2)
    S = (1 + BETA * T_AXIS) * np.exp(1j * IDEAL_FMCW_PHASE)

    S_R = np.real(S)
    S_R_F = np.fft.fft(S_R)
    S_R_F_dB = 20 * np.log10(np.abs(S_R_F))

    """======================================================================="""
    """ IQ Demodulation  """
    Y_I, Y_Q = IQModulation.IQModule.IQ_demodulator(S_R, LO_FC, FS)

    Y_I_F = np.fft.fft(Y_I)
    Y_I_F_dB = 20 * np.log10(np.abs(Y_I_F))

    Y_Q_F = np.fft.fft(Y_Q)
    Y_Q_F_dB = 20 * np.log10(np.abs(Y_Q_F))

    Y = Y_I + 1j * Y_Q
    Y_F = np.fft.fft(Y)
    Y_F_dB = 20 * np.log10(np.abs(Y_F))

    """======================================================================="""
    """ Filtering """
    NUM_FILTER = 127
    WC = 0.1 * np.pi
    N_AXIS = np.arange(-int((NUM_FILTER-1)//2), int((NUM_FILTER+1)//2))
    H = np.sin(WC * N_AXIS) / np.pi / N_AXIS
    H[int((NUM_FILTER-1)//2)] = WC / np.pi

    WIN = hamming(NUM_FILTER)
    H_LPF = H * WIN
    NUM = H_LPF
    DEN = [1]

    FREQ_RESPONSE = FilterModule.FilterModule.digital_bode(NUM, DEN,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, NUM_POINT)))
    FREQ_RESPONSE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE))
    FREQ_RESPONSE_dB = FREQ_RESPONSE_dB - np.amax(FREQ_RESPONSE_dB)

    Y_I_FILTER = FilterModule.FilterModule.digital_filter(NUM, DEN, Y_I)
    Y_I_FILTER_F = np.fft.fft(Y_I_FILTER)
    Y_I_FILTER_F_dB = 20 * np.log10(np.abs(Y_I_FILTER_F))

    Y_Q_FILTER = FilterModule.FilterModule.digital_filter(NUM, DEN, Y_Q)
    Y_Q_FILTER_F = np.fft.fft(Y_Q_FILTER)
    Y_Q_FILTER_F_dB = 20 * np.log10(np.abs(Y_Q_FILTER_F))

    Y_FILTER = Y_I_FILTER + 1j * Y_Q_FILTER
    Y_FILTER_F = np.fft.fft(Y_FILTER)
    Y_FILTER_F_dB = 20 * np.log10(np.abs(Y_FILTER_F))

    """======================================================================="""
    """ Figure """
    SPECTRUM_NUMPY = np.zeros((8, NUM_POINT), dtype=float)
    SPECTRUM_NUMPY[0] = np.fft.fftshift(S_R_F_dB)
    SPECTRUM_NUMPY[1] = np.fft.fftshift(Y_I_F_dB)
    SPECTRUM_NUMPY[2] = np.fft.fftshift(Y_I_FILTER_F_dB)
    SPECTRUM_NUMPY[3] = np.fft.fftshift(Y_F_dB)
    SPECTRUM_NUMPY[4] = FREQ_RESPONSE_dB
    SPECTRUM_NUMPY[5] = np.fft.fftshift(Y_Q_F_dB)
    SPECTRUM_NUMPY[6] = np.fft.fftshift(Y_Q_FILTER_F_dB)
    SPECTRUM_NUMPY[7] = np.fft.fftshift(Y_FILTER_F_dB)

    plot_spectrum_mult(F_AXIS_SHIFT, SPECTRUM_NUMPY, SAVE_NAME)

    """======================================================================="""
    """ ============================= Case 2 ================================="""
    """======================================================================="""
    """ input signal """
    FS = 200 * MHz
    TS = 1 / FS

    B = 10 * MHz
    FC = 30 * MHz
    LO_FC = 35 * MHz
    PRI = 10.24 * us
    NUM_POINT = int(FS * PRI)
    T_AXIS = np.arange(NUM_POINT) * TS
    ALPHA = B / PRI
    AMP = 2
    BETA = AMP / PRI
    SAVE_NAME = 'IQ_center_freq'

    F_AXIS = np.arange(NUM_POINT) / NUM_POINT * FS
    F_AXIS_SHIFT = np.arange(-int(NUM_POINT / 2), int(NUM_POINT / 2)) / NUM_POINT * FS

    """======================================================================="""
    """ input signal """
    IDEAL_FMCW_PHASE = 2 * np.pi * (FC * T_AXIS + ALPHA * (T_AXIS ** 2) / 2)
    S = (1 + BETA * T_AXIS) * np.exp(1j * IDEAL_FMCW_PHASE)

    S_R = np.real(S)
    S_R_F = np.fft.fft(S_R)
    S_R_F_dB = 20 * np.log10(np.abs(S_R_F))

    """======================================================================="""
    """ IQ Demodulation  """
    Y_I, Y_Q = IQModulation.IQModule.IQ_demodulator(S_R, LO_FC, FS)

    Y_I_F = np.fft.fft(Y_I)
    Y_I_F_dB = 20 * np.log10(np.abs(Y_I_F))

    Y_Q_F = np.fft.fft(Y_Q)
    Y_Q_F_dB = 20 * np.log10(np.abs(Y_Q_F))

    Y = Y_I + 1j * Y_Q
    Y_F = np.fft.fft(Y)
    Y_F_dB = 20 * np.log10(np.abs(Y_F))

    """======================================================================="""
    """ Filtering """
    NUM_FILTER = 127
    WC = 0.1 * np.pi
    N_AXIS = np.arange(-int((NUM_FILTER - 1) // 2), int((NUM_FILTER + 1) // 2))
    H = np.sin(WC * N_AXIS) / np.pi / N_AXIS
    H[int((NUM_FILTER - 1) // 2)] = WC / np.pi

    WIN = hamming(NUM_FILTER)
    H_LPF = H * WIN
    NUM = H_LPF
    DEN = [1]

    FREQ_RESPONSE = FilterModule.FilterModule.digital_bode(NUM, DEN,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, NUM_POINT)))
    FREQ_RESPONSE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE))
    FREQ_RESPONSE_dB = FREQ_RESPONSE_dB - np.amax(FREQ_RESPONSE_dB)

    Y_I_FILTER = FilterModule.FilterModule.digital_filter(NUM, DEN, Y_I)
    Y_I_FILTER_F = np.fft.fft(Y_I_FILTER)
    Y_I_FILTER_F_dB = 20 * np.log10(np.abs(Y_I_FILTER_F))

    Y_Q_FILTER = FilterModule.FilterModule.digital_filter(NUM, DEN, Y_Q)
    Y_Q_FILTER_F = np.fft.fft(Y_Q_FILTER)
    Y_Q_FILTER_F_dB = 20 * np.log10(np.abs(Y_Q_FILTER_F))

    Y_FILTER = Y_I_FILTER + 1j * Y_Q_FILTER
    Y_FILTER_F = np.fft.fft(Y_FILTER)
    Y_FILTER_F_dB = 20 * np.log10(np.abs(Y_FILTER_F))

    """======================================================================="""
    """ Figure """
    SPECTRUM_NUMPY = np.zeros((8, NUM_POINT), dtype=float)
    SPECTRUM_NUMPY[0] = np.fft.fftshift(S_R_F_dB)
    SPECTRUM_NUMPY[1] = np.fft.fftshift(Y_I_F_dB)
    SPECTRUM_NUMPY[2] = np.fft.fftshift(Y_I_FILTER_F_dB)
    SPECTRUM_NUMPY[3] = np.fft.fftshift(Y_F_dB)
    SPECTRUM_NUMPY[4] = FREQ_RESPONSE_dB
    SPECTRUM_NUMPY[5] = np.fft.fftshift(Y_Q_F_dB)
    SPECTRUM_NUMPY[6] = np.fft.fftshift(Y_Q_FILTER_F_dB)
    SPECTRUM_NUMPY[7] = np.fft.fftshift(Y_FILTER_F_dB)

    plot_spectrum_mult(F_AXIS_SHIFT, SPECTRUM_NUMPY, SAVE_NAME)

    plt.show()



