import matplotlib.font_manager as fm
import os
import sys
import numpy as np

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'PhaseModule'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
import PhaseModule
import FilterModule

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9


def hamming(num_order):
    n = np.arange(num_order)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_order - 1))
    return win


def qiyong_win(num_order):
    n = np.arange(num_order)
    win = 15/16 - 1/16 * np.cos(2 * np.pi * n / (num_order - 1))
    return win


def ricky_win(num_order):
    n = np.arange(num_order)
    win = 1 / 16 - 15 / 16 * np.sin(np.pi * n / (num_order - 1))
    return win
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    NUM_FILTER = 127
    WC = 0.5 * np.pi
    N_AXIS = np.arange(-int((NUM_FILTER - 1) // 2), int((NUM_FILTER + 1) // 2))
    H = np.sin(WC * N_AXIS) / np.pi / N_AXIS
    H[int((NUM_FILTER - 1) // 2)] = WC / np.pi

    HAMMING_WIN = hamming(NUM_FILTER)
    H_HAMMING = H * HAMMING_WIN

    QIYONG_WIN = qiyong_win(NUM_FILTER)
    H_QIYONG = H * QIYONG_WIN

    RICKY_WIN = ricky_win(NUM_FILTER)
    H_RICKY = H * RICKY_WIN

    b = H_HAMMING
    b_qiyong = H_QIYONG
    b_ricky = H_RICKY
    a = [1]

    HAMMING_WIN_F = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(HAMMING_WIN, n=1024))))
    HAMMING_WIN_F = HAMMING_WIN_F - np.amax(HAMMING_WIN_F)

    QIYONG_WIN_F = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(QIYONG_WIN, n=1024))))
    QIYONG_WIN_F = QIYONG_WIN_F - np.amax(QIYONG_WIN_F)

    RICKY_WIN_F = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(RICKY_WIN, n=1024))))
    RICKY_WIN_F = RICKY_WIN_F - np.amax(RICKY_WIN_F)

    fig_spec = plt.figure()
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title('Frequency Response of Phase-Domain Filter',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=8, fontproperties=font_times)
    ax_spec.plot(HAMMING_WIN_F, 'k', linewidth=1, label='HAMMING')
    ax_spec.plot(QIYONG_WIN_F, 'b', linewidth=1, label='qiyong')
    ax_spec.plot(RICKY_WIN_F, 'r', linewidth=1, label='ricky')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=8)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, 1000)))
    freq_response_dB = 20 * np.log10(np.abs(freq_response))
    freq_response_dB = freq_response_dB - np.amax(freq_response_dB)

    freq_response_qiyong = FilterModule.FilterModule.digital_bode(b_qiyong, a,
                                                                  np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, 1000)))
    freq_response_qiyong_dB = 20 * np.log10(np.abs(freq_response_qiyong))
    freq_response_qiyong_dB = freq_response_qiyong_dB - np.amax(freq_response_qiyong_dB)

    freq_response_ricky = FilterModule.FilterModule.digital_bode(b_ricky, a,
                                                                 np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, 1000)))
    freq_response_ricky_dB = 20 * np.log10(np.abs(freq_response_ricky))
    freq_response_ricky_dB = freq_response_ricky_dB - np.amax(freq_response_ricky_dB)

    f_axis_shift = np.arange(-int(1000 / 2), int(1000 / 2)) / 1000
    fig_spec = plt.figure()
    ax_spec = fig_spec.add_subplot()
    ax_spec.set_title('Frequency Response of Phase-Domain Filter',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=8, fontproperties=font_times)
    ax_spec.plot(f_axis_shift / MHz, freq_response_dB, 'k', linewidth=1, label='HAMMING')
    ax_spec.plot(f_axis_shift / MHz, freq_response_qiyong_dB, 'b--', linewidth=1, label='qiyong')
    ax_spec.plot(f_axis_shift / MHz, freq_response_ricky_dB, 'r--', linewidth=1, label='ricky')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=8)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()