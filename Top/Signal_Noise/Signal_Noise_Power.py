import matplotlib.font_manager as fm
import os
import sys
import numpy as np
import time
current_time = time.strftime('%Y%m%d_%H%M%S')

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'Window'))
import Window

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

title_font = 25
label_font = 22
legend_font = 12
tick_font = 20


def plot_sub(x, y, fig, num_row, num_col, sub_fig, title, xlabel, ylabel, ylim):
    ax = fig.add_subplot(num_row, num_col, sub_fig)
    ax.set_title('Fig.' + str(sub_fig) + '-' + title,
                 fontsize=title_font, fontproperties=font_times)
    ax.set_xlabel(xlabel, fontsize=label_font, fontproperties=font_times)
    ax.set_ylabel(ylabel, fontsize=label_font, fontproperties=font_times)
    ax.plot(x, y, 'b-')
    # plt.legend(fontsize=legend_font)
    plt.ylim(ylim[0], ylim[1])
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


def signal_noise_adjust(signal, noise, snr):
    signal_norm = np.linalg.norm(signal)
    # print('signal_power:', signal_norm**2)

    noise_norm = np.linalg.norm(noise)
    # print('noise_power:', noise_norm**2)

    signal_to_noise_desired = 10 ** (snr / 10)
    signal_to_noise_input = (signal_norm ** 2) / (noise_norm ** 2)

    noise_gain = signal_to_noise_desired / signal_to_noise_input
    # print('noise factor:', 1 / np.sqrt(noise_gain))

    noise_adjust = 1 / np.sqrt(noise_gain) * noise
    noise_adjust_norm = np.linalg.norm(noise_adjust)
    # print('noise_adjust_power:', noise_adjust_norm ** 2)

    snr_adjust = 20 * np.log10(signal_norm / noise_adjust_norm)
    # print('SNR:', snr_adjust)

    signal_noise = signal + noise_adjust
    return signal, noise_adjust, signal_noise


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    FIG_SIG = plt.figure(figsize=(50, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    NUM_COL = 6
    NUM_ROW = 3

    """##############################################################################################################"""
    MAX_SIMU = 1000
    SNR = 10
    NUM_POINT = 2**10

    FFT_FACTOR = 4
    NUM_FFT = NUM_POINT * FFT_FACTOR
    N_AXIS = np.arange(NUM_POINT)
    SAMPLE_RATE = 100
    T_AXIS = N_AXIS / SAMPLE_RATE
    F_AXIS = np.arange(NUM_FFT) / NUM_FFT * SAMPLE_RATE

    print('===========================')
    print('Case 1:')
    print('SNR:', SNR)
    print('Num of point:', NUM_POINT)

    FC = 300 / NUM_FFT * 100
    AMP = 1
    SIGNAL = AMP * np.cos(2 * np.pi * FC / SAMPLE_RATE * N_AXIS)
    WIN = Window.WindowModule.hanning(NUM_POINT, 'periodic')

    SIGNAL_F = np.fft.fft(SIGNAL, n=NUM_FFT)
    SIGNAL_F_dB = 20 * np.log10(np.abs(SIGNAL_F))

    SIGNAL_WIN_F = np.fft.fft(SIGNAL * WIN, n=NUM_FFT)
    SIGNAL_WIN_F_dB = 20 * np.log10(np.abs(SIGNAL_WIN_F))

    SIGNAL_NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    for k in range(MAX_SIMU):
        # print('Simu:', k)
        NOISE = np.random.randn(NUM_POINT)
        _, NOISE_ADJUST, SIGNAL_NOISE = signal_noise_adjust(SIGNAL, NOISE, SNR)
        NOISE_SIMU[k] = NOISE_ADJUST.copy()
        SIGNAL_NOISE_SIMU[k] = SIGNAL_NOISE.copy()
        del NOISE_ADJUST, SIGNAL_NOISE

    NOISE_F = np.fft.fft(NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(NOISE_F) ** 2, axis=0)
    NOISE_COMP_F_dB = 10 * np.log10(np.abs(NOISE_COMP_F))

    SIGNAL_NOISE_F = np.fft.fft(SIGNAL_NOISE_SIMU, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_F) ** 2, axis=0)
    SIGNAL_NOISE_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_COMP_F))

    SIGNAL_NOISE_WIN_F = np.fft.fft(SIGNAL_NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_WIN_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_WIN_F) ** 2, axis=0)
    SIGNAL_NOISE_WIN_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_WIN_COMP_F))

    print('In-band SNR', SNR + 10 * np.log10(NUM_POINT))
    print('Calculate In-band SNR with window', 10 * np.log10(np.sum(SIGNAL_NOISE_WIN_COMP_F[300 - 6:300 + 7])
                                                             / np.average(SIGNAL_NOISE_WIN_COMP_F[300 - 37:300 - 7])))

    SUB_FIG = 1
    plot_sub(T_AXIS[0:128], SIGNAL[0:128], FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Time Domain', 'Time',
             'Amplitude', [-1.1, 1.1])

    SUB_FIG = 2
    plot_sub(F_AXIS, SIGNAL_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 3
    plot_sub(F_AXIS, SIGNAL_WIN_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 4
    plot_sub(F_AXIS, SIGNAL_NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 5
    plot_sub(F_AXIS, SIGNAL_NOISE_WIN_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 6
    plot_sub(F_AXIS, NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])
    """##############################################################################################################"""
    MAX_SIMU = 1000
    SNR = 10
    NUM_POINT = (2 ** 10) * 10

    FFT_FACTOR = 4
    NUM_FFT = NUM_POINT * FFT_FACTOR
    N_AXIS = np.arange(NUM_POINT)
    SAMPLE_RATE = 100
    T_AXIS = N_AXIS / SAMPLE_RATE
    F_AXIS = np.arange(NUM_FFT) / NUM_FFT * SAMPLE_RATE

    print('===========================')
    print('Case 2:')
    print('SNR:', SNR)
    print('Num of point:', NUM_POINT)

    FC = 3000 / NUM_FFT * 100
    AMP = 1
    SIGNAL = AMP * np.cos(2 * np.pi * FC / SAMPLE_RATE * N_AXIS)
    WIN = Window.WindowModule.hanning(NUM_POINT, 'periodic')

    SIGNAL_F = np.fft.fft(SIGNAL, n=NUM_FFT)
    SIGNAL_F_dB = 20 * np.log10(np.abs(SIGNAL_F))

    SIGNAL_WIN_F = np.fft.fft(SIGNAL * WIN, n=NUM_FFT)
    SIGNAL_WIN_F_dB = 20 * np.log10(np.abs(SIGNAL_WIN_F))

    SIGNAL_NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    for k in range(MAX_SIMU):
        # print('Simu:', k)
        NOISE = np.random.randn(NUM_POINT)
        _, NOISE_ADJUST, SIGNAL_NOISE = signal_noise_adjust(SIGNAL, NOISE, SNR)
        NOISE_SIMU[k] = NOISE_ADJUST.copy()
        SIGNAL_NOISE_SIMU[k] = SIGNAL_NOISE.copy()
        del NOISE_ADJUST, SIGNAL_NOISE

    NOISE_F = np.fft.fft(NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(NOISE_F) ** 2, axis=0)
    NOISE_COMP_F_dB = 10 * np.log10(np.abs(NOISE_COMP_F))

    SIGNAL_NOISE_F = np.fft.fft(SIGNAL_NOISE_SIMU, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_F) ** 2, axis=0)
    SIGNAL_NOISE_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_COMP_F))

    SIGNAL_NOISE_WIN_F = np.fft.fft(SIGNAL_NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_WIN_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_WIN_F) ** 2, axis=0)
    SIGNAL_NOISE_WIN_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_WIN_COMP_F))

    print('In-band SNR', SNR + 10 * np.log10(NUM_POINT))
    print('Calculate In-band SNR with window', 10 * np.log10(np.sum(SIGNAL_NOISE_WIN_COMP_F[3000 - 16:3000 + 17])
                                                             / np.average(SIGNAL_NOISE_WIN_COMP_F[3000 - 57:3000 - 17])))

    SUB_FIG = 7
    plot_sub(T_AXIS[0:128], SIGNAL[0:128], FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Time Domain', 'Time',
             'Amplitude', [-1.1, 1.1])

    SUB_FIG = 8
    plot_sub(F_AXIS, SIGNAL_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 9
    plot_sub(F_AXIS, SIGNAL_WIN_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 10
    plot_sub(F_AXIS, SIGNAL_NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 11
    plot_sub(F_AXIS, SIGNAL_NOISE_WIN_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 12
    plot_sub(F_AXIS, NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    """##############################################################################################################"""
    MAX_SIMU = 1000
    SNR = 20
    NUM_POINT = (2 ** 10) * 10

    FFT_FACTOR = 4
    NUM_FFT = NUM_POINT * FFT_FACTOR
    N_AXIS = np.arange(NUM_POINT)
    SAMPLE_RATE = 100
    T_AXIS = N_AXIS / SAMPLE_RATE
    F_AXIS = np.arange(NUM_FFT) / NUM_FFT * SAMPLE_RATE

    print('===========================')
    print('Case 3:')
    print('SNR:', SNR)
    print('Num of point:', NUM_POINT)

    FC = 3000 / NUM_FFT * 100
    AMP = 1
    SIGNAL = AMP * np.cos(2 * np.pi * FC / SAMPLE_RATE * N_AXIS)
    WIN = Window.WindowModule.hanning(NUM_POINT, 'periodic')

    SIGNAL_F = np.fft.fft(SIGNAL, n=NUM_FFT)
    SIGNAL_F_dB = 20 * np.log10(np.abs(SIGNAL_F))

    SIGNAL_WIN_F = np.fft.fft(SIGNAL * WIN, n=NUM_FFT)
    SIGNAL_WIN_F_dB = 20 * np.log10(np.abs(SIGNAL_WIN_F))

    SIGNAL_NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    NOISE_SIMU = np.zeros((MAX_SIMU, NUM_POINT), dtype=float)
    for k in range(MAX_SIMU):
        # print('Simu:', k)
        NOISE = np.random.randn(NUM_POINT)
        _, NOISE_ADJUST, SIGNAL_NOISE = signal_noise_adjust(SIGNAL, NOISE, SNR)
        NOISE_SIMU[k] = NOISE_ADJUST.copy()
        SIGNAL_NOISE_SIMU[k] = SIGNAL_NOISE.copy()
        del NOISE_ADJUST, SIGNAL_NOISE

    NOISE_F = np.fft.fft(NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(NOISE_F) ** 2, axis=0)
    NOISE_COMP_F_dB = 10 * np.log10(np.abs(NOISE_COMP_F))

    SIGNAL_NOISE_F = np.fft.fft(SIGNAL_NOISE_SIMU, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_F) ** 2, axis=0)
    SIGNAL_NOISE_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_COMP_F))

    SIGNAL_NOISE_WIN_F = np.fft.fft(SIGNAL_NOISE_SIMU * WIN, n=NUM_FFT, axis=-1)
    SIGNAL_NOISE_WIN_COMP_F = 1 / MAX_SIMU * np.sum(np.abs(SIGNAL_NOISE_WIN_F) ** 2, axis=0)
    SIGNAL_NOISE_WIN_COMP_F_dB = 10 * np.log10(np.abs(SIGNAL_NOISE_WIN_COMP_F))

    print('In-band SNR', SNR + 10 * np.log10(NUM_POINT))
    print('Calculate In-band SNR with window', 10 * np.log10(np.sum(SIGNAL_NOISE_WIN_COMP_F[3000 - 16:3000 + 17])
                                                             / np.average(SIGNAL_NOISE_WIN_COMP_F[3000 - 57:3000 - 17])))

    SUB_FIG = 13
    plot_sub(T_AXIS[0:128], SIGNAL[0:128], FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Time Domain', 'Time',
             'Amplitude', [-1.1, 1.1])

    SUB_FIG = 14
    plot_sub(F_AXIS, SIGNAL_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 15
    plot_sub(F_AXIS, SIGNAL_WIN_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 16
    plot_sub(F_AXIS, SIGNAL_NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum without window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 17
    plot_sub(F_AXIS, SIGNAL_NOISE_WIN_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])

    SUB_FIG = 18
    plot_sub(F_AXIS, NOISE_COMP_F_dB, FIG_SIG, NUM_ROW, NUM_COL, SUB_FIG, 'Spectrum with window', 'Freq',
             'Magnitude - dB', [0, 85])
    """##############################################################################################################"""
    plt.savefig('Signal_Noise_SNR_NumPoint_'+str(MAX_SIMU)+'_'+current_time+'.pdf')
    plt.show()


