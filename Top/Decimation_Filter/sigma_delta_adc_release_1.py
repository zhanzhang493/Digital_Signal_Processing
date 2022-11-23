import re
import matplotlib.font_manager as fm
import os
import sys
import numpy as np
import time
current_time = time.strftime('%Y%m%d_%H%M%S')

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'ADCModule'))
import SigmaDeltaADCModule
import FilterModule

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """##############################################################################################################"""
    """##############################################################################################################"""
    """ Parameter Setting """
    """##############################################################################################################"""
    """##############################################################################################################"""
    HALF_SPREAD = 1
    NUM_BIT = 1
    DELTA_RANGE = 2 * HALF_SPREAD / NUM_BIT
    N_SAMPLE = 256
    FC = 30
    SAMPLE_RATE = 100
    OSR = 20
    N_SH = 1
    FFT_SIZE_FACTOR = 4
    NUM_OSR = N_SAMPLE * OSR
    OSR_SAMPLE_RATE = SAMPLE_RATE * OSR
    ANALOG_RATE = OSR_SAMPLE_RATE * N_SH
    NUM_ANALOG = NUM_OSR * N_SH

    OSR_FFT_SIZE = NUM_OSR * FFT_SIZE_FACTOR
    SAMPLE_FFT_SIZE = N_SAMPLE * FFT_SIZE_FACTOR

    NUM_SPEC = NUM_OSR

    F_AXIS_SHIFT_ANALOG = np.arange(-int(NUM_ANALOG / 2), int(NUM_ANALOG / 2)) / NUM_ANALOG * ANALOG_RATE
    F_AXIS_SHIFT_SPEC = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC * OSR_SAMPLE_RATE
    F_AXIS_SHIFT_OSR_FFT = np.arange(-int(OSR_FFT_SIZE / 2), int(OSR_FFT_SIZE / 2)) / OSR_FFT_SIZE * OSR_SAMPLE_RATE

    T_AXIS_ANALOG = np.arange(NUM_ANALOG) / ANALOG_RATE
    T_AXIS_OSR = T_AXIS_ANALOG[::4]
    T_AXIS = T_AXIS_OSR[::OSR]

    """##############################################################################################################"""
    """##############################################################################################################"""
    """ Filter Plot """
    """##############################################################################################################"""
    """##############################################################################################################"""
    fig_spec = plt.figure(figsize=(45, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.13, hspace=0.25)
    num_col = 3
    num_row = 3

    title_font = 25
    label_font = 22
    legend_font = 12
    tick_font = 20

    """ CIC Filter """
    NUM_FILTER = 5
    NUM_CASCADE = 4

    """ 4-stage CIC Filter """
    CIC_NUM = [1, 4, 10, 20, 35, 52, 68, 80, 85, 80, 68, 52, 35, 20, 10, 4, 1]
    DEN = [1]
    FREQ_RESPONSE_4_CIC = FilterModule.FilterModule.digital_bode(CIC_NUM, DEN,
                                                                 np.exp(1j * 2 * np.pi *
                                                                        np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_4_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_4_CIC))
    FREQ_RESPONSE_RELATIVE_4_STAGE_dB = FREQ_RESPONSE_4_STAGE_dB - np.amax(FREQ_RESPONSE_4_STAGE_dB)

    SUB_FIG = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade 4 CIC of 5-Order',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC, FREQ_RESPONSE_RELATIVE_4_STAGE_dB, 'b-', linewidth=1.5, label='4-stage CIC')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-1000, 1000)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ Half-band Filter """
    NUM_ORDER = 11
    N_AXIS = np.arange(-int((NUM_ORDER - 1) // 2), int((NUM_ORDER + 1) // 2))
    # COE_HALF_BAND, FREQ_RESPONSE_HALF_BAND = CIC_Half_Band_Filter.half_band_filter(NUM_ORDER, NUM_SPEC)

    H_HALF = np.zeros(NUM_ORDER)
    with open('half_band_coe.txt') as f:
        for k in range(NUM_ORDER):
            line = f.readline()
            data_str = re.search('.*', line).group(0)
            data = eval(data_str)
            H_HALF[k] = data
    FREQ_RESPONSE_HALF_BAND = FilterModule.FilterModule.digital_bode(H_HALF, DEN,
                                                                     np.exp(1j * 2 * np.pi *
                                                                            np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_HALF_BAND_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND))
    FREQ_RESPONSE_HALF_BAND_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_dB - np.amax(FREQ_RESPONSE_HALF_BAND_dB)
    # H = COE_HALF_BAND[0]
    H = H_HALF
    H_5 = np.zeros(5 * (NUM_ORDER - 1) + 1)

    for k in range(len(H)):
        H_5[5 * k] = H[k]

    FREQ_RESPONSE_HALF_BAND_5 = FilterModule.FilterModule.digital_bode(H_5, DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_HALF_BAND_5_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND_5))
    FREQ_RESPONSE_HALF_BAND_5_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_5_dB - np.amax(FREQ_RESPONSE_HALF_BAND_5_dB)

    SUB_FIG = 2
    F_AXIS_SHIFT_5 = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC * OSR_SAMPLE_RATE / 5
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Half-Band Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_5, FREQ_RESPONSE_HALF_BAND_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Half-band filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-200, 200)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 3
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Half-Band Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC, FREQ_RESPONSE_HALF_BAND_5_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Half-band filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-1000, 1000)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    FREQ_RESPONSE_CIC_HALF_BAND_5 = FREQ_RESPONSE_HALF_BAND_5 * FREQ_RESPONSE_4_CIC
    FREQ_RESPONSE_CIC_HALF_BAND_5_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_CIC_HALF_BAND_5))
    FREQ_RESPONSE_CIC_HALF_BAND_5_RELATIVE_dB = FREQ_RESPONSE_CIC_HALF_BAND_5_dB - \
                                                np.amax(FREQ_RESPONSE_CIC_HALF_BAND_5_dB)

    SUB_FIG = 4
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade CIC and Half-Band Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC, FREQ_RESPONSE_CIC_HALF_BAND_5_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Cascade CIC and Half-band filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-1000, 1000)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ Compensation Fitler """
    H_COMP = np.zeros(51)
    with open('comb_coe.txt') as f:
        for k in range(51):
            line = f.readline()
            data_str = re.search('.*', line).group(0)
            data = eval(data_str)
            H_COMP[k] = data

    FREQ_RESPONSE_H_COMP = FilterModule.FilterModule.digital_bode(H_COMP, DEN, np.exp(1j * 2 * np.pi *
                                                                                      np.linspace(-0.5, 0.5, NUM_SPEC)))

    FREQ_RESPONSE_H_COMP_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_H_COMP))
    FREQ_RESPONSE_H_COMP_RELATIVE_dB = FREQ_RESPONSE_H_COMP_dB - np.amax(FREQ_RESPONSE_H_COMP_dB)

    SUB_FIG = 5
    F_AXIS_SHIFT_10 = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC * OSR_SAMPLE_RATE / 10
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_10, FREQ_RESPONSE_H_COMP_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-100, 100)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    H_COMP_10 = np.zeros(10 * (51 - 1) + 1)

    for k in range(len(H_COMP)):
        H_COMP_10[10 * k] = H_COMP[k]

    FREQ_RESPONSE_H_COMP_10 = FilterModule.FilterModule.digital_bode(H_COMP_10, DEN,
                                                                     np.exp(1j * 2 * np.pi
                                                                            * np.linspace(-0.5, 0.5, NUM_SPEC)))

    FREQ_RESPONSE_H_COMP_10_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_H_COMP_10))
    FREQ_RESPONSE_H_COMP_10_RELATIVE_dB = FREQ_RESPONSE_H_COMP_10_dB - np.amax(FREQ_RESPONSE_H_COMP_10_dB)

    SUB_FIG = 6
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC, FREQ_RESPONSE_H_COMP_10_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-1000, 1000)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10 = FREQ_RESPONSE_H_COMP_10 * FREQ_RESPONSE_HALF_BAND_5 * FREQ_RESPONSE_4_CIC
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_CIC_HALF_BAND_COMP_10))
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB = FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB - \
                                                      np.amax(FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB)

    SUB_FIG = 7
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Cascade CIC, Half-Band, and Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC[NUM_SPEC//2:], FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB[NUM_SPEC//2:],
                 'b-', linewidth=1.5,
                 label='Cascade CIC, half-band, and compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1000)
    plt.ylim(-150, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 8
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Cascade CIC, Half-Band, and Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC[int(NUM_SPEC // 2):int(NUM_SPEC / 20 * 11)],
                 FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB[int(NUM_SPEC // 2):int(NUM_SPEC / 20 * 11)], 'b-',
                 linewidth=1.5,
                 label='Cascade CIC, half-band, and compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 100)
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 9
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Cascade CIC, Half-Band, and Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_SPEC[int(NUM_SPEC // 2):int(NUM_SPEC / 40 * 21)],
                 FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB[int(NUM_SPEC // 2):int(NUM_SPEC / 40 * 21)], 'b-',
                 linewidth=1.5,
                 label='Cascade CIC, half-band, and compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 50)
    plt.ylim(-0.5, 0.1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('Decimation_Filter_CIC_HB_COMP.pdf')

    """##############################################################################################################"""
    """##############################################################################################################"""
    """ SDM Plot """
    """##############################################################################################################"""
    """##############################################################################################################"""
    fig_spec = plt.figure(figsize=(45, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.13, hspace=0.25)
    num_col = 2
    num_row = 2

    title_font = 25
    label_font = 22
    legend_font = 12
    tick_font = 20

    AMP = 0.7
    # input signal
    S_ANALOG = AMP * np.sin(2 * FC * np.pi * T_AXIS_ANALOG) + 0.1 * (np.random.rand(NUM_ANALOG) - 0.5)

    """ sigma-delta Quantization """

    # _, _, v = SigmaDeltaADCModule.ADCModule.first_order(S_ANALOG, DELTA_RANGE)
    # _, _, _, _, _, v = SigmaDeltaADCModule.ADCModule.second_order_sample_hold(S_ANALOG, DELTA_RANGE, N_SH)
    # _, _, _, _, _, v = SigmaDeltaADCModule.ADCModule.second_order(S_ANALOG, DELTA_RANGE)
    # _, _, _, _, v = SigmaDeltaADCModule.ADCModule.third_order(S_ANALOG, DELTA_RANGE)
    v = SigmaDeltaADCModule.ADCModule.third_order_mash(S_ANALOG)

    V_F = np.abs(np.fft.fft(v, n=OSR_FFT_SIZE))
    V_F_dB = 20 * np.log10(V_F)
    MAX_VALUE_V_F = np.max(V_F_dB)
    V_F_RELATIVE_dB = V_F_dB - MAX_VALUE_V_F

    SUB_FIG = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Noise Shaping of Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[len(V_F_RELATIVE_dB) // 2:],
                     V_F_RELATIVE_dB[0:len(V_F_RELATIVE_dB) // 2], 'b--', label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 2
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Noise Shaping of Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[len(V_F_RELATIVE_dB) // 2:],
                 V_F_RELATIVE_dB[0:len(V_F_RELATIVE_dB) // 2], 'b--', label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    v_CIC = FilterModule.FilterModule.digital_filter(CIC_NUM, DEN, v)
    v_CIC_HALF = FilterModule.FilterModule.digital_filter(H_5, DEN, v_CIC)
    v_CIC_HALF_COMP = FilterModule.FilterModule.digital_filter(H_COMP_10, DEN, v_CIC_HALF)

    V_CIC_HALF_COMP = np.abs(np.fft.fft(v_CIC_HALF_COMP, n=OSR_FFT_SIZE))
    V_CIC_HALF_COMP_dB = 20 * np.log10(V_CIC_HALF_COMP)
    V_CIC_HALF_COMP_RELATIVE_dB = V_CIC_HALF_COMP_dB - np.amax(V_CIC_HALF_COMP_dB)

    SUB_FIG = 3
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title(
        'Fig.' + str(SUB_FIG) + ' - Decimation Filtering Result',
        fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[len(V_F_RELATIVE_dB) // 2:],
                     V_CIC_HALF_COMP_RELATIVE_dB[0:len(V_CIC_HALF_COMP_RELATIVE_dB) // 2],
                     'b--', linewidth=2, label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    v_CIC_HALF_COMP_DECIMATE = v_CIC_HALF_COMP[::OSR]
    V_CIC_HALF_COMP_DECIMATE = np.abs(np.fft.fft(v_CIC_HALF_COMP_DECIMATE, n=SAMPLE_FFT_SIZE))
    V_CIC_HALF_COMP_DECIMATE_dB = 20 * np.log10(V_CIC_HALF_COMP_DECIMATE)
    V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB = V_CIC_HALF_COMP_DECIMATE_dB - np.amax(V_CIC_HALF_COMP_DECIMATE_dB)

    SUB_FIG = 4
    F_AXIS_SHIFT_20 = np.arange(-int(SAMPLE_FFT_SIZE / 2), int(SAMPLE_FFT_SIZE / 2)) / SAMPLE_FFT_SIZE * SAMPLE_RATE
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title(
        'Fig.' + str(SUB_FIG) + ' - Decimation Filtering Result',
        fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_20[len(F_AXIS_SHIFT_20) // 2:],
                 V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB[0:len(V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB) // 2],
                 'b--', linewidth=2, label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    # plt.savefig('Sigma_delta_adc_decimation_'+current_time+'.pdf')

    plt.show()