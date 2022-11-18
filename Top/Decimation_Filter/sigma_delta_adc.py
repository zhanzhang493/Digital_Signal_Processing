import re
import matplotlib.font_manager as fm
import os
import sys
import numpy as np

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'ADCModule'))
import SigmaDeltaADCModule
import FilterModule
import CIC_Filter

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig_spec = plt.figure(figsize=(45, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.13, hspace=0.25)
    num_col = 4
    num_row = 3
    
    title_font = 25
    label_font = 22
    legend_font = 12
    tick_font = 20
    
    """##############################################################################################################"""
    """ Sigma Delta Modulator Test"""
    HALF_SPREAD = 1
    NUM_BIT = 1
    N_SAMPLE = 1024
    FC = 30
    SAMPLE_RATE = 100
    OSR = 20
    NUM_OSR = N_SAMPLE * OSR
    OSR_SAMPLE_RATE = SAMPLE_RATE * OSR

    NUM_SPEC = NUM_OSR

    F_AXIS_SHIFT_OSR = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC * OSR_SAMPLE_RATE

    T_AXIS_OSR = np.arange(NUM_OSR) / OSR_SAMPLE_RATE
    T_AXIS = T_AXIS_OSR[::OSR]

    AMP = 0.8
    # input signal
    S_ANALOG = AMP * np.sin(2 * FC * np.pi * T_AXIS_OSR)

    """ DS Quantization """
    S_SAMPLE = S_ANALOG[::OSR]
    S_SAMPLE_Q = SigmaDeltaADCModule.ADCModule.quantizer_add_noise(HALF_SPREAD, NUM_BIT, S_SAMPLE)

    u, y, v = SigmaDeltaADCModule.ADCModule.first_order(S_ANALOG, HALF_SPREAD, NUM_BIT)
    u2, x2, y2, v2 = SigmaDeltaADCModule.ADCModule.second_order(S_ANALOG, HALF_SPREAD, NUM_BIT)
    u3, x3_0, x3_1, y3, v3 = SigmaDeltaADCModule.ADCModule.third_order(S_ANALOG, HALF_SPREAD, NUM_BIT)

    S_ANALOG_Q = SigmaDeltaADCModule.ADCModule.quantizer_add_noise(HALF_SPREAD, NUM_BIT, S_ANALOG)
    S_ANALOG_Q_F = np.abs(np.fft.fft(S_ANALOG_Q))
    S_ANALOG_Q_F_dB = 20 * np.log10(S_ANALOG_Q_F)
    S_ANALOG_Q_F_RELATIVE_dB = S_ANALOG_Q_F_dB - np.amax(S_ANALOG_Q_F_dB)

    V_F = np.abs(np.fft.fft(v3))
    V_F_dB = 20 * np.log10(V_F)
    MAX_VALUE_V_F = np.max(V_F_dB)
    V_F_RELATIVE_dB = V_F_dB - MAX_VALUE_V_F

    SUB_FIG = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Noise Shaping of Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
                     S_ANALOG_Q_F_RELATIVE_dB[0:len(V_F_RELATIVE_dB) // 2], 'r-', label='DQ')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
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
    ax_spec.plot(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
                 S_ANALOG_Q_F_RELATIVE_dB[0:len(V_F_RELATIVE_dB) // 2], 'r-', label='DQ')
    ax_spec.plot(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
                 V_F_RELATIVE_dB[0:len(V_F_RELATIVE_dB) // 2], 'b--', label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    """##############################################################################################################"""
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

    SUB_FIG = 3
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of 4 Cascade 5-Order CIC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR, FREQ_RESPONSE_RELATIVE_4_STAGE_dB, 'b-', linewidth=1.5, label='4-stage CIC')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ Half-band Filter """
    NUM_ORDER = 13
    N_AXIS = np.arange(-int((NUM_ORDER - 1) // 2), int((NUM_ORDER + 1) // 2))
    COE_HALF_BAND, FREQ_RESPONSE_HALF_BAND = CIC_Filter.half_band_filter(NUM_ORDER, NUM_SPEC)
    FREQ_RESPONSE_HALF_BAND_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND))
    FREQ_RESPONSE_HALF_BAND_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_dB - np.amax(FREQ_RESPONSE_HALF_BAND_dB)
    H = COE_HALF_BAND[0]
    H_5 = np.zeros(5 * (NUM_ORDER - 1) + 1)

    for k in range(len(H)):
        H_5[5 * k] = H[k]

    FREQ_RESPONSE_HALF_BAND_5 = FilterModule.FilterModule.digital_bode(H_5, DEN,
                                                                        np.exp(1j * 2 * np.pi *
                                                                               np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_HALF_BAND_5_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND_5))
    FREQ_RESPONSE_HALF_BAND_5_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_5_dB - np.amax(FREQ_RESPONSE_HALF_BAND_5_dB)

    SUB_FIG = 4
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
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 5
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Half-Band Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR, FREQ_RESPONSE_HALF_BAND_5_RELATIVE_dB, 'b-', linewidth=1.5, label='Half-band filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    FREQ_RESPONSE_CIC_HALF_BAND_5 = FREQ_RESPONSE_HALF_BAND_5 * FREQ_RESPONSE_4_CIC
    FREQ_RESPONSE_CIC_HALF_BAND_5_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_CIC_HALF_BAND_5))
    FREQ_RESPONSE_CIC_HALF_BAND_5_RELATIVE_dB = FREQ_RESPONSE_CIC_HALF_BAND_5_dB - \
                                                 np.amax(FREQ_RESPONSE_CIC_HALF_BAND_5_dB)

    SUB_FIG = 6
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade CIC and Half-Band Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR, FREQ_RESPONSE_CIC_HALF_BAND_5_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Cascade CIC and Half-band filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
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

    SUB_FIG = 7
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

    SUB_FIG = 8
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR, FREQ_RESPONSE_H_COMP_10_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10 = FREQ_RESPONSE_H_COMP_10 * FREQ_RESPONSE_HALF_BAND_5 * FREQ_RESPONSE_4_CIC
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_CIC_HALF_BAND_COMP_10))
    FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB = FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB - \
                                                      np.amax(FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_dB)

    SUB_FIG = 9
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade CIC, Half-Band, and Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR, FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB, 'b-', linewidth=1.5,
                 label='Cascade CIC, half-band, and compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 10
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade CIC, Half-Band, and Compensation Filter',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR[int(NUM_SPEC//2):int(NUM_SPEC / 20 * 11)],
                 FREQ_RESPONSE_CIC_HALF_BAND_COMP_10_RELATIVE_dB[int(NUM_SPEC//2):int(NUM_SPEC / 20 * 11)], 'b-',
                 linewidth=1.5,
                 label='Cascade CIC, half-band, and compensation filter')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    v_CIC = FilterModule.FilterModule.digital_filter(CIC_NUM, DEN, v3)
    v_CIC_HALF = FilterModule.FilterModule.digital_filter(H_5, DEN, v_CIC)
    v_CIC_HALF_COMP = FilterModule.FilterModule.digital_filter(H_COMP_10, DEN, v_CIC_HALF)

    V_CIC_HALF_COMP = np.abs(np.fft.fft(v_CIC_HALF_COMP))
    V_CIC_HALF_COMP_dB = 20 * np.log10(V_CIC_HALF_COMP)
    V_CIC_HALF_COMP_RELATIVE_dB = V_CIC_HALF_COMP_dB - np.amax(V_CIC_HALF_COMP_dB)

    S_ANALOG_Q_CIC = FilterModule.FilterModule.digital_filter(CIC_NUM, DEN, S_ANALOG_Q)
    S_ANALOG_Q_CIC_HALF = FilterModule.FilterModule.digital_filter(H_5, DEN, S_ANALOG_Q_CIC)
    S_ANALOG_Q_CIC_HALF_COMP = FilterModule.FilterModule.digital_filter(H_COMP_10, DEN, S_ANALOG_Q_CIC_HALF)

    S_ANALOG_Q_CIC_HALF_COMP_F = np.abs(np.fft.fft(S_ANALOG_Q_CIC_HALF_COMP))
    S_ANALOG_Q_CIC_HALF_COMP_F_dB = 20 * np.log10(S_ANALOG_Q_CIC_HALF_COMP_F)
    S_ANALOG_Q_CIC_HALF_COMP_F_RELATIVE_dB = S_ANALOG_Q_CIC_HALF_COMP_F_dB - np.amax(S_ANALOG_Q_CIC_HALF_COMP_F_dB)

    SUB_FIG = 11
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title(
        'Fig.' + str(SUB_FIG) + ' - Decimation Filtering Result',
        fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
                 S_ANALOG_Q_CIC_HALF_COMP_F_RELATIVE_dB[0:len(V_CIC_HALF_COMP_RELATIVE_dB) // 2], 'r-', linewidth=2,
                 label='DQ')
    ax_spec.plot(F_AXIS_SHIFT_OSR[len(V_F_RELATIVE_dB) // 2:],
                 V_CIC_HALF_COMP_RELATIVE_dB[0:len(V_CIC_HALF_COMP_RELATIVE_dB) // 2], 'b--', linewidth=2, label='SD3')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    v_CIC_HALF_COMP_DECIMATE = v_CIC_HALF_COMP[::OSR]
    V_CIC_HALF_COMP_DECIMATE = np.abs(np.fft.fft(v_CIC_HALF_COMP_DECIMATE))
    V_CIC_HALF_COMP_DECIMATE_dB = 20 * np.log10(V_CIC_HALF_COMP_DECIMATE)
    V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB = V_CIC_HALF_COMP_DECIMATE_dB - np.amax(V_CIC_HALF_COMP_DECIMATE_dB)

    S_ANALOG_Q_CIC_HALF_COMP_DECIMATE = S_ANALOG_Q_CIC_HALF_COMP[::OSR]
    S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F = np.abs(np.fft.fft(S_ANALOG_Q_CIC_HALF_COMP_DECIMATE))
    S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F_dB = 20 * np.log10(S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F)
    S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F_RELATIVE_dB = S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F_dB -\
                                                               np.amax(S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F_dB)

    # S_SAMPLE_Q_DECIMATE = S_ANALOG_Q[::OSR]
    # S_SAMPLE_Q_DECIMATE_F = np.abs(np.fft.fft(S_SAMPLE_Q_DECIMATE))
    # S_SAMPLE_Q_DECIMATE_F_dB = 20 * np.log10(S_SAMPLE_Q_DECIMATE_F)
    # S_SAMPLE_Q_DECIMATE_F_RELATIVE_dB = S_SAMPLE_Q_DECIMATE_F_dB - np.amax(S_SAMPLE_Q_DECIMATE_F_dB)

    SUB_FIG = 12
    F_AXIS_SHIFT_20 = np.arange(-int(N_SAMPLE / 2), int(N_SAMPLE / 2)) / N_SAMPLE * SAMPLE_RATE
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title(
        'Fig.' + str(SUB_FIG) + ' - Decimation Filtering Result',
        fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    # ax_spec.plot(F_AXIS_SHIFT_20[len(F_AXIS_SHIFT_20) // 2:],
    #              S_SAMPLE_Q_DECIMATE_F_RELATIVE_dB[0:len(V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB) // 2],
    #              'k-', linewidth=2, label='no filter')
    ax_spec.plot(F_AXIS_SHIFT_20[len(F_AXIS_SHIFT_20) // 2:],
                 S_ANALOG_Q_CIC_HALF_COMP_DECIMATE_DECIMATE_F_RELATIVE_dB[0:len(V_CIC_HALF_COMP_DECIMATE_RELATIVE_dB) // 2],
                 'r-', linewidth=2, label='DQ')
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
    plt.savefig('Sigma_delta_adc_decimation.pdf')

    plt.show()