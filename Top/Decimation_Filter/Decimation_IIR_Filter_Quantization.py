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
import CIC_Half_Band_Filter

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig_spec = plt.figure(figsize=(50, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    num_col = 6
    num_row = 3

    title_font = 25
    label_font = 22
    legend_font = 12
    tick_font = 20

    I = -1
    Q = 1
    COMPLEX = I + 1j * Q
    print(np.angle(COMPLEX))
    print(np.arctan(Q/I))

    """##############################################################################################################"""
    """ dec 2 """
    a1 = [-0.388190791409149, -0.285741152113574, -0.138801927534664, -0.025907440076305]
    a2 = [0.057398668463065, 0.186286541864827, 0.4230987127079, 0.76740478730534]
    b1 = [1.917977679727313, 1.419457762233652, 0.897984940087441, 0.615899604416527]

    a1_t = np.zeros(len(b1), dtype=float)
    a2_t = np.zeros(len(b1), dtype=float)
    b1_t = np.zeros(len(b1), dtype=float)
    for k in range(len(b1)):
        # a1_t[k] = round(a1[k] * (2 ** 6)) & 0xff
        a1_t[k] = round(a1[k] * (2 ** 6)) / (2 ** 6)
        # a2_t[k] = round(a2[k] * (2 ** 8)) & 0xff
        a2_t[k] = round(a2[k] * (2 ** 8)) / (2 ** 8)
        # b1_t[k] = round(b1[k] * (2 ** 6)) & 0xff
        b1_t[k] = round(b1[k] * (2 ** 6)) / (2 ** 6)
    print(a1_t)
    print(a2_t)
    print(b1_t)

    # a1_fix = [0xe7, 0xee, 0xf7, 0xfe]
    # a2_fix = [0x0f, 0x30, 0x6c, 0xc4]
    # b1_fix = [0x7b, 0x5b, 0x39, 0x27]
    # print(a1_fix)
    # print(a2_fix)
    # print(b1_fix)

    NUM_SPEC = 2**10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC2 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC2_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC2_ALL = np.ones(NUM_SPEC, dtype=complex)

    FREQ_RESPONSE_DEC2_FIX = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC2_FIX_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC2_FIX_ALL = np.ones(NUM_SPEC, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC2_NUM_FIX = [1, b1_t[k], 1]
        DEC2_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        FREQ_RESPONSE_DEC2[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC2_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2[k]))
        FREQ_RESPONSE_DEC2_ALL = FREQ_RESPONSE_DEC2_ALL * FREQ_RESPONSE_DEC2[k]

        FREQ_RESPONSE_DEC2_FIX[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM_FIX, DEC2_DEN_FIX,
                                                                           np.exp(1j * 2 * np.pi *
                                                                                  np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC2_FIX_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2_FIX[k]))
        FREQ_RESPONSE_DEC2_FIX_ALL = FREQ_RESPONSE_DEC2_FIX_ALL * FREQ_RESPONSE_DEC2_FIX[k]

        SUB_FIG = k+1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS '+str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC//2:], FREQ_RESPONSE_DEC2_dB[k][NUM_SPEC//2:], 'b', label='SOS '+str(k+1))
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_FIX_dB[k][NUM_SPEC // 2:], 'r--',
                     label='Quantized SOS' + str(k + 1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC2_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2_ALL))
    FREQ_RESPONSE_DEC2_ALL_dB = FREQ_RESPONSE_DEC2_ALL_dB - np.amax(FREQ_RESPONSE_DEC2_ALL_dB)

    FREQ_RESPONSE_DEC2_FIX_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2_FIX_ALL))
    FREQ_RESPONSE_DEC2_FIX_ALL_dB = FREQ_RESPONSE_DEC2_FIX_ALL_dB - np.amax(FREQ_RESPONSE_DEC2_FIX_ALL_dB)

    SUB_FIG = 5
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_FIX_ALL_dB[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC2_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC2_ALL)/np.real(FREQ_RESPONSE_DEC2_ALL))
    FREQ_RESPONSE_DEC2_ALL_ANG = np.angle(FREQ_RESPONSE_DEC2_ALL)
    FREQ_RESPONSE_DEC2_FIX_ALL_ANG = np.angle(FREQ_RESPONSE_DEC2_FIX_ALL)
    SUB_FIG = 6
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_FIX_ALL_ANG[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    plt.ylim(-np.pi, np.pi)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ dec 4 """
    a1 = [0.48243904855719, 0.694860383191498, 0.986516747573276, 1.264655494853073]
    a2 = [0.077096685422396, 0.25846165607158, 0.525852511749838, 0.826540395172634]
    b1 = [1.575900485199193, 0.038934471446067, -0.73188478016503, -0.999373895925541]

    a1_t = np.zeros(len(b1), dtype=float)
    a2_t = np.zeros(len(b1), dtype=float)
    b1_t = np.zeros(len(b1), dtype=float)
    for k in range(len(b1)):
        # a1_t[k] = round(a1[k] * (2 ** 6)) & 0xff
        a1_t[k] = round(a1[k] * (2 ** 6)) / (2 ** 6)
        # a2_t[k] = round(a2[k] * (2 ** 8)) & 0xff
        a2_t[k] = round(a2[k] * (2 ** 8)) / (2 ** 8)
        # b1_t[k] = round(b1[k] * (2 ** 6)) & 0xff
        b1_t[k] = round(b1[k] * (2 ** 6)) / (2 ** 6)
    # print(a1_t)
    # print(a2_t)
    # print(b1_t)

    # a1_fix = [0x1f, 0x2c, 0x3f, 0x51]
    # a2_fix = [0x14, 0x42, 0x87, 0xd4]
    # b1_fix = [0x65, 0x02, 0xd1, 0xc0]
    # print(a1_fix)
    # print(a2_fix)
    # print(b1_fix)

    NUM_SPEC = 2 ** 10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC4 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC4_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC4_ALL = np.ones(NUM_SPEC, dtype=complex)

    FREQ_RESPONSE_DEC4_FIX = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC4_FIX_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC4_FIX_ALL = np.ones(NUM_SPEC, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC4_NUM_FIX = [1, b1_t[k], 1]
        DEC4_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        FREQ_RESPONSE_DEC4[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC4_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4[k]))
        FREQ_RESPONSE_DEC4_ALL = FREQ_RESPONSE_DEC4_ALL * FREQ_RESPONSE_DEC4[k]

        FREQ_RESPONSE_DEC4_FIX[k] = FilterModule.FilterModule.digital_bode(DEC4_NUM_FIX, DEC4_DEN_FIX,
                                                                           np.exp(1j * 2 * np.pi *
                                                                                  np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC4_FIX_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4_FIX[k]))
        FREQ_RESPONSE_DEC4_FIX_ALL = FREQ_RESPONSE_DEC4_FIX_ALL * FREQ_RESPONSE_DEC4_FIX[k]

        SUB_FIG = 6 + k + 1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS ' + str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_dB[k][NUM_SPEC // 2:], 'b', label='SOS ' + str(k+1))
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_FIX_dB[k][NUM_SPEC // 2:], 'r--',
                     label='Quantized SOS' + str(k + 1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC4_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4_ALL))
    FREQ_RESPONSE_DEC4_ALL_dB = FREQ_RESPONSE_DEC4_ALL_dB - np.amax(FREQ_RESPONSE_DEC4_ALL_dB)

    FREQ_RESPONSE_DEC4_FIX_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4_FIX_ALL))
    FREQ_RESPONSE_DEC4_FIX_ALL_dB = FREQ_RESPONSE_DEC4_FIX_ALL_dB - np.amax(FREQ_RESPONSE_DEC4_FIX_ALL_dB)

    SUB_FIG = 11
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_FIX_ALL_dB[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC4_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC4_ALL) / np.real(FREQ_RESPONSE_DEC4_ALL))
    FREQ_RESPONSE_DEC4_ALL_ANG = np.angle(FREQ_RESPONSE_DEC4_ALL)
    FREQ_RESPONSE_DEC4_FIX_ALL_ANG = np.angle(FREQ_RESPONSE_DEC4_FIX_ALL)
    SUB_FIG = 12
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_FIX_ALL_ANG[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    plt.ylim(-np.pi, np.pi)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ dec 8 """
    a1 = [1.089165879969373, 1.293469185243449, 1.535701747984973, 1.742160371287951]
    a2 = [0.307225706291077, 0.484768643825525, 0.701544980890458, 0.900155860753369]
    b1 = [0.672052035914994, -1.204826510675321, -1.601088682038418, -1.705029177707256]

    a1_t = np.zeros(len(b1), dtype=float)
    a2_t = np.zeros(len(b1), dtype=float)
    b1_t = np.zeros(len(b1), dtype=float)
    for k in range(len(b1)):
        # a1_t[k] = round(a1[k] * (2 ** 6)) & 0xff
        a1_t[k] = round(a1[k] * (2 ** 6)) / (2 ** 6)
        # a2_t[k] = round(a2[k] * (2 ** 8)) & 0xff
        a2_t[k] = round(a2[k] * (2 ** 8)) / (2 ** 8)
        # b1_t[k] = round(b1[k] * (2 ** 6)) & 0xff
        b1_t[k] = round(b1[k] * (2 ** 6)) / (2 ** 6)
    # print(a1_t)
    # print(a2_t)
    # print(b1_t)

    # a1_fix = [0x46, 0x53, 0x62, 0x6f]
    # a2_fix = [0x4f, 0x7c, 0xb4, 0xe6]
    # b1_fix = [0x2b, 0xb3, 0x9a, 0x93]
    # print(a1_fix)
    # print(a2_fix)
    # print(b1_fix)

    NUM_SPEC = 2 ** 10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC8 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC8_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC8_ALL = np.ones(NUM_SPEC, dtype=complex)

    FREQ_RESPONSE_DEC8_FIX = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC8_FIX_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC8_FIX_ALL = np.ones(NUM_SPEC, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC8_NUM_FIX = [1, b1_t[k], 1]
        DEC8_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        FREQ_RESPONSE_DEC8[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC8_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8[k]))
        FREQ_RESPONSE_DEC8_ALL = FREQ_RESPONSE_DEC8_ALL * FREQ_RESPONSE_DEC8[k]

        FREQ_RESPONSE_DEC8_FIX[k] = FilterModule.FilterModule.digital_bode(DEC8_NUM_FIX, DEC8_DEN_FIX,
                                                                           np.exp(1j * 2 * np.pi *
                                                                                  np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC8_FIX_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8_FIX[k]))
        FREQ_RESPONSE_DEC8_FIX_ALL = FREQ_RESPONSE_DEC8_FIX_ALL * FREQ_RESPONSE_DEC8_FIX[k]

        SUB_FIG = 12 + k + 1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS ' + str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_dB[k][NUM_SPEC // 2:], 'b', label='SOS ' + str(k+1))
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_FIX_dB[k][NUM_SPEC // 2:], 'r--',
                     label='Quantized SOS' + str(k + 1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC8_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8_ALL))
    FREQ_RESPONSE_DEC8_ALL_dB = FREQ_RESPONSE_DEC8_ALL_dB - np.amax(FREQ_RESPONSE_DEC8_ALL_dB)

    FREQ_RESPONSE_DEC8_FIX_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8_FIX_ALL))
    FREQ_RESPONSE_DEC8_FIX_ALL_dB = FREQ_RESPONSE_DEC8_FIX_ALL_dB - np.amax(FREQ_RESPONSE_DEC8_FIX_ALL_dB)

    SUB_FIG = 17
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_FIX_ALL_dB[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC8_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC8_ALL) / np.real(FREQ_RESPONSE_DEC8_ALL))
    FREQ_RESPONSE_DEC8_ALL_ANG = np.angle(FREQ_RESPONSE_DEC8_ALL)
    FREQ_RESPONSE_DEC8_FIX_ALL_ANG = np.angle(FREQ_RESPONSE_DEC8_FIX_ALL)
    SUB_FIG = 18
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_FIX_ALL_ANG[NUM_SPEC // 2:], 'r--',
                 label='Quantized Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    plt.ylim(-np.pi, np.pi)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('Decimation_IIR_Filter_Quantization.pdf')
    plt.show()
