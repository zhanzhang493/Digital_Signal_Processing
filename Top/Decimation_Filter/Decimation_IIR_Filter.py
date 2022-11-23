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

    NUM_SPEC = 2**10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC2 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC2_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC2_ALL = np.ones(NUM_SPEC, dtype=complex)
    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]
        FREQ_RESPONSE_DEC2[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC2_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2[k]))
        FREQ_RESPONSE_DEC2_ALL = FREQ_RESPONSE_DEC2_ALL * FREQ_RESPONSE_DEC2[k]

        SUB_FIG = k+1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS '+str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC//2:], FREQ_RESPONSE_DEC2_dB[k][NUM_SPEC//2:], 'b', label='SOS'+str(k+1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC2_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC2_ALL))
    FREQ_RESPONSE_DEC2_ALL_dB = FREQ_RESPONSE_DEC2_ALL_dB - np.amax(FREQ_RESPONSE_DEC2_ALL_dB)
    SUB_FIG = 5
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC2_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC2_ALL)/np.real(FREQ_RESPONSE_DEC2_ALL))
    FREQ_RESPONSE_DEC2_ALL_ANG = np.angle(FREQ_RESPONSE_DEC2_ALL)
    SUB_FIG = 6
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC2_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
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

    NUM_SPEC = 2 ** 10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC4 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC4_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC4_ALL = np.ones(NUM_SPEC, dtype=complex)
    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]
        FREQ_RESPONSE_DEC4[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC4_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4[k]))
        FREQ_RESPONSE_DEC4_ALL = FREQ_RESPONSE_DEC4_ALL * FREQ_RESPONSE_DEC4[k]

        SUB_FIG = 6 + k + 1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS ' + str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_dB[k][NUM_SPEC // 2:], 'b', label='SOS' + str(k+1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC4_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC4_ALL))
    FREQ_RESPONSE_DEC4_ALL_dB = FREQ_RESPONSE_DEC4_ALL_dB - np.amax(FREQ_RESPONSE_DEC4_ALL_dB)
    SUB_FIG = 11
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC4_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC4_ALL) / np.real(FREQ_RESPONSE_DEC4_ALL))
    FREQ_RESPONSE_DEC4_ALL_ANG = np.angle(FREQ_RESPONSE_DEC4_ALL)
    SUB_FIG = 12
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC4_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
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

    NUM_SPEC = 2 ** 10
    F_AXIS_SHIFT = np.linspace(-1, 1, NUM_SPEC)
    FREQ_RESPONSE_DEC8 = np.zeros((4, NUM_SPEC), dtype=complex)
    FREQ_RESPONSE_DEC8_dB = np.zeros((4, NUM_SPEC), dtype=float)
    FREQ_RESPONSE_DEC8_ALL = np.ones(NUM_SPEC, dtype=complex)
    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]
        FREQ_RESPONSE_DEC8[k] = FilterModule.FilterModule.digital_bode(DEC2_NUM, DEC2_DEN,
                                                                       np.exp(1j * 2 * np.pi *
                                                                              np.linspace(-0.5, 0.5, NUM_SPEC)))
        FREQ_RESPONSE_DEC8_dB[k] = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8[k]))
        FREQ_RESPONSE_DEC8_ALL = FREQ_RESPONSE_DEC8_ALL * FREQ_RESPONSE_DEC8[k]

        SUB_FIG = 12 + k + 1
        ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
        ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of SOS ' + str(k),
                          fontsize=title_font, fontproperties=font_times)
        ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
        ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_dB[k][NUM_SPEC // 2:], 'b', label='SOS' + str(k+1))
        plt.legend(fontsize=legend_font)
        plt.tick_params(labelsize=tick_font)
        plt.grid('on')
        plt.xlim(0, 1)
        plt.ylim(-60, 25)
        labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

    FREQ_RESPONSE_DEC8_ALL_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_DEC8_ALL))
    FREQ_RESPONSE_DEC8_ALL_dB = FREQ_RESPONSE_DEC8_ALL_dB - np.amax(FREQ_RESPONSE_DEC8_ALL_dB)
    SUB_FIG = 17
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_ALL_dB[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # FREQ_RESPONSE_DEC8_ALL_ANG = np.arctan(np.imag(FREQ_RESPONSE_DEC8_ALL) / np.real(FREQ_RESPONSE_DEC8_ALL))
    FREQ_RESPONSE_DEC8_ALL_ANG = np.angle(FREQ_RESPONSE_DEC8_ALL)
    SUB_FIG = 18
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Phase Response of Cascade SOS ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Omega(pi)', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - rad', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT[NUM_SPEC // 2:], FREQ_RESPONSE_DEC8_ALL_ANG[NUM_SPEC // 2:], 'b', label='Cascade SOS')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(0, 1)
    plt.ylim(-np.pi, np.pi)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('Decimation_IIR_Filter.pdf')
    plt.show()
