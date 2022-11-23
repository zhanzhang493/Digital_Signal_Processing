import matplotlib.font_manager as fm
import os
import sys
import numpy as np

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
import FilterModule

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9


def hamming(num_order):
    n = np.arange(num_order)
    win = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_order - 1))
    return win


def cic_filter(num_fitler, num_spec):
    h = np.zeros(num_fitler+1, dtype=int)
    h[0] = 1
    h[num_fitler] = -1
    b = h
    a = [1, -1]

    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, num_spec)))
    filter_coe = [b, a]
    return filter_coe, freq_response


def cic_filter_FIR(num_fitler, num_spec):
    h = np.ones(num_fitler)
    b = h
    a = [1]

    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, num_spec)))
    filter_coe = [b, a]
    return filter_coe, freq_response


def half_band_filter(num_filter, num_spec):
    assert (num_filter - 1) % 2 == 0
    n = np.arange(-int((num_filter - 1) // 2), int((num_filter + 1) // 2))
    h = np.sin(np.pi/2 * n) / np.pi / n
    h[int((num_filter - 1) // 2)] = 0.5
    # win = hamming(num_filter)
    # h = h * win
    a = [1]

    freq_response = FilterModule.FilterModule.digital_bode(h, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, num_spec)))
    filter_coe = [h, a]
    return filter_coe, freq_response


def cic_compensation_filter(b, num_filter, num_spec):
    a = [1]
    amp = -2**(-b-2)
    b_1st = -(2**(b+2) + 2)
    b = np.zeros(3*num_filter+1)
    b[0] = 0
    b[num_filter] = -2**(-4)
    b[2*num_filter] = 2**(-4)*(2**4+2)
    b[3*num_filter] = -2**(-4)
    filter_coe = [b, a]
    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, num_spec)))

    return filter_coe, freq_response


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """##############################################################################################################"""
    """ CIC 1st Filter"""
    NUM_SPEC = 1000
    NUM_FILTER = 10
    NUM_CASCADE = 4

    """ 1-stage CIC Filter """
    FILTER_COE, FREQ_RESPONSE = cic_filter(NUM_FILTER, NUM_SPEC)
    FREQ_RESPONSE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE))
    FREQ_RESPONSE_RELATIVE_dB = FREQ_RESPONSE_dB - np.amax(FREQ_RESPONSE_dB)

    """ 2-stage CIC Filter """
    FREQ_RESPONSE_2_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE ** 2))
    FREQ_RESPONSE_RELATIVE_2_STAGE_dB = FREQ_RESPONSE_2_STAGE_dB - np.amax(FREQ_RESPONSE_2_STAGE_dB)

    """ 3-stage CIC Filter"""
    FREQ_RESPONSE_3_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE ** 3))
    FREQ_RESPONSE_RELATIVE_3_STAGE_dB = FREQ_RESPONSE_3_STAGE_dB - np.amax(FREQ_RESPONSE_3_STAGE_dB)

    """ 4-stage CIC Filter"""
    FREQ_RESPONSE_4_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE ** 4))
    FREQ_RESPONSE_RELATIVE_4_STAGE_dB = FREQ_RESPONSE_4_STAGE_dB - np.amax(FREQ_RESPONSE_4_STAGE_dB)

    f_axis_shift = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC
    
    fig_spec = plt.figure(figsize=(31.5, 6), dpi=50)
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.85, wspace=0.13, hspace=0.25)
    num_col = 3
    num_row = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, 1)
    ax_spec.set_title('Fig.1 - Frequency Response of CIC Filter',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - [-0.5, 0.5]', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=22, fontproperties=font_times)
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_RELATIVE_dB, 'k', linewidth=3, label='1-stage CIC, M = 5')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_RELATIVE_2_STAGE_dB, 'b', linewidth=3, label='2-stage CIC, M = 5')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_RELATIVE_3_STAGE_dB, 'r', linewidth=3, label='3-stage CIC, M = 5')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_RELATIVE_4_STAGE_dB, 'g', linewidth=3, label='4-stage CIC, M = 5')
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=22)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    """============================================================================================================="""
    NUM = [1]
    DEN = [1, -1]
    FREQ_RESPONSE_I_FILTER = FilterModule.FilterModule.digital_bode(NUM, DEN,
                                                                    np.exp(1j * 2 * np.pi *
                                                                           np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_I_FILTER_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_I_FILTER))
    FREQ_RESPONSE_I_FILTER_RELATIVE_dB = FREQ_RESPONSE_I_FILTER_dB - np.amax(FREQ_RESPONSE_I_FILTER_dB)

    FREQ_RESPONSE_I_FILTER_2_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_I_FILTER ** 2))
    FREQ_RESPONSE_I_FILTER_RELATIVE_2_STAGE_dB = FREQ_RESPONSE_I_FILTER_2_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_I_FILTER_2_STAGE_dB)

    FREQ_RESPONSE_I_FILTER_3_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_I_FILTER ** 3))
    FREQ_RESPONSE_I_FILTER_RELATIVE_3_STAGE_dB = FREQ_RESPONSE_I_FILTER_3_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_I_FILTER_3_STAGE_dB)

    FREQ_RESPONSE_I_FILTER_4_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_I_FILTER ** 4))
    FREQ_RESPONSE_I_FILTER_RELATIVE_4_STAGE_dB = FREQ_RESPONSE_I_FILTER_4_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_I_FILTER_4_STAGE_dB)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 2)
    ax_spec.set_title('Fig.2 - Frequency Response of integrator Filter',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - [-0.5, 0.5]', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=22, fontproperties=font_times)
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_I_FILTER_RELATIVE_dB, 'k', linewidth=3, label='1-stage integrator filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_I_FILTER_RELATIVE_2_STAGE_dB, 'b', linewidth=3,
                 label='2-stage integrator filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_I_FILTER_RELATIVE_3_STAGE_dB, 'r', linewidth=3,
                 label='3-stage integrator filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_I_FILTER_RELATIVE_4_STAGE_dB, 'g', linewidth=3,
                 label='4-stage integrator filter')
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=22)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """============================================================================================================="""
    NUM = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]
    DEN = [1]
    FREQ_RESPONSE_C_FILTER = FilterModule.FilterModule.digital_bode(NUM, DEN,
                                                                    np.exp(1j * 2 * np.pi
                                                                           * np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_C_FILTER_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_C_FILTER))
    FREQ_RESPONSE_C_FILTER_RELATIVE_dB = FREQ_RESPONSE_C_FILTER_dB - np.amax(FREQ_RESPONSE_C_FILTER_dB)

    FREQ_RESPONSE_C_FILTER_2_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_C_FILTER ** 2))
    FREQ_RESPONSE_C_FILTER_RELATIVE_2_STAGE_dB = FREQ_RESPONSE_C_FILTER_2_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_C_FILTER_2_STAGE_dB)

    FREQ_RESPONSE_C_FILTER_3_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_C_FILTER ** 3))
    FREQ_RESPONSE_C_FILTER_RELATIVE_3_STAGE_dB = FREQ_RESPONSE_C_FILTER_3_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_C_FILTER_3_STAGE_dB)

    FREQ_RESPONSE_C_FILTER_4_STAGE_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_C_FILTER ** 4))
    FREQ_RESPONSE_C_FILTER_RELATIVE_4_STAGE_dB = FREQ_RESPONSE_C_FILTER_4_STAGE_dB - \
                                                 np.amax(FREQ_RESPONSE_C_FILTER_4_STAGE_dB)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 3)
    ax_spec.set_title('Fig.3 - Frequency Response of "comb" Filter',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - [-0.5, 0.5]', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=22, fontproperties=font_times)
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_C_FILTER_RELATIVE_dB, 'k', linewidth=3, label='1-stage comb filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_C_FILTER_RELATIVE_2_STAGE_dB, 'b', linewidth=3,
                 label='2-stage comb filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_C_FILTER_RELATIVE_3_STAGE_dB, 'r', linewidth=3,
                 label='3-stage comb filter')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_C_FILTER_RELATIVE_4_STAGE_dB, 'g', linewidth=3,
                 label='4-stage comb filter')
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=22)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('CIC_Filter_multi.pdf')
    
    """##############################################################################################################"""
    """Half band filter"""
    NUM_ORDER = 11
    N_AXIS = np.arange(-int((NUM_ORDER - 1) // 2), int((NUM_ORDER + 1) // 2))
    COE_HALF_BAND, FREQ_RESPONSE_HALF_BAND = half_band_filter(NUM_ORDER, NUM_SPEC)
    FREQ_RESPONSE_HALF_BAND_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND))
    FREQ_RESPONSE_HALF_BAND_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_dB - np.amax(FREQ_RESPONSE_HALF_BAND_dB)
    H = COE_HALF_BAND[0]

    NUM_ORDER = 27
    N_AXIS_27 = np.arange(-int((NUM_ORDER - 1) // 2), int((NUM_ORDER + 1) // 2))
    COE_HALF_BAND_27, FREQ_RESPONSE_HALF_BAND_27 = half_band_filter(NUM_ORDER, NUM_SPEC)
    FREQ_RESPONSE_HALF_BAND_27_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND_27))
    FREQ_RESPONSE_HALF_BAND_27_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_27_dB - np.amax(FREQ_RESPONSE_HALF_BAND_27_dB)
    H_27 = COE_HALF_BAND_27[0]

    NUM_ORDER = 51
    N_AXIS_51 = np.arange(-int((NUM_ORDER - 1) // 2), int((NUM_ORDER + 1) // 2))
    COE_HALF_BAND_51, FREQ_RESPONSE_HALF_BAND_51 = half_band_filter(NUM_ORDER, NUM_SPEC)
    FREQ_RESPONSE_HALF_BAND_51_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_HALF_BAND_51))
    FREQ_RESPONSE_HALF_BAND_51_RELATIVE_dB = FREQ_RESPONSE_HALF_BAND_51_dB - np.amax(FREQ_RESPONSE_HALF_BAND_51_dB)
    H_51 = COE_HALF_BAND_51[0]

    fig_spec = plt.figure(figsize=(31.5, 6), dpi=50)
    plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.85, wspace=0.13, hspace=0.25)
    num_col = 3
    num_row = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, 1)

    ax_spec.set_title('Fig.1 - Impulse Response of Half-Band Filter',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('n', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Amplitude', fontsize=10, fontproperties=font_times)
    # ax_spec.plot(f_axis_shift, FREQ_RESPONSE_HALF_BAND_RELATIVE_dB, 'k', linewidth=1.5, label='Half-band filter')
    ax_spec.plot(N_AXIS, H, 'ko-', linewidth=1.5, label='M = 11')
    ax_spec.plot(N_AXIS_27, H_27, 'bx--', linewidth=1.5, label='M = 27')
    ax_spec.plot(N_AXIS_51, H_51, 'rv--', linewidth=1.5, label='M = 51')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=10)
    plt.grid('on')
    # plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 2)
    ax_spec.set_title('Fig.2 - Frequency Response of Half-Band Filter',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - [-0.5, 0.5]', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude', fontsize=10, fontproperties=font_times)
    ax_spec.plot(f_axis_shift, np.abs(FREQ_RESPONSE_HALF_BAND), 'k', linewidth=1.5, label='M = 11')
    ax_spec.plot(f_axis_shift, np.abs(FREQ_RESPONSE_HALF_BAND_27), 'b', linewidth=1.5, label='M = 27')
    ax_spec.plot(f_axis_shift, np.abs(FREQ_RESPONSE_HALF_BAND_51), 'r', linewidth=1.5, label='M = 51')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=10)
    plt.grid('on')
    # plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 3)
    ax_spec.set_title('Fig.3 - Frequency Response of Half-Band Filter',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - [-0.5, 0.5]', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=10, fontproperties=font_times)
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_HALF_BAND_RELATIVE_dB, 'k', linewidth=3, label='M = 11')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_HALF_BAND_27_RELATIVE_dB, 'b', linewidth=2, label='M = 27')
    ax_spec.plot(f_axis_shift, FREQ_RESPONSE_HALF_BAND_51_RELATIVE_dB, 'r--', linewidth=1.5, label='M = 51')
    plt.legend(fontsize=8)
    plt.tick_params(labelsize=10)
    plt.grid('on')
    plt.ylim(-100, 1)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    
    """##############################################################################################################"""
    plt.savefig('half_band_filter.pdf')

    plt.show()