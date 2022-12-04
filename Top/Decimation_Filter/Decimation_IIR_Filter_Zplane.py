import matplotlib.font_manager as fm
import os
import sys
import numpy as np
from matplotlib.patches import Circle

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'ADCModule'))
import FilterModule

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

title_font = 25
label_font = 22
legend_font = 12
tick_font = 20


def plot_sub(z, p, qz, qp, fig, num_row, num_col, sub_fig):
    ax = fig.add_subplot(num_row, num_col, sub_fig)
    if sub_fig == 5 or sub_fig == 10 or sub_fig == 15:
        ax.set_title('Fig.' + str(sub_fig) + ' - Zplane of Cascade SOS',
                     fontsize=title_font, fontproperties=font_times)
    else:
        ax.set_title('Fig.' + str(sub_fig) + ' - Zplane of SOS ' + str(k),
                     fontsize=title_font, fontproperties=font_times)
    ax.set_xlabel('Re', fontsize=label_font, fontproperties=font_times)
    ax.set_ylabel('Im', fontsize=label_font, fontproperties=font_times)
    circle = Circle(xy=(0.0, 0.0), radius=1, alpha=0.9, facecolor='white')
    theta = np.linspace(0, 2 * np.pi, 200)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.add_patch(circle)
    ax.plot(x, y, color="k", linewidth=1.5)
    ax.plot(z.real, z.imag, linestyle='none', ms=10, markeredgecolor='b', markerfacecolor='b',
            marker='o', label='zeros')
    ax.plot(p.real, p.imag, linestyle='none', ms=10, markeredgecolor='b', markerfacecolor='b',
            marker='x', label='poles')

    ax.plot(qz.real, qz.imag, linestyle='none', ms=10, markeredgecolor='r', markerfacecolor='none',
            marker='o', label='Quantized zeros')
    ax.plot(qp.real, qp.imag, linestyle='none', ms=10, markeredgecolor='r', markerfacecolor='none',
            marker='x', label='Quantized poles')

    # plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig_zplane = plt.figure(figsize=(30, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    NUM_COL = 5
    NUM_ROW = 3

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

    ZEROS_2_CAS = np.zeros(8, dtype=complex)
    POLES_2_CAS = np.zeros(8, dtype=complex)

    ZEROS_2_FIX_CAS = np.zeros(8, dtype=complex)
    POLES_2_FIX_CAS = np.zeros(8, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC2_NUM_FIX = [1, b1_t[k], 1]
        DEC2_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        ZEROS_2, POLES_2, GAIN_2 = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM, DEC2_DEN)
        ZEROS_2_FIX, POLES_2_FIX, GAIN_2_FIX = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM_FIX, DEC2_DEN_FIX)

        ZEROS_2_CAS[2 * k:2 * (k + 1)] = ZEROS_2
        POLES_2_CAS[2 * k:2 * (k + 1)] = POLES_2
        ZEROS_2_FIX_CAS[2 * k:2 * (k + 1)] = ZEROS_2_FIX
        POLES_2_FIX_CAS[2 * k:2 * (k + 1)] = POLES_2_FIX

        SUB_FIG = k+1
        plot_sub(ZEROS_2, POLES_2, ZEROS_2_FIX, POLES_2_FIX, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

    SUB_FIG = 5
    plot_sub(ZEROS_2_CAS, POLES_2_CAS, ZEROS_2_FIX_CAS, POLES_2_FIX_CAS, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

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
    print(a1_t)
    print(a2_t)
    print(b1_t)

    # a1_fix = [0x1f, 0x2c, 0x3f, 0x51]
    # a2_fix = [0x14, 0x42, 0x87, 0xd4]
    # b1_fix = [0x65, 0x02, 0xd1, 0xc0]
    # print(a1_fix)
    # print(a2_fix)
    # print(b1_fix)

    ZEROS_4_CAS = np.zeros(8, dtype=complex)
    POLES_4_CAS = np.zeros(8, dtype=complex)

    ZEROS_4_FIX_CAS = np.zeros(8, dtype=complex)
    POLES_4_FIX_CAS = np.zeros(8, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC2_NUM_FIX = [1, b1_t[k], 1]
        DEC2_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        ZEROS_4, POLES_4, GAIN_2 = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM, DEC2_DEN)
        ZEROS_4_FIX, POLES_4_FIX, GAIN_2_FIX = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM_FIX, DEC2_DEN_FIX)

        ZEROS_4_CAS[2 * k:2 * (k + 1)] = ZEROS_4
        POLES_4_CAS[2 * k:2 * (k + 1)] = POLES_4
        ZEROS_4_FIX_CAS[2 * k:2 * (k + 1)] = ZEROS_4_FIX
        POLES_4_FIX_CAS[2 * k:2 * (k + 1)] = POLES_4_FIX

        SUB_FIG = 5 + k + 1
        plot_sub(ZEROS_4, POLES_4, ZEROS_4_FIX, POLES_4_FIX, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

    SUB_FIG = 10
    plot_sub(ZEROS_4_CAS, POLES_4_CAS, ZEROS_4_FIX_CAS, POLES_4_FIX_CAS, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

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
    print(a1_t)
    print(a2_t)
    print(b1_t)

    # a1_fix = [0x46, 0x53, 0x62, 0x6f]
    # a2_fix = [0x4f, 0x7c, 0xb4, 0xe6]
    # b1_fix = [0x2b, 0xb3, 0x9a, 0x93]
    # print(a1_fix)
    # print(a2_fix)
    # print(b1_fix)

    ZEROS_8_CAS = np.zeros(8, dtype=complex)
    POLES_8_CAS = np.zeros(8, dtype=complex)

    ZEROS_8_FIX_CAS = np.zeros(8, dtype=complex)
    POLES_8_FIX_CAS = np.zeros(8, dtype=complex)

    for k in range(4):
        DEC2_NUM = [1, b1[k], 1]
        DEC2_DEN = [1, -a1[k], a2[k]]

        DEC2_NUM_FIX = [1, b1_t[k], 1]
        DEC2_DEN_FIX = [1, -a1_t[k], a2_t[k]]

        ZEROS_8, POLES_8, GAIN_2 = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM, DEC2_DEN)
        ZEROS_8_FIX, POLES_8_FIX, GAIN_2_FIX = FilterModule.FilterModule.cal_zeros_poles(DEC2_NUM_FIX, DEC2_DEN_FIX)

        ZEROS_8_CAS[2 * k:2 * (k + 1)] = ZEROS_8
        POLES_8_CAS[2 * k:2 * (k + 1)] = POLES_8
        ZEROS_8_FIX_CAS[2 * k:2 * (k + 1)] = ZEROS_8_FIX
        POLES_8_FIX_CAS[2 * k:2 * (k + 1)] = POLES_8_FIX

        SUB_FIG = 10 + k + 1
        plot_sub(ZEROS_8, POLES_8, ZEROS_8_FIX, POLES_8_FIX, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

    SUB_FIG = 15
    plot_sub(ZEROS_8_CAS, POLES_8_CAS, ZEROS_8_FIX_CAS, POLES_8_FIX_CAS, fig_zplane, NUM_ROW, NUM_COL, SUB_FIG)

    plt.savefig('Decimation_IIR_Filter_Zplane.pdf')
    plt.show()
