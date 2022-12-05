import re
import matplotlib.font_manager as fm
import os
import sys
import numpy as np

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FilterDesign'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'ADCModule'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'FixedPoint'))
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'Window'))
import FilterModule
import Quantization
import Window

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9

title_font = 25
label_font = 22
legend_font = 12
tick_font = 20


def digital_filter(num, den, ipt):
    num_u = len(num)
    num_y = len(den)
    u = np.zeros(num_u)
    y = np.zeros(num_y)

    opt = np.zeros(np.shape(ipt))

    for idx, value in np.ndenumerate(ipt):
        u[0] = value

        y[0] = 0
        for uix in range(num_u):
            y[0] += num[uix] * u[uix]

        if num_y != 1:
            for yix in range(num_y - 1):
                y[0] -= den[yix + 1] * y[yix + 1]

        y[0] = y[0] / den[0]

        if num_u != 1:
            for uidx in range(num_u - 1):
                u[num_u - uidx - 1] = u[num_u - uidx - 2]

        if num_y != 1:
            for yidx in range(num_y - 1):
                y[num_y - yidx - 1] = y[num_y - yidx - 2]

        opt[idx] = y[0]

    return opt


def decimation_IIR_filter_float(coe, ipt):
    # coe 0, 1, 2: b1, a1, a2
    assert len(coe) == 3

    num_u = 3
    num_y = 3

    u = np.zeros(num_u)
    y = np.zeros(num_y)

    opt = np.zeros(np.shape(ipt))

    for idx, value in np.ndenumerate(ipt):
        u[0] = value

        y[0] = 0

        u0_tmp = u[0]
        u1_tmp = u[1] * coe[0]
        u2_tmp = u[2]

        y1_tmp = y[1] * coe[1]
        y2_tmp = y[2] * coe[2]

        uy2_tmp = u2_tmp - y2_tmp

        uy1_tmp = u1_tmp + y1_tmp + uy2_tmp

        y[0] = u0_tmp + uy1_tmp

        for uidx in range(num_u - 1):
            u[num_u - uidx - 1] = u[num_u - uidx - 2]

        for yidx in range(num_y - 1):
            y[num_y - yidx - 1] = y[num_y - yidx - 2]

        opt[idx] = y[0]

    return opt


def decimation_IIR_filter_fix_coe(coe, ipt):
    # coe 0, 1, 2: b1, a1, a2
    assert len(coe) == 3

    coe_fixed = np.zeros(3, dtype=float)
    coe_hex = []
    _, coe_fixed[0], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[0], 8, 2, True)
    coe_hex.append(fix_point)
    _, coe_fixed[1], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[1], 8, 2, True)
    coe_hex.append(fix_point)
    _, coe_fixed[2], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[2], 8, 0, False)
    coe_hex.append(fix_point)

    num_u = 3
    num_y = 3

    u = np.zeros(num_u)
    y = np.zeros(num_y)

    opt = np.zeros(np.shape(ipt))

    for idx, value in np.ndenumerate(ipt):
        u[0] = value

        y[0] = 0

        u0_tmp = u[0]
        u1_tmp = u[1] * coe_fixed[0]
        u2_tmp = u[2]

        y1_tmp = y[1] * coe_fixed[1]
        y2_tmp = y[2] * coe_fixed[2]

        uy2_tmp = u2_tmp - y2_tmp

        uy1_tmp = u1_tmp + y1_tmp + uy2_tmp

        y[0] = u0_tmp + uy1_tmp

        for uidx in range(num_u - 1):
            u[num_u - uidx - 1] = u[num_u - uidx - 2]

        for yidx in range(num_y - 1):
            y[num_y - yidx - 1] = y[num_y - yidx - 2]

        opt[idx] = y[0]

    return coe_fixed, coe_hex, opt


def decimation_IIR_filter(coe, ipt):
    # coe 0, 1, 2: b1, a1, a2
    assert len(coe) == 3

    coe_fixed = np.zeros(3, dtype=float)
    coe_hex = []
    _, coe_fixed[0], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[0], 8, 2, True)
    coe_hex.append(fix_point)
    _, coe_fixed[1], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[1], 8, 2, True)
    coe_hex.append(fix_point)
    _, coe_fixed[2], fix_point = Quantization.Quantizer.float_to_fix_single_point(coe[2], 8, 0, False)
    coe_hex.append(fix_point)

    num_u = 3
    num_y = 3

    u = np.zeros(num_u)
    y = np.zeros(num_y)

    opt = np.zeros(np.shape(ipt))

    for idx, value in np.ndenumerate(ipt):
        _, u[0], _ = Quantization.Quantizer.float_to_fix_single_point(value, 16, 1, True)

        y[0] = 0

        _, u0_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(u[0], 16, 1, True)
        _, u1_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(u[1] * coe_fixed[0], 19, 1, True)
        _, u2_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(u[2], 16, 1, True)

        _, y1_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(y[1] * coe_fixed[1], 20, 3, True)
        _, y2_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(y[2] * coe_fixed[2], 18, 1, True)

        _, uy2_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(u2_tmp - y2_tmp, 19, 1, True)

        _, uy1_tmp, _ = Quantization.Quantizer.float_to_fix_single_point(u1_tmp + y1_tmp + uy2_tmp, 18, 1, True)

        _, y[0], _ = Quantization.Quantizer.float_to_fix_single_point(u0_tmp + uy1_tmp, 17, 1, True)

        for uidx in range(num_u - 1):
            u[num_u - uidx - 1] = u[num_u - uidx - 2]

        for yidx in range(num_y - 1):
            y[num_y - yidx - 1] = y[num_y - yidx - 2]

        opt[idx] = y[0]

    return coe_fixed, coe_hex, opt


def plot_sub(x_float, x_fixed, fig, num_row, num_col, sub_fig):
    ax = fig.add_subplot(num_row, num_col, sub_fig)
    num_point = len(x_fixed)
    if np.mod(sub_fig, 2) == 1:
        ax.set_title('Fig.' + str(sub_fig) + ' - time domain',
                     fontsize=title_font, fontproperties=font_times)
        ax.set_xlabel('n', fontsize=label_font, fontproperties=font_times)
        ax.set_ylabel('Amplitude', fontsize=label_font, fontproperties=font_times)
        n = np.arange(num_point)
        ax.plot(n, x_float, 'b-', linewidth=1, label='float')
        ax.plot(n, x_fixed, 'r--', linewidth=0.5, label='fixed')
    else:
        ax.set_title('Fig.' + str(sub_fig) + ' - frequency domain',
                     fontsize=title_font, fontproperties=font_times)
        ax.set_xlabel('freq - n', fontsize=label_font, fontproperties=font_times)
        ax.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
        n = np.arange(-num_point//2, num_point//2) / num_point
        ax.plot(n, np.fft.fftshift(x_float), 'b-', linewidth=1, label='float')
        ax.plot(n, np.fft.fftshift(x_fixed), 'r--', linewidth=0.5, label='fixed')

    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


def plot_4_cascade_IIR(ipt_float, a1, a2, b1, s, final_scal):
    fig = plt.figure(figsize=(50, 18), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    num_col = 5
    num_row = 2

    _, ipt_fixed, ipt_hex = Quantization.Quantizer.float_to_fix_1d(ipt_float, 16, 1, True)
    ipt_fixed_db = 20 * np.log10(np.abs(np.fft.fft(ipt_fixed * s[0] * WIN)))
    ipt_float_db = 20 * np.log10(np.abs(np.fft.fft(ipt_float * s[0] * WIN)))

    sub_fig = 1
    plot_sub(ipt_float * s[0], ipt_fixed * s[0], fig, num_row, num_col, sub_fig)

    sub_fig = 2
    plot_sub(ipt_float_db, ipt_fixed_db, fig, num_row, num_col, sub_fig)

    for k in range(4):
        sos = k

        dec2_num = [1, b1[sos], 1]
        dec_den = [1, -a1[sos], a2[sos]]

        coe = [b1[sos], a1[sos], a2[sos]]

        ipt_fixed = ipt_fixed * s[k]
        ipt_float = ipt_float * s[k]

        # coe_fixed, coe_hex, opt_fixed = decimation_IIR_filter(coe, ipt_fixed)
        coe_fixed, coe_hex, opt_fixed = decimation_IIR_filter_fix_coe(coe, ipt_fixed)
        opt_float = decimation_IIR_filter_float(coe, ipt_float)
        # opt = digital_filter(dec2_num, dec_den, ipt_float)

        if k == 3:
            opt_fixed = opt_fixed * final_scal
            opt_float = opt_float * final_scal

            opt_fixed_db = 20 * np.log10(np.abs(np.fft.fft(opt_fixed * WIN)))
            opt_float_db = 20 * np.log10(np.abs(np.fft.fft(opt_float * WIN)))

            sub_fig = 2 * k + 2 + 1
            plot_sub(opt_float, opt_fixed, fig, num_row, num_col, sub_fig)

            sub_fig = 2 * k + 2 + 2
            plot_sub(opt_float_db, opt_fixed_db, fig, num_row, num_col, sub_fig)
        else:
            opt_fixed_db = 20 * np.log10(np.abs(np.fft.fft(opt_fixed * WIN)))
            opt_float_db = 20 * np.log10(np.abs(np.fft.fft(opt_float * WIN)))

            sub_fig = 2 * k + 2 + 1
            plot_sub(opt_float, opt_fixed, fig, num_row, num_col, sub_fig)

            sub_fig = 2 * k + 2 + 2
            plot_sub(opt_float_db, opt_fixed_db, fig, num_row, num_col, sub_fig)

        ipt_fixed = opt_fixed
        ipt_float = opt_float

    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # fig_spec = plt.figure(figsize=(50, 18), dpi=30)
    # plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    # NUM_COL = 5
    # NUM_ROW = 2
    #

    WIN = np.zeros(1000)
    WIN[0:500] = Window.WindowModule.hanning(500, 'symmetric')
    IPT = np.zeros(1000)
    IPT[0:500] = 0.3 * np.sin(2 * np.pi * 10 * np.arange(500) / 500) + 0.01 * np.random.randn(500)
    # IPT[0:500] = 0.3 + 0.1 * np.random.randn(500)

    """##############################################################################################################"""
    """ dec 2 """
    A1 = [-0.388190791409149, -0.285741152113574, -0.138801927534664, -0.025907440076305]
    A2 = [0.057398668463065, 0.186286541864827, 0.4230987127079, 0.76740478730534]
    B1 = [1.917977679727313, 1.419457762233652, 0.897984940087441, 0.615899604416527]
    S = [1/4, 1/4, 1/4, 1/4]
    FS = [1.8779492299300875]

    X, Y, Z = Quantization.Quantizer.float_to_fix_1d(FS, 10, 2, True)
    print(X)
    print(Y)
    print(Z)
    # X, Y, Z = Quantization.Quantizer.float_to_fix_1d(A1, 8, 2, True)
    # print(X)
    # print(Y)
    # print(Z)

    plot_4_cascade_IIR(IPT, A1, A2, B1, S, FS)
    # plt.savefig('Decimation_IIR_Filter_Quantization_stable_Dec2.pdf')

    """##############################################################################################################"""
    """ dec 4 """
    A1 = [0.48243904855719, 0.694860383191498, 0.986516747573276, 1.264655494853073]
    A2 = [0.077096685422396, 0.25846165607158, 0.525852511749838, 0.826540395172634]
    B1 = [1.575900485199193, 0.038934471446067, -0.73188478016503, -0.999373895925541]

    S = [1 / 8, 1 / 4, 1 / 4, 1 / 4]
    FS = [1.4051939812328247]

    X, Y, Z = Quantization.Quantizer.float_to_fix_1d(FS, 10, 2, True)
    print(X)
    print(Y)
    print(Z)

    plot_4_cascade_IIR(IPT, A1, A2, B1, S, FS)
    # plt.savefig('Decimation_IIR_Filter_Quantization_stable_Dec4.pdf')

    """##############################################################################################################"""
    """ dec 8 """
    A1 = [1.089165879969373, 1.293469185243449, 1.535701747984973, 1.742160371287951]
    A2 = [0.307225706291077, 0.484768643825525, 0.701544980890458, 0.900155860753369]
    B1 = [0.672052035914994, -1.204826510675321, -1.601088682038418, -1.705029177707256]

    S = [1 / 16, 1 / 8, 1 / 4, 1 / 4]
    FS = [1.119204868923377]

    X, Y, Z = Quantization.Quantizer.float_to_fix_1d(FS, 10, 2, True)
    print(X)
    print(Y)
    print(Z)

    plot_4_cascade_IIR(IPT, A1, A2, B1, S, FS)
    # plt.savefig('Decimation_IIR_Filter_Quantization_stable_Dec8.pdf')

    plt.show()
