import numpy as np
import os
import sys
import time
current_time = time.strftime('%Y%m%d_%H%M%S')
current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, '..', 'FilterDesign')
sys.path.append(src_path)  # add module path


import FilterModule


class ADCModule:
    name = 'ADC Module'

    def __init__(self):
        pass

    @staticmethod
    def one_bit_quantizer(y):
        num_point = len(y)
        v = np.zeros(num_point, dtype=int)
        for id in range(num_point):
            v[id] = ADCModule.one_bit_quantizer_one_point(y[id])
        return v

    @staticmethod
    def one_bit_quantizer_one_point(y):
        v = (1 if y > 0 else -1)
        return v

    """ Sigma Delta Modulator"""
    @staticmethod
    def first_order(u, delta_range):
        # x = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            # x[id] = u[id-1] - v[id-1]
            # y[id] = x[id] + y[id-1]
            y[id] = u[id - 1] - v[id - 1] + y[id - 1]
            v[id] = ADCModule.quantizer_add_noise(delta_range, y[id])

        return u, y, v    # u - input, y - quantizer input, v - quantizer output

    @staticmethod
    def first_order_one_bit(u):
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=int)

        for id in range(1, len(u)):
            y[id] = u[id - 1] - v[id - 1] + y[id - 1]
            v[id] = ADCModule.one_bit_quantizer_one_point(y[id])

        return u, y, v  # u - input, y - quantizer input, v - quantizer output

    @staticmethod
    def second_order(u, delta_range):
        x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        x2 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            x0[id] = u[id - 1] - v[id - 1]
            x1[id] = x0[id] + x1[id - 1]
            x2[id] = x1[id] - v[id - 1]
            y[id] = x2[id] + y[id - 1]
            v[id] = ADCModule.quantizer_add_noise(delta_range, y[id])

        return u, x0, x1, x2, y, v

    @staticmethod
    def second_order_one_bit(u):
        x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        x2 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            x0[id] = u[id - 1] - v[id - 1]
            x1[id] = x0[id] + x1[id - 1]
            x2[id] = x1[id] - v[id - 1]
            y[id] = x2[id] + y[id - 1]
            v[id] = ADCModule.one_bit_quantizer_one_point(y[id])

        return u, x0, x1, x2, y, v

    @staticmethod
    def second_order_mash(u):
        x1 = np.zeros(len(u), dtype=float)
        x2 = np.zeros(len(u), dtype=float)
        y1 = np.zeros(len(u), dtype=float)
        y2 = np.zeros(len(u), dtype=float)
        v1 = np.zeros(len(u), dtype=float)
        v2 = np.zeros(len(u), dtype=float)
        q1 = np.zeros(len(u), dtype=float)

        diff_num = [1, -1]
        diff_den = [1]

        for id in range(1, len(u)):
            x1[id] = u[id] - v1[id - 1]
            y1[id] = x1[id] + y1[id - 1]
            v1[id] = ADCModule.one_bit_quantizer_one_point(y1[id])

            q1[id] = y1[id - 1] - v1[id - 1]

            x2[id] = q1[id] - v2[id - 1]
            y2[id] = x2[id] + y2[id - 1]
            v2[id] = ADCModule.one_bit_quantizer_one_point(y2[id])

        v2_filter1 = FilterModule.FilterModule.digital_filter(diff_num, diff_den, v2)
        v = v1 + v2_filter1

        return v

    @staticmethod
    def third_order(u, delta_range):
        x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            x0[id] = u[id] - v[id - 1] + x0[id - 1]
            x1[id] = x0[id] - v[id - 1] + x1[id - 1]
            y[id] = x1[id] - v[id - 1] + y[id - 1]

            v[id] = ADCModule.quantizer_add_noise(delta_range, y[id])

        return u, x0, x1, y, v

    @staticmethod
    def third_order_one_bit(u):
        x0 = np.zeros(len(u), dtype=float)
        x1 = np.zeros(len(u), dtype=float)
        y = np.zeros(len(u), dtype=float)
        v = np.zeros(len(u), dtype=float)

        for id in range(1, len(u)):
            x0[id] = u[id] - v[id - 1] + x0[id - 1]
            x1[id] = x0[id] - v[id - 1] + x1[id - 1]
            y[id] = x1[id] - v[id - 1] + y[id - 1]

            v[id] = ADCModule.one_bit_quantizer_one_point(y[id])

        return u, x0, x1, y, v

    @staticmethod
    def third_order_mash(u):
        x1 = np.zeros(len(u), dtype=float)
        x2 = np.zeros(len(u), dtype=float)
        x3 = np.zeros(len(u), dtype=float)
        y1 = np.zeros(len(u), dtype=float)
        y2 = np.zeros(len(u), dtype=float)
        y3 = np.zeros(len(u), dtype=float)
        v1 = np.zeros(len(u), dtype=float)
        v2 = np.zeros(len(u), dtype=float)
        v3 = np.zeros(len(u), dtype=float)
        q1 = np.zeros(len(u), dtype=float)
        q2 = np.zeros(len(u), dtype=float)

        diff_num = [1, -2, 1]
        diff_den = [1]

        for id in range(1, len(u)):
            x1[id] = u[id] - v1[id - 1]
            y1[id] = x1[id] + y1[id - 1]
            v1[id] = ADCModule.one_bit_quantizer_one_point(y1[id])
            # v1[id] = ADCModule.quantizer_add_noise(delta_range, y1[id])

            q1[id] = y1[id-1] - v1[id - 1]

            x2[id] = q1[id] - v2[id-1]
            y2[id] = x2[id] + y2[id-1]
            v2[id] = ADCModule.one_bit_quantizer_one_point(y2[id])
            # v2[id] = ADCModule.quantizer_add_noise(delta_range, y2[id])

            q2[id] = y2[id-1] - v2[id-1]

            x3[id] = q2[id] - v3[id-1]
            y3[id] = x3[id] + y3[id]
            v3[id] = ADCModule.one_bit_quantizer_one_point(y3[id])
            # v3[id] = ADCModule.quantizer_add_noise(delta_range, y3[id])

        v2_filter1 = FilterModule.FilterModule.digital_filter(diff_num, diff_den, v2)
        v3_filter1 = FilterModule.FilterModule.digital_filter(diff_num, diff_den, v3)
        v = v1 + v2_filter1 + v3_filter1

        # v3_filter1_1 = FilterModule.FilterModule.digital_filter([1, -1], diff_den, v3)
        # v3_filter1_2 = FilterModule.FilterModule.digital_filter([1, -1], diff_den, v3_filter1_1)

        return v

    """ Quantizer """
    @staticmethod
    def quantizer_mid_rise(half_spread, num_bit, y):
        num_step = 2 ** num_bit
        step = 2 * half_spread / num_step
        assert (y <= half_spread + step/2) and (y >= - (half_spread + step/2))

        e = abs((y/step) - int(y/step))
        if e > 0:
            if y >= 0:
                step_id = int(y/step) + 1
            else:
                step_id = int(y/step) - 1
        else:
            if y >= 0:
                step_id = int(y / step)
            else:
                step_id = int(y / step)

        if abs(step_id) > (num_step//2):
            step_id = (num_step//2) * (step_id / abs(step_id))

        v = step_id * step

        return v

    @staticmethod
    def quantizer_mid_tread(half_spread, num_bit, y):
        assert num_bit > 1
        num_step = 2 ** num_bit
        step = 2 * half_spread / num_step
        assert (y <= half_spread + step/2) and (y >= - (half_spread + step/2))

        e = abs((y/step) - int(y/step))
        if e > 0.5:
            if y >= 0:
                step_id = int(y / step) + 1
            else:
                step_id = int(y / step) - 1
        else:
            if y >= 0:
                step_id = int(y / step)
            else:
                step_id = int(y / step)

        if abs(step_id) > (num_step//2 - 1):
            step_id = (num_step//2-1) * (step_id / abs(step_id))

        v = step_id * step

        return v

    @staticmethod
    def quantizer_add_noise(delta_range, y):
        step = delta_range

        v = y + step * np.random.uniform(-1, 1, np.shape(y)) / 2

        return v


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    font_times = fm.FontProperties(family='Times New Roman', stretch=0)
    fig_spec = plt.figure(figsize=(45, 24), dpi=30)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    num_col = 3
    num_row = 4

    title_font = 25
    label_font = 22
    legend_font = 12
    tick_font = 20

    """ Sigma Delta Modulator Test"""
    HALF_SPREAD = 1
    NUM_BIT = 1
    DELTA_RANGE = 2 * HALF_SPREAD / NUM_BIT
    N_SAMPLE = 1024
    FC = 30
    FS = 100
    OSR = 20
    NUM_OSR = N_SAMPLE * OSR
    FOSR = FS * OSR
    NUM_SPEC = N_SAMPLE * OSR
    FFT_SIZE_FACTOR = 4

    T_AXIS_OSR = np.arange(NUM_OSR) / FOSR
    T_AXIS = T_AXIS_OSR[::OSR]

    OSR_FFT_SIZE = NUM_OSR * FFT_SIZE_FACTOR

    F_AXIS_SHIFT_OSR = np.arange(-int(NUM_SPEC / 2), int(NUM_SPEC / 2)) / NUM_SPEC * FOSR
    F_AXIS_SHIFT_OSR_FFT = np.arange(-int(OSR_FFT_SIZE / 2), int(OSR_FFT_SIZE / 2)) / OSR_FFT_SIZE * FOSR
    """##############################################################################################################"""
    amp = 0.7
    # input signal
    S_ANALOG = amp * np.sin(2 * FC * np.pi * T_AXIS_OSR) + 0.001 * (np.random.rand(NUM_OSR) - 0.5)
    S_ANALOG_F = np.fft.fft(S_ANALOG, n=OSR_FFT_SIZE)
    S_ANALOG_F_dB = 20 * np.log10(np.abs(S_ANALOG_F))

    """ One-bit Quantization """
    S_ANALOG_ONE_BIT = ADCModule.one_bit_quantizer(S_ANALOG)
    S_ANALOG_ONE_BIT_F = np.fft.fft(S_ANALOG_ONE_BIT, n=OSR_FFT_SIZE)
    S_ANALOG_ONE_BIT_F_dB = 20 * np.log10(np.abs(S_ANALOG_ONE_BIT_F))

    """ White noise model for One-bit Quantization """
    S_ANALOG_WHITE_OBIT = ADCModule.quantizer_add_noise(DELTA_RANGE, S_ANALOG)
    S_ANALOG_WHITE_OBIT_F = np.fft.fft(S_ANALOG_WHITE_OBIT, n=OSR_FFT_SIZE)
    S_ANALOG_WHITE_OBIT_F_dB = 20 * np.log10(np.abs(S_ANALOG_WHITE_OBIT_F))

    SUB_FIG = 1
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Quantized Signal ',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('t', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Amplitude', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG[:200],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG_WHITE_OBIT[:200],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG_ONE_BIT[:200],
                 'r--', label='one-bit')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 2
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of Quantized Signal',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], S_ANALOG_WHITE_OBIT_F_dB[0: OSR_FFT_SIZE // 2],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], S_ANALOG_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                 'r--', label='one-bit')

    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    """ 1st-order SDM """
    NUM_1_SDM = [1, -1]
    DEN_1_SDM = [1]
    FREQ_RESPONSE_1_SDM = FilterModule.FilterModule.digital_bode(NUM_1_SDM, DEN_1_SDM,
                                                                 np.exp(1j * 2 * np.pi *
                                                                        np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_1_SDM_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_1_SDM))

    """ 1st-order SDM """
    NUM_2_SDM = [1, -2, 1]
    DEN_2_SDM = [1]
    FREQ_RESPONSE_2_SDM = FilterModule.FilterModule.digital_bode(NUM_2_SDM, DEN_2_SDM,
                                                                 np.exp(1j * 2 * np.pi *
                                                                        np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_2_SDM_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_2_SDM))

    NUM_3_SDM = [1, -3, 3, -1]
    DEN_3_SDM = [1]
    FREQ_RESPONSE_3_SDM = FilterModule.FilterModule.digital_bode(NUM_3_SDM, DEN_3_SDM,
                                                                 np.exp(1j * 2 * np.pi *
                                                                        np.linspace(-0.5, 0.5, NUM_SPEC)))
    FREQ_RESPONSE_3_SDM_dB = 20 * np.log10(np.abs(FREQ_RESPONSE_3_SDM))

    SUB_FIG = 3
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Frequency Response of NTF',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR[NUM_SPEC//2:], FREQ_RESPONSE_1_SDM_dB[NUM_SPEC//2:],
                     'b-', linewidth=2, label='1st-order SDM')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR[NUM_SPEC // 2:], FREQ_RESPONSE_2_SDM_dB[NUM_SPEC // 2:],
                     'r-', linewidth=2, label='2nd-order SDM')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR[NUM_SPEC // 2:], FREQ_RESPONSE_3_SDM_dB[NUM_SPEC // 2:],
                     'g-', linewidth=2, label='3rd-order SDM')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    plt.xlim(1, FOSR//2)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    _, _, V_1st_WHITE_ONE_BIT = ADCModule.first_order(S_ANALOG, DELTA_RANGE)
    _, _, V_1st_ONE_BIT = ADCModule.first_order_one_bit(S_ANALOG)

    """ SDM Out """
    SUB_FIG = 4
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Output of 1st-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('t', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Amplitude', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG[:200],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(T_AXIS_OSR[:200], V_1st_WHITE_ONE_BIT[:200],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(T_AXIS_OSR[:200], V_1st_ONE_BIT[:200],
                 'r--', label='one-bit')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """ One-bit Quantization """
    V_1st_ONE_BIT_F = np.fft.fft(V_1st_ONE_BIT, n=OSR_FFT_SIZE)
    V_1st_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_1st_ONE_BIT_F))

    """ White noise model for One-bit Quantization """
    V_1st_WHITE_ONE_BIT_F = np.fft.fft(V_1st_WHITE_ONE_BIT, n=OSR_FFT_SIZE)
    V_1st_WHITE_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_1st_WHITE_ONE_BIT_F))

    SUB_FIG = 5
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 1st-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_1st_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_1st_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'r--', label='one-bit')

    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 6
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 1st-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'k-', linewidth=4, label='analog')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], V_1st_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                     'b.-', label='white noise for one-bit')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_1st_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'r--', label='one-bit')
    plt.xlim(1, FOSR // 2)
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    _, _, _, _, _, V_2nd_WHITE_ONE_BIT = ADCModule.second_order(S_ANALOG, DELTA_RANGE)
    V_2nd_ONE_BIT = ADCModule.second_order_mash(S_ANALOG)

    """ SDM Out """
    SUB_FIG = 7
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Output of 2nd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('t', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Amplitude', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG[:200],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(T_AXIS_OSR[:200], V_2nd_WHITE_ONE_BIT[:200],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(T_AXIS_OSR[:200], V_2nd_ONE_BIT[:200],
                 'r--', label='one-bit')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """ One-bit Quantization """
    V_2nd_ONE_BIT_F = np.fft.fft(V_2nd_ONE_BIT, n=OSR_FFT_SIZE)
    V_2nd_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_2nd_ONE_BIT_F))

    """ White noise model for One-bit Quantization """
    V_2nd_WHITE_ONE_BIT_F = np.fft.fft(V_2nd_WHITE_ONE_BIT, n=OSR_FFT_SIZE)
    V_2nd_WHITE_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_2nd_WHITE_ONE_BIT_F))

    SUB_FIG = 8
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 2nd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], V_2nd_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_2nd_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'r--', label='one-bit')

    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 9
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 2nd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'k-', linewidth=4, label='analog')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], V_2nd_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                     'b.-', label='white noise for one-bit')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_2nd_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'r--', label='one-bit')
    plt.xlim(1, FOSR // 2)
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """##############################################################################################################"""
    _, _, _, _, V_3rd_WHITE_ONE_BIT = ADCModule.third_order(S_ANALOG, DELTA_RANGE)
    V_3rd_ONE_BIT = ADCModule.third_order_mash(S_ANALOG)

    """ SDM Out """
    SUB_FIG = 10
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Output of 3rd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('t', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Amplitude', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(T_AXIS_OSR[:200], S_ANALOG[:200],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(T_AXIS_OSR[:200], V_3rd_WHITE_ONE_BIT[:200],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(T_AXIS_OSR[:200], V_3rd_ONE_BIT[:200],
                 'r--', label='one-bit')
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """ One-bit Quantization """
    V_3rd_ONE_BIT_F = np.fft.fft(V_3rd_ONE_BIT, n=OSR_FFT_SIZE)
    V_3rd_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_3rd_ONE_BIT_F))

    """ White noise model for One-bit Quantization """
    V_3rd_WHITE_ONE_BIT_F = np.fft.fft(V_3rd_WHITE_ONE_BIT, n=OSR_FFT_SIZE)
    V_3rd_WHITE_ONE_BIT_F_dB = 20 * np.log10(np.abs(V_3rd_WHITE_ONE_BIT_F))

    SUB_FIG = 11
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 3rd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'k-', linewidth=4, label='analog')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], V_3rd_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                 'b.-', label='white noise for one-bit')
    ax_spec.plot(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_3rd_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                 'r--', label='one-bit')

    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    SUB_FIG = 12
    ax_spec = fig_spec.add_subplot(num_row, num_col, SUB_FIG)
    ax_spec.set_title('Fig.' + str(SUB_FIG) + ' - Spectrum of 3rd-Order Sigma-Delta ADC',
                      fontsize=title_font, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=label_font, fontproperties=font_times)
    ax_spec.set_ylabel('Magnitude - dB', fontsize=label_font, fontproperties=font_times)
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], S_ANALOG_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'k-', linewidth=4, label='analog')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2:], V_3rd_WHITE_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2],
                     'b.-', label='white noise for one-bit')
    ax_spec.semilogx(F_AXIS_SHIFT_OSR_FFT[OSR_FFT_SIZE // 2::8], V_3rd_ONE_BIT_F_dB[0: OSR_FFT_SIZE // 2:8],
                     'r--', label='one-bit')
    plt.xlim(1, FOSR // 2)
    plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.savefig('sigma_delta_adc_'+current_time+'.pdf')
    plt.show()


