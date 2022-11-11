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


def create_pll_phase(cfg):
    fs = cfg['fs']
    ts = 1 / fs
    b = cfg['B']
    pri = cfg['PRI']
    alpha = b / pri
    num_point = int(fs * PRI)
    t_axis = np.arange(num_point) * ts
    ctrl = cfg['ctrl']
    tau = cfg['tau']

    """ Ideal phase """
    if ctrl == 'positive':
        ideal_phase = 2 * np.pi * ((alpha * (t_axis ** 2) / 2) - b / 2 * t_axis)
        # up sweep --> Nyquist sampling
        ideal_phase_delay = 2 * np.pi * ((alpha * ((t_axis - tau) ** 2) / 2) - b / 2 * (t_axis - tau))
        # up sweep --> Nyquist sampling
    else:
        ideal_phase = 2 * np.pi * (b / 2 * t_axis - alpha * (t_axis ** 2) / 2)  # down sweep --> Nyquist sampling
        ideal_phase_delay = 2 * np.pi * (b / 2 * (t_axis - tau) - alpha * ((t_axis - tau) ** 2) / 2)

    return ideal_phase, ideal_phase_delay


def create_phase_error(cfg):
    """ create phase error, including real phase error and delay phase error """
    fs = cfg['fs']
    ts = 1 / fs
    pri = cfg['PRI']
    num_point = int(fs * pri)
    t_axis = np.arange(num_point) * ts
    tau = cfg['tau']
    amp = cfg['amp_error']
    f_error = cfg['f_error']
    assert len(amp) == len(f_error)
    num_error = len(f_error)
    ini_error = cfg['ini_error']
    assert len(ini_error) == len(f_error)

    phase_error_ideal = np.zeros(num_point, dtype=float)
    phase_error_delay = np.zeros(num_point, dtype=float)
    for k in range(num_error):
        phase_error_ideal = phase_error_ideal + amp[k] * np.sin(2 * np.pi * f_error[k] * MHz * t_axis +
                                                                2 * np.pi * ini_error[k])
        phase_error_delay = phase_error_delay + amp[k] * np.sin(2 * np.pi * f_error[k] * MHz * (t_axis - tau) +
                                                                2 * np.pi * ini_error[k])

    return phase_error_ideal, phase_error_delay


def filter_LPF(ipt):
    num_filter = 127
    wc = 0.5 * np.pi
    n = np.arange(-int((num_filter - 1) // 2), int((num_filter + 1) // 2))
    h = np.sin(wc * n) / np.pi / n
    h[int((num_filter - 1) // 2)] = wc / np.pi

    win = hamming(num_filter)
    h_lpf = h * win

    b = h_lpf
    a = [1]
    opt = FilterModule.FilterModule.digital_filter(b, a, ipt)
    return opt


def phase_estimation_module_1(s_i, s_q, ideal_phase_unwrap):
    """ phase estimator, phase unwrapping, relative phase """

    """ Phase estimator """
    # s_complex = s_I + 1j * s_Q
    # practical_phase_python = np.angle(s_complex)
    real_phase_est = PhaseModule.PhaseModule.phase_estimator(s_i, s_q)
    # print(practical_phase_python - practical_phase)

    """ phase unwrapping """
    real_phase_est_unwrap = PhaseModule.PhaseModule.phase_unwrapping(real_phase_est)

    """ phase de-ramping"""
    phase_error_unwrap_est = real_phase_est_unwrap - ideal_phase_unwrap

    """ relative phase """
    relative_phase_error_unwrap_est = phase_error_unwrap_est - phase_error_unwrap_est[0]

    """ phase wrapping """
    relative_phase_error_unwrap_est_wrap = PhaseModule.PhaseModule.phase_wrapping(relative_phase_error_unwrap_est)

    return real_phase_est, real_phase_est_unwrap, phase_error_unwrap_est, \
           relative_phase_error_unwrap_est, relative_phase_error_unwrap_est_wrap


def phase_estimation_module_2(s_i, s_q, ideal_phase_est):
    """ phase estimator, phase unwrapping, relative phase """

    """ Phase estimator """
    # s_complex = s_I + 1j * s_Q
    # practical_phase_python = np.angle(s_complex)
    real_phase_est = PhaseModule.PhaseModule.phase_estimator(s_i, s_q)
    # print(practical_phase_python - practical_phase)

    """ phase de-ramping """
    phase_error_est = real_phase_est - ideal_phase_est
    phase_error_est = PhaseModule.PhaseModule.phase_wrapping(phase_error_est)

    """ phase unwrapping """
    phase_error_est_unwrap = PhaseModule.PhaseModule.phase_unwrapping(phase_error_est)

    """ relative phase """
    relative_phase_error_est_unwrap = phase_error_est_unwrap - phase_error_est_unwrap[0]

    """ phase filtering """
    relative_phase_error_est_unwrap_filter = filter_LPF(relative_phase_error_est_unwrap)

    """ phase wrapping """
    relative_phase_error_est_unwrap_filter_wrap = PhaseModule.PhaseModule.phase_wrapping(
        relative_phase_error_est_unwrap_filter)

    return real_phase_est, phase_error_est, phase_error_est_unwrap, relative_phase_error_est_unwrap, \
           relative_phase_error_est_unwrap_filter, relative_phase_error_est_unwrap_filter_wrap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """ =================================================================================================== """
    Config = {
        'case': 'no_delay',
        'fs': 100 * MHz,
        'B': 400 * MHz,
        'PRI': 10.24 * us,
        'phi0': np.pi / 2,
        'ctrl': 'negative',
        'tau': 0 * ns,
        'f_error': [15, 7.5, 5],  # MHz
        'amp_error': [1, 0.8, 1],
        'ini_error': [0.1, 0.2, 0.3],
    }

    CTRL = Config['ctrl']
    FS = Config['fs']
    TS = 1 / FS
    PRI = Config['PRI']
    NUM_POINT = int(FS * PRI)
    T_AXIS = np.arange(NUM_POINT) * TS

    NUM_TAP = 127
    WC = 0.5 * np.pi
    N_AXIS = np.arange(-int((NUM_TAP - 1) // 2), int((NUM_TAP + 1) // 2))
    HT = np.sin(WC * N_AXIS) / np.pi / N_AXIS
    HT[int((NUM_TAP - 1) // 2)] = WC / np.pi

    WIN = hamming(NUM_TAP)
    HF = HT * WIN

    b = HF
    a = [1]
    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.linspace(-0.5, 0.5, 1000)))
    freq_response_dB = 20 * np.log10(np.abs(freq_response))
    freq_response_dB = freq_response_dB - np.amax(freq_response_dB)

    """ =================================================================================================== """
    """ ideal_phase, including Nyquist and sub-Nyquist sampling """
    IDEAL_PHASE, IDEAL_PHASE_DELAY = create_pll_phase(Config)  # IDEAL_PHASE， IDEAL_PHASE_DELAY: 2 * pi

    IDEAL_FREQ = (IDEAL_PHASE[1:] - IDEAL_PHASE[0:-1]) / TS / 2 / np.pi  # freq --> Nyquist sampling

    """ Ideal IQ Demodulation """
    S_IDEAL = np.exp(1j * IDEAL_PHASE)
    S_IDEAL_I = np.real(S_IDEAL)
    S_IDEAL_Q = np.imag(S_IDEAL)

    """ sub-Nyquist sampling - wrap phase """
    # IDEAL_PHASE_EST1 = PhaseModule.PhaseModule.phase_wrapping(IDEAL_PHASE/2/np.pi)
    IDEAL_PHASE_EST = PhaseModule.PhaseModule.phase_estimator(S_IDEAL_I, S_IDEAL_Q)
    # print(IDEAL_PHASE_EST1 - IDEAL_PHASE_EST)

    """ sub-Nyquist sampling - unwrap phase """
    IDEAL_PHASE_UNWRAP = PhaseModule.PhaseModule.phase_unwrapping(IDEAL_PHASE_EST)
    IDEAL_FREQ_UNWRAP = (IDEAL_PHASE_UNWRAP[1:] - IDEAL_PHASE_UNWRAP[0:-1]) / TS  # sub-Nyquist sampling

    """ =================================================================================================== """
    """ phase error """
    IDEAL_PHASE_ERROR, IDEAL_PHASE_ERROR_DELAY = create_phase_error(Config)
    # IDEAL_PHASE_ERROR, IDEAL_PHASE_ERROR_DELAY: 2 * pi

    IDEAL_RELATIVE_PHASE_ERROR = IDEAL_PHASE_ERROR - IDEAL_PHASE_ERROR[0]
    IDEAL_RELATIVE_PHASE_ERROR = IDEAL_RELATIVE_PHASE_ERROR / 2 / np.pi

    IDEAL_RELATIVE_PHASE_ERROR_WRAP = PhaseModule.PhaseModule.phase_wrapping(IDEAL_RELATIVE_PHASE_ERROR)
    IDEAL_RELATIVE_PHASE_ERROR_WRAP_UNWRAP = PhaseModule.PhaseModule.phase_unwrapping(IDEAL_RELATIVE_PHASE_ERROR_WRAP)
    IDEAL_FREQ_ERROR_WRAP_UNWRAP = (IDEAL_RELATIVE_PHASE_ERROR_WRAP_UNWRAP[1:] - IDEAL_RELATIVE_PHASE_ERROR_WRAP_UNWRAP[
                                                                                 0:-1]) / TS  # sub-Nyquist sampling

    """ =================================================================================================== """
    """ create IQ demodulation and LPF -- real phase"""
    REAL_PHASE = IDEAL_PHASE_DELAY + IDEAL_PHASE_ERROR_DELAY
    REAL_PHASE_WRAP = PhaseModule.PhaseModule.phase_wrapping(REAL_PHASE / 2 / np.pi)
    REAL_PHASE_WRAP_UNWRAP = PhaseModule.PhaseModule.phase_unwrapping(REAL_PHASE_WRAP)

    S_REAL = np.exp(1j * REAL_PHASE)
    S_REAL_I = np.real(S_REAL)
    S_REAL_Q = np.imag(S_REAL)

    """ sub-Nyquist sampling - wrap phase """
    REAL_PHASE_EST = PhaseModule.PhaseModule.phase_estimator(S_REAL_I, S_REAL_Q)

    """ sub-Nyquist sampling - unwrap phase """
    REAL_PHASE_UNWRAP = PhaseModule.PhaseModule.phase_unwrapping(REAL_PHASE_EST)
    REAL_FREQ_UNWRAP = (REAL_PHASE_UNWRAP[1:] - REAL_PHASE_UNWRAP[0:-1]) / TS  # sub-Nyquist sampling

    """ =================================================================================================== """
    """ Phase error estimation 2 """
    REAL_PHASE_EST_2, PHASE_ERROR_EST, PHASE_ERROR_EST_UNWRAP, \
    RELATIVE_PHASE_ERROR_EST_UNWRAP, RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER, \
    RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER_WRAP = phase_estimation_module_2(S_REAL_I, S_REAL_Q, IDEAL_PHASE_EST)

    """ Phase error estimation 1 """
    REAL_PHASE_EST_1, REAL_PHASE_EST_UNWRAP, PHASE_ERROR_UNWRAP_EST, \
    RELATIVE_PHASE_ERROR_UNWRAP_EST, RELATIVE_PHASE_ERROR_UNWRAP_EST_WRAP \
        = phase_estimation_module_1(S_REAL_I, S_REAL_Q, IDEAL_PHASE_UNWRAP)

    REAL_FREQ_EST_UNWRAP = (REAL_PHASE_EST_UNWRAP[1:] - REAL_PHASE_EST_UNWRAP[0:-1]) / TS
    """ =================================================================================================== """
    """ Figure """

    fig_spec = plt.figure(figsize=(45, 18), dpi=25)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    num_col = 4
    num_row = 3
    ax_spec = fig_spec.add_subplot(num_row, num_col, 1)
    ax_spec.set_title('Fig.1 - Ideal Phase of PLL after IQ Demodulation',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS / us, IDEAL_PHASE / 2 / np.pi, 'k', linewidth=3, label='ideal phase')
    ax_spec.plot(T_AXIS / us, IDEAL_PHASE_UNWRAP, 'b--', linewidth=2, label='unwrap ideal phase')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 2)
    ax_spec.set_title('Fig.2 - Sweeping Frequency of PLL after IQ Demodulation',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Freq - MHz', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[1:] / us, IDEAL_FREQ / MHz, 'k', linewidth=3, label='ideal freq')
    ax_spec.plot(T_AXIS[1:] / us, IDEAL_FREQ_UNWRAP / MHz, 'b--', linewidth=2, label='ideal alias freq')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 3)
    ax_spec.set_title('Fig.3 - Un-Wrapping ideal and real phase',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS / us, IDEAL_PHASE_UNWRAP, 'b', linewidth=5, label='unwrapped ideal phase')
    ax_spec.plot(T_AXIS / us, REAL_PHASE_EST_UNWRAP, 'orange', linewidth=2, label='unwrapped real phase')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 4)
    ax_spec.set_title('Fig.4 - Estimated Frequency from Un-Wrapping ideal and real phase',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Freq', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[1:] / us, REAL_FREQ_EST_UNWRAP / MHz, 'orange', linewidth=1, label='real alias freq')
    ax_spec.plot(T_AXIS[1:] / us, IDEAL_FREQ_UNWRAP / MHz, 'b', linewidth=3, label='ideal alias freq')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 5)
    ax_spec.set_title('Fig.5 - Phase Estimator',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:256] / us, IDEAL_PHASE_EST[:256], 'kx-', linewidth=2, label='estimated ideal phase')
    ax_spec.plot(T_AXIS[:256] / us, REAL_PHASE_EST_2[:256], 'ro--', linewidth=1, label='estimated real phase')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 6)
    ax_spec.set_title('Fig.6 - Phase De-Ramping',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:256] / us, PHASE_ERROR_EST[:256], 'go--', label='estimated phase error')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 7)
    ax_spec.set_title('Fig.7 - Phase Un-Wrapping',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:256] / us, PHASE_ERROR_EST_UNWRAP[:256], 'go--', linewidth=1,
                 label='unwrapped estimated phase error')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    f_axis_shift = np.arange(-int(1000 / 2), int(1000 / 2)) / 1000 * FS
    ax_spec = fig_spec.add_subplot(num_row, num_col, 8)
    ax_spec.set_title('Fig.8 - Frequency Response of Phase-Domain Filter',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('Freq - MHz', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(f_axis_shift / MHz, freq_response_dB, 'k', linewidth=2, label='LPF frequency response')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 9)
    ax_spec.set_title('Fig.9 - Phase-Domain Filter Filtering',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:256] / us, RELATIVE_PHASE_ERROR_EST_UNWRAP[:256], 'gx-', linewidth=2,
                 label='input - estimated relative phase error')
    ax_spec.plot(T_AXIS[:256] / us, RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER[:256], 'ro--', linewidth=1, label='output')
    plt.legend(fontsize=22, loc='lower right')
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 10)
    ax_spec.set_title('Fig.10 - Phase Wrapping',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:256] / us, IDEAL_RELATIVE_PHASE_ERROR_WRAP[:256], 'kx-', linewidth=5,
                 label='relative phase error')
    ax_spec.plot(T_AXIS[:256] / us, RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER_WRAP[63:63+256], 'ro--', linewidth=2,
                 label='estimated relative phase error')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 11)
    ax_spec.set_title('Fig.11 - Phase Wrapping',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS / us, IDEAL_RELATIVE_PHASE_ERROR_WRAP, 'k', linewidth=2,
                 label='relative phase error')
    ax_spec.plot(T_AXIS[:-63] / us, RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER_WRAP[63:], 'r--', linewidth=0.5,
                 label='estimated relative phase error')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 12)
    ax_spec.set_title('Fig.12 - Estimation error of Phase error',
                      fontsize=25, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=22, fontproperties=font_times)
    ax_spec.set_ylabel('Phase', fontsize=22, fontproperties=font_times)
    ax_spec.plot(T_AXIS[:-63] / us, IDEAL_RELATIVE_PHASE_ERROR_WRAP[:-63] -
                 RELATIVE_PHASE_ERROR_EST_UNWRAP_FILTER_WRAP[63:], 'k', linewidth=2,
                 label='estimation error of phase error')
    plt.legend(fontsize=22)
    plt.tick_params(labelsize=22)
    labels = ax_spec.get_xticklabels() + ax_spec.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.savefig('Phase_estimation_center_chirp_' + CTRL + '_0_1.pdf')

    plt.show()