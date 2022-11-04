import matplotlib.font_manager as fm
import os
import sys
import numpy as np

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..', '..', 'src', 'PhaseModule'))
import PhaseModule

GHz = 1e9
MHz = 1e6
us = 1e-6
ns = 1e-9


def create_ideal_phase(cfg):
    fs = cfg['fs']
    ts = 1/fs
    B = cfg['B']
    PRI = cfg['PRI']
    alpha = B / PRI
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']


    """ Ideal phase """
    if ctrl == 'positive':
        ideal_phase = 2 * np.pi * (alpha * (t_axis ** 2) / 2)  # up sweep --> Nyquist sampling
    elif ctrl == 'negative':
        ideal_phase = 2 * np.pi * (B * t_axis - alpha * (t_axis ** 2) / 2)  # down sweep --> Nyquist sampling
    else:
        assert (ctrl == 'positive') or (ctrl == 'negative')

    ideal_freq = (ideal_phase[1:] - ideal_phase[0:-1]) / ts / 2 / np.pi  # freq --> Nyquist sampling

    s_ideal = np.exp(1j * ideal_phase)
    s_i = np.real(s_ideal)
    s_q = np.imag(s_ideal)

    """ sub-Nyquist sampling - wrap phase """
    ideal_phase_wrap = PhaseModule.PhaseModule.phase_estimator(s_i, s_q)
    # ideal_phase_wrap = np.angle(s_ideal)

    """ sub-Nyquist sampling - unwrap phase """
    ideal_phase_unwrap = PhaseModule.PhaseModule.phase_unwrapping(ideal_phase_wrap)
    ideal_freq_unwrap = (ideal_phase_unwrap[1:] - ideal_phase_unwrap[0:-1]) / ts / 2 / np.pi  # sub-Nyquist sampling

    """ Figure """
    """ ideal Phase """
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Phase of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Unwrapping Phase', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, ideal_phase, label='ideal phase - Nyquist')
    ax_phase.plot(t_axis / us, ideal_phase_unwrap, label='ideal phase - sub-Nyquist sampling')
    # ax_phase.plot(t_axis / us, practical_phase_unwrap, label='estimated phase - unwrap')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_Sweeping_phase.pdf')

    """ ideal Frequency """
    fig_freq = plt.figure(figsize=(7, 5), dpi=100)
    ax_freq = fig_freq.add_subplot()
    ax_freq.set_title('Sweeping Frequency of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_freq.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_freq.set_ylabel('Freq', fontsize=10, fontproperties=font_times)
    ax_freq.plot(t_axis[1:] / us, ideal_freq / MHz, label='ideal freq - Nyquist')
    ax_freq.plot(t_axis[1:] / us, ideal_freq_unwrap / MHz, label='ideal freq - sub-Nyquist sampling')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_Sweeping_Freq.pdf')
    return ideal_phase, ideal_freq, ideal_phase_unwrap, ideal_freq_unwrap


def create_phase_error(cfg):
    fs = cfg['fs']
    ts = 1 / fs
    B = cfg['B']
    PRI = cfg['PRI']
    alpha = B / PRI
    phi0 = cfg['phi0']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']
    tau = cfg['tau']
    amp = cfg['amp_error']
    f_error = cfg['f_error']
    assert len(amp) == len(f_error)
    num_error = len(f_error)

    phase_error = np.zeros(N, dtype=float)
    for k in range(num_error):
        phase_error = phase_error + amp[k] * np.cos(2 * np.pi * f_error[k] * MHz * (t_axis - tau))

    return phase_error


def create_real_signal(cfg):
    fs = cfg['fs']
    ts = 1 / fs
    B = cfg['B']
    PRI = cfg['PRI']
    alpha = B / PRI
    phi0 = cfg['phi0']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']
    tau = cfg['tau']

    phase_error = create_phase_error(cfg)

    """ real phase """
    if ctrl == 'positive':
        ideal_phase_delay = 2 * np.pi * (alpha * ((t_axis - tau) ** 2) / 2)  # up sweep --> Nyquist sampling
    elif ctrl == 'negative':
        ideal_phase_delay = 2 * np.pi * (B * (t_axis - tau) - alpha * ((t_axis - tau) ** 2) / 2)
        # down sweep --> Nyquist sampling
    real_phase = ideal_phase_delay + phase_error - phi0

    """ ideal IQ demodulation and LPF """
    s = np.exp(1j * real_phase)

    s_i = np.real(s)
    s_q = np.imag(s)
    return s_i, s_q


def phase_estimation_module(s_i, s_q):
    """ phase estimator """
    # s_complex = s_I + 1j * s_Q
    # practical_phase_python = np.angle(s_complex)
    practical_phase = PhaseModule.PhaseModule.phase_estimator(s_i, s_q)
    # print(practical_phase_python - practical_phase)

    """ phase unwrapping """
    practical_phase_unwrap = PhaseModule.PhaseModule.phase_unwrapping(practical_phase)

    """ relative phase """
    relative_practical_phase_unwrap = practical_phase_unwrap - practical_phase_unwrap[0]

    return relative_practical_phase_unwrap


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Config = {
        'fs': 100 * MHz,
        'B': 400 * MHz,
        'PRI': 10.24 * us,
        'phi0': np.pi/2,
        'ctrl': 'positive',
        'tau': 0 * ns,
        'f_error': [15, 7.5, 5],  # MHz
        'amp_error': [0.5, 0.8, 1],
    }

    """ Parameter setting """

    fs = Config['fs']
    ts = 1 / fs
    B = Config['B']
    PRI = Config['PRI']
    alpha = B / PRI
    phi0 = Config['phi0']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = Config['ctrl']
    tau = Config['tau']
    case = 'Ideal-case'

    ideal_phase, ideal_freq, ideal_phase_unwrap, ideal_freq_unwrap = create_ideal_phase(Config)

    phase_error = create_phase_error(Config)

    """ non-linearity phase error """
    phase_error_relative = phase_error - phase_error[0] # relative phase step
    phase_error_relative_wrap = PhaseModule.PhaseModule.phase_wrapping(phase_error_relative)

    """ real phase """
    s_i, s_q = create_real_signal(Config)

    ########################################
    """ Method 1 """
    practical_relative_phase_unwrap = phase_estimation_module(s_i, s_q)

    phase_error_est1 = practical_relative_phase_unwrap - ideal_phase_unwrap
    phase_error_est1_wrap = PhaseModule.PhaseModule.phase_wrapping(phase_error_est1)

    """ Figure """
    """ real Phase """
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Unwrapping estimated phase of PLL',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Unwrapping Phase', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, ideal_phase_unwrap, label='ideal phase - sub-Nyquist sampling')
    ax_phase.plot(t_axis / us, practical_relative_phase_unwrap, label='unwrapping estimated phase')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_unwrap_PLL_phase.pdf')

    """ Phase error """
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Relative Phase Error Estimation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis / us, phase_error_relative, linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis / us, phase_error_est1, '--', linewidth=1,
                  label='estimated relative phase error')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_unwrap_Rela_Phase_error_est.pdf')

    """ wrap Phase error """
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Wrapping Relative Phase Error Estimation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis / us, phase_error_relative_wrap, linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis / us, phase_error_est1_wrap, '--', linewidth=1,
                  label='estimated relative phase error')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_Rela_Phase_error_est.pdf')

    zoom = [0, 256]
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Wrapping Relative Phase Error Estimation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis[zoom[0]:zoom[1]] / us, phase_error_relative_wrap[zoom[0]:zoom[1]], linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis[zoom[0]:zoom[1]] / us, phase_error_est1_wrap[zoom[0]:zoom[1]], '--', linewidth=1,
                  label='estimated relative phase error')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_Rela_Phase_error_est_zoom.pdf')

    plt.show()