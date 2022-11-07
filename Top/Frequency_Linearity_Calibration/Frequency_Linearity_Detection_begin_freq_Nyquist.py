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


def create_pll_phase(cfg):
    fs = cfg['fs']
    ts = 1/fs
    B = cfg['B']
    PRI = cfg['PRI']
    alpha = B / PRI
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']
    tau = cfg['tau']

    """ Ideal phase """
    if ctrl == 'positive':
        ideal_phase = 2 * np.pi * (alpha * (t_axis ** 2) / 2)  # up sweep --> Nyquist sampling
        phase_delay = 2 * np.pi * (alpha * ((t_axis - tau) ** 2) / 2)  # up sweep --> Nyquist sampling
    else:
        ideal_phase = 2 * np.pi * (B * t_axis - alpha * (t_axis ** 2) / 2)  # down sweep --> Nyquist sampling
        phase_delay = 2 * np.pi * (B * (t_axis - tau) - alpha * ((t_axis - tau) ** 2) / 2)

    ideal_freq = (ideal_phase[1:] - ideal_phase[0:-1]) / ts / 2 / np.pi  # freq --> Nyquist sampling

    """ Ideal IQ Demodulation """
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
    ax_phase.set_title('Ideal Phase of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Unwrapping Phase - rad.', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, ideal_phase, label='ideal phase - Nyquist')
    ax_phase.plot(t_axis / us, ideal_phase_unwrap, label='ideal phase - sub-Nyquist sampling')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_ideal_sweeping_phase.pdf')

    """ ideal Frequency """
    fig_freq = plt.figure(figsize=(7, 5), dpi=100)
    ax_freq = fig_freq.add_subplot()
    ax_freq.set_title('Sweeping Frequency of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_freq.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_freq.set_ylabel('Freq - MHz', fontsize=10, fontproperties=font_times)
    ax_freq.plot(t_axis[1:] / us, ideal_freq / MHz, label='ideal freq - Nyquist')
    ax_freq.plot(t_axis[1:] / us, ideal_freq_unwrap / MHz, label='ideal freq - sub-Nyquist sampling')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_ideal_sweeping_Freq.pdf')
    return ideal_phase, ideal_freq, ideal_phase_unwrap, ideal_freq_unwrap, phase_delay


def create_phase_error(cfg):
    """ create phase error, including real phase error and delay phase error """
    fs = cfg['fs']
    ts = 1 / fs
    PRI = cfg['PRI']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    tau = cfg['tau']
    amp = cfg['amp_error']
    f_error = cfg['f_error']
    assert len(amp) == len(f_error)
    num_error = len(f_error)
    ctrl = cfg['ctrl']

    phase_error_ideal = np.zeros(N, dtype=float)
    phase_error_delay = np.zeros(N, dtype=float)
    for k in range(num_error):
        phase_error_ideal = phase_error_ideal + amp[k] * np.sin(2 * np.pi * f_error[k] * MHz * t_axis)
        phase_error_delay = phase_error_delay + amp[k] * np.sin(2 * np.pi * f_error[k] * MHz * (t_axis - tau))

    real_freq_diff = (phase_error_ideal[1:] - phase_error_ideal[0:-1]) / ts / 2 / np.pi

    """ Figure """
    """ Phase error """
    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Phase error',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - rad.', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, phase_error_ideal, label='real phase error')
    plt.legend(fontsize=8)
    plt.savefig('real_phase_error.pdf')

    """ ideal Frequency """
    fig_freq = plt.figure(figsize=(7, 5), dpi=100)
    ax_freq = fig_freq.add_subplot()
    ax_freq.set_title('Real frequency difference',
                      fontsize=12, fontproperties=font_times)
    ax_freq.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_freq.set_ylabel('Freq - MHz', fontsize=10, fontproperties=font_times)
    ax_freq.plot(t_axis[1:] / us, real_freq_diff / MHz, label='real freq difference')
    plt.legend(fontsize=8)
    plt.savefig('real_freq_difference.pdf')

    return phase_error_ideal, phase_error_delay


def phase_estimation_module(cfg, s_i, s_q, ideal_phase_unwrap, ideal_freq_unwrap, phase_error_ideal):
    """ phase estimator, phase unwrapping, relative phase """
    fs = cfg['fs']
    ts = 1 / fs
    PRI = cfg['PRI']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']
    case = cfg['case']

    """ Phase estimator """
    # s_complex = s_I + 1j * s_Q
    # practical_phase_python = np.angle(s_complex)
    real_phase_est = PhaseModule.PhaseModule.phase_estimator(s_i, s_q)
    # print(practical_phase_python - practical_phase)

    """ phase unwrapping """
    real_phase_est_unwrap = PhaseModule.PhaseModule.phase_unwrapping(real_phase_est)

    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Unwrapping estimated phase of PLL',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_phase.set_ylabel('Unwrapping Phase', fontsize=10, fontproperties=font_times)
    ax_phase.plot(t_axis / us, ideal_phase_unwrap, label='ideal phase - sub-Nyquist sampling')
    ax_phase.plot(t_axis / us, real_phase_est_unwrap, label='unwrapping estimated phase')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_unwrap_est_phase.pdf')

    # freq
    real_freq_est_unwrap = (real_phase_est_unwrap[1:] - real_phase_est_unwrap[0:-1]) / ts / 2 / np.pi  # sub-Nyquist sampling

    fig_freq = plt.figure(figsize=(7, 5), dpi=100)
    ax_freq = fig_freq.add_subplot()
    ax_freq.set_title('Sweeping Frequency of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_freq.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_freq.set_ylabel('Freq', fontsize=10, fontproperties=font_times)
    ax_freq.plot(t_axis[1:] / us, ideal_freq_unwrap / MHz, label='ideal freq')
    ax_freq.plot(t_axis[1:] / us, real_freq_est_unwrap / MHz, label='real freq')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_unwrap_est_freq.pdf')

    """ phase de-ramping"""
    phase_error_est = real_phase_est_unwrap - ideal_phase_unwrap

    """ relative phase """
    relative_phase_error_est = phase_error_est - phase_error_est[0]

    relative_phase_error= phase_error_ideal - phase_error_ideal[0]

    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Relative Phase Error Estimation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis / us, relative_phase_error, linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis / us, relative_phase_error_est, '--', linewidth=1,
                  label='estimated relative phase error')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_unwrap_Rela_Phase_error_est.pdf')

    """ phase wrapping """
    relative_phase_error_est_wrap = PhaseModule.PhaseModule.phase_wrapping(relative_phase_error_est)

    relative_phase_error_wrap = PhaseModule.PhaseModule.phase_wrapping(relative_phase_error)

    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Wrapping Relative Phase Error Estimation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis / us, relative_phase_error_wrap, linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis / us, relative_phase_error_est_wrap, '--', linewidth=1,
                  label='estimated relative phase error')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_wrap_Rela_Phase_error_est.pdf')

    return real_phase_est_unwrap, real_freq_est_unwrap, relative_phase_error_wrap, relative_phase_error_est_wrap


def compensation_chirp_tau(cfg, relative_phase_error_wrap, relative_phase_error_est_wrap):
    fs = cfg['fs']
    ts = 1 / fs
    B = cfg['B']
    PRI = cfg['PRI']
    alpha = B / PRI
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts
    ctrl = cfg['ctrl']
    tau = cfg['tau']
    case = cfg['case']

    phi_comp = 2 * np.pi * alpha * ts * tau * np.arange(N) - np.pi * alpha * tau ** 2

    relative_phase_error_est_wrap_comp = PhaseModule.PhaseModule.phase_wrapping(relative_phase_error_est_wrap +
                                                                                phi_comp -
                                                                                relative_phase_error_est_wrap[1] -
                                                                                phi_comp[1])

    fig_phase = plt.figure(figsize=(7, 5), dpi=100)
    ax_phase = fig_phase.add_subplot()
    ax_phase.set_title('Wrapping Relative Phase Error Estimation after compensation',
                       fontsize=12, fontproperties=font_times)
    ax_phase.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_phase.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_phase.plot(t_axis / us, relative_phase_error_wrap, linewidth=2,
                  label='relative phase error')
    ax_phase.plot(t_axis[0:-1] / us, relative_phase_error_est_wrap_comp[1:], '--', linewidth=1,
                  label='estimated relative phase error after compensation')
    plt.legend(fontsize=8)
    plt.savefig(ctrl + '_' + case + '_comp_wrap_Rela_Phase_error_est.pdf')

    return phi_comp, relative_phase_error_est_wrap_comp


# def compensation_phase_error_delay(cfg, relative_phase_error_est_wrap_comp):
#     FilterModule.digital_filter(num, den, relative_phase_error_est_wrap_comp)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """ =================================================================================================== """
    Config = {
            'case': 'no_delay',
            'fs': 1000 * MHz,
            'B': 400 * MHz,
            'PRI': 10.24 * us,
            'phi0': np.pi / 2,
            'ctrl': 'negative',
            'tau': 0 * ns,
            'f_error': [15, 7.5, 5],  # MHz
            'amp_error': [1, 0.8, 1],
    }
    ctrl = Config['ctrl']
    fs = Config['fs']
    ts = 1 / fs
    PRI = Config['PRI']
    N = int(fs * PRI)
    t_axis = np.arange(N) * ts

    """ =================================================================================================== """
    """ ideal_phase, including Nyquist and sub-Nyquist sampling """
    ideal_phase, ideal_freq, ideal_phase_unwrap, ideal_freq_unwrap, ideal_phase_delay = create_pll_phase(Config)

    """ phase error """
    phase_error_ideal, phase_error_delay = create_phase_error(Config)
    phase_error_ideal_relative = phase_error_ideal - phase_error_ideal[0]  # relative phase step
    phase_error_ideal_relative_wrap = PhaseModule.PhaseModule.phase_wrapping(phase_error_ideal_relative)

    """ create IQ demodulation and LPF """
    real_phase = ideal_phase_delay + phase_error_delay
    s = np.exp(1j * real_phase)
    s_i = np.real(s)
    s_q = np.imag(s)

    """ =================================================================================================== """
    """ Phase error estimation """
    real_phase_est_unwrap, real_freq_est_unwrap, relative_phase_error_wrap, relative_phase_error_est_wrap \
        = phase_estimation_module(Config, s_i, s_q, ideal_phase_unwrap, ideal_freq_unwrap, phase_error_ideal)

    """ Compensation for delay in chirp """
    phi_comp, relative_phase_error_est_wrap_comp = compensation_chirp_tau(Config,
                                                                          relative_phase_error_wrap,
                                                                          relative_phase_error_est_wrap)

    """ =================================================================================================== """
    """ Figure """
    fig_spec = plt.figure(figsize=(36, 12), dpi=50)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    num_col = 3
    num_row = 2
    ax_spec = fig_spec.add_subplot(num_row, num_col, 1)
    ax_spec.set_title('Ideal Phase of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Unwrapping Phase - rad.', fontsize=10, fontproperties=font_times)
    ax_spec.plot(t_axis / us, ideal_phase, 'k', label='ideal phase')
    ax_spec.plot(t_axis / us, ideal_phase_unwrap, 'b:', label='unwrapped ideal phase')
    plt.legend(fontsize=8)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 4)
    ax_spec.set_title('Sweeping Frequency of PLL after IQ Demodulation - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Freq - MHz', fontsize=10, fontproperties=font_times)
    ax_spec.plot(t_axis[1:] / us, ideal_freq / MHz, 'k', label='ideal freq')
    ax_spec.plot(t_axis[1:] / us, ideal_freq_unwrap / MHz, 'b:', label='ideal alias freq')
    plt.legend(fontsize=8)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 2)
    ax_spec.set_title('Unwrapping estimated phase of PLL',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Unwrapping Phase', fontsize=10, fontproperties=font_times)
    ax_spec.plot(t_axis / us, ideal_phase_unwrap, 'b', label='unwrapped ideal phase')
    ax_spec.plot(t_axis / us, real_phase_est_unwrap, 'orange', label='unwrapping estimated phase')
    plt.legend(fontsize=8)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 5)
    ax_spec.set_title('Estimated Frequency with unwrapping estimated phase of PLL - ' + ctrl + ' chirp',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=10, fontproperties=font_times)
    ax_spec.set_ylabel('Freq', fontsize=10, fontproperties=font_times)
    ax_spec.plot(t_axis[1:] / us, real_freq_est_unwrap / MHz, 'orange', label='real freq')
    ax_spec.plot(t_axis[1:] / us, ideal_freq_unwrap / MHz, 'b', label='ideal alias freq')
    plt.legend(fontsize=8)

    relative_phase_error = phase_error_ideal - phase_error_ideal[0]

    phase_error_est = real_phase_est_unwrap - ideal_phase_unwrap
    relative_phase_error_est = phase_error_est - phase_error_est[0]

    ax_spec = fig_spec.add_subplot(num_row, num_col, 3)
    ax_spec.set_title('Relative Phase Error Estimation',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_spec.plot(t_axis / us, relative_phase_error, 'b', linewidth=2,
                 label='relative phase error')
    ax_spec.plot(t_axis / us, relative_phase_error_est, color='orange', linestyle='--', linewidth=1,
                 label='estimated relative phase error')
    plt.legend(fontsize=8)

    ax_spec = fig_spec.add_subplot(num_row, num_col, 6)
    ax_spec.set_title('Wrapping Relative Phase Error Estimation',
                      fontsize=12, fontproperties=font_times)
    ax_spec.set_xlabel('time - us', fontsize=8, fontproperties=font_times)
    ax_spec.set_ylabel('Phase - [-pi, pi]', fontsize=8, fontproperties=font_times)
    ax_spec.plot(t_axis / us, relative_phase_error_wrap, 'b', linewidth=2,
                 label='relative phase error')
    ax_spec.plot(t_axis / us, relative_phase_error_est_wrap, color='orange', linestyle='--', linewidth=1,
                 label='estimated relative phase error')
    plt.legend(fontsize=8)

    plt.savefig('Phase_estimation_Nyquist_begin_chirp_' + ctrl + '.pdf')

    plt.show()

    plt.show()