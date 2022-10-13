import numpy as np
import matplotlib . pyplot as plt
from scipy import signal

if __name__ == '__main__':
    import FilterModule

    K = 1
    omegan = 2 * np.pi * 275.0
    zeta = 0.3
    num_2ndpole = [K * omegan * omegan]
    den_2ndpole = [1, 2 * zeta*omegan, omegan * omegan]
    secondpole_h = signal.lti(num_2ndpole, den_2ndpole)
    w, mag, phase = signal.bode(secondpole_h, np.arange(100000))

    secondpole_h_z = secondpole_h.to_discrete(dt=200.0e-6, method='tustin')

    print(w)
    print(mag)
    print(phase)
    print(num_2ndpole, den_2ndpole)
    print(secondpole_h)
    print(secondpole_h_z)

    freq_response = FilterModule.FilterModule.analog_bode(num_2ndpole, den_2ndpole, np.arange(100000))

    fig_mag = plt.figure(figsize=(7, 4), dpi=100)
    ax_mag = fig_mag.add_subplot()
    ax_mag.set_title('Magnitude of Frequency Response', fontsize=12)
    ax_mag.set_xlabel('log w', fontsize=10)
    ax_mag.set_ylabel('Magnitude - dB', fontsize=10)
    ax_mag.semilogx(w / 2 / np.pi, mag, label='auto')
    ax_mag.semilogx(w / 2 / np.pi, 20 * np.log10(np.abs(freq_response)), label='filter module')
    plt.legend()

    fig_ang = plt.figure(figsize=(7, 4), dpi=100)
    ax_ang = fig_ang.add_subplot()
    ax_ang.set_title('Phase of Frequency Response', fontsize=10)
    ax_ang.set_xlabel('log w', fontsize=10)
    ax_ang.set_ylabel('Magnitude - dB', fontsize=10)
    ax_ang.semilogx(w / 2 / np.pi, phase, label='auto')
    ax_ang.semilogx(w / 2 / np.pi, np.angle(freq_response) / np.pi * 180, label='filter module')
    plt.legend()

    t_duration = 1.0
    t_step = 1.0e-6
    no_of_data = int(t_duration / t_step)
    time_array = np.arange(no_of_data) * t_step
    frequency = 50.0
    omega = 2 * np.pi * frequency
    omega_noise = 2 * np.pi * 550.0
    inp_mag = np.sqrt(2) * 240.0
    ip_voltage_signal = (np.sin(time_array*omega) + 0.3 * np.sin(time_array * omega_noise))
    t_sample = 200.0e-6
    no_of_skip = int(t_sample / t_step)
    tsample_array = time_array[::no_of_skip]
    ip_voltage_samples = ip_voltage_signal[::no_of_skip]
    op_voltage_samples = np.zeros(ip_voltage_samples.size)
    u = np.zeros(3)
    y = np.zeros(3)

    for volt_index, volt_value in np.ndenumerate(ip_voltage_samples):
        u[0] = volt_value
        y[0] = (secondpole_h_z.num[0] * u[0]
                + secondpole_h_z.num[1] * u[1]
                + secondpole_h_z.num[2] * u[2]
                - secondpole_h_z.den[1] * y[1]
                - secondpole_h_z.den[2] * y[2]) / secondpole_h_z.den[0]
        u[2] = u[1]
        y[2] = y[1]
        u[1] = u[0]
        y[1] = y[0]
        op_voltage_samples[volt_index] = y[0]

    opt = FilterModule.FilterModule.digital_filter(secondpole_h_z.num, secondpole_h_z.den, ip_voltage_samples)

    fig_filter = plt.figure(figsize=(7, 4), dpi=100)
    ax_filter = fig_filter.add_subplot()
    ax_filter.plot(tsample_array, ip_voltage_samples, label ='input', ds ='steps')
    ax_filter.plot(tsample_array, op_voltage_samples, label ='auto', ds ='steps')
    ax_filter.plot(tsample_array, opt, label='filter module', ds='steps')
    ax_filter.set_title('Input versus output', fontsize=12)
    ax_filter.set_xlabel('Time', fontsize=10)
    ax_filter.set_ylabel('Volts', fontsize=10)
    plt.legend()

    # freq_response = FilterModule.FilterModule.digital_bode(secondpole_h_z.num, secondpole_h_z.den,
    #                                                        np.exp(1j * 2 * np.pi * np.arange(-0.5, 0.5, 0.001)))

    b = [0.0563, -9.3524e-4, -9.3524e-4, 0.0563]
    a = [1, -2.1291, 1.7834, -0.5435]

    freq_response = FilterModule.FilterModule.digital_bode(b, a,
                                                           np.exp(1j * 2 * np.pi * np.arange(-0.5, 0.5, 0.001)))

    fig_mag = plt.figure(figsize=(7, 4), dpi=100)
    ax_mag = fig_mag.add_subplot()
    ax_mag.set_title('Magnitude of Frequency Response', fontsize=12)
    ax_mag.set_xlabel('log w', fontsize=10)
    ax_mag.set_ylabel('Magnitude - dB', fontsize=10)
    ax_mag.plot(np.arange(-0.5, 0.5, 0.001), 20 * np.log10(np.abs(freq_response)), label='filter module')
    plt.legend()

    plt.show()