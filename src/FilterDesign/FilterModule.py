import numpy as np


class FilterModule:
    name = 'Filter Module'

    def __init__(self):
        pass

    @staticmethod
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
                for yix in range(num_y-1):
                    y[0] -= den[yix+1] * y[yix+1]

            y[0] = y[0] / den[0]

            if num_u != 1:
                for uidx in range(num_u-1):
                    u[num_u - uidx - 1] = u[num_u - uidx - 2]

            if num_y != 1:
                for yidx in range(num_y-1):
                    y[num_y - yidx - 1] = y[num_y - yidx - 2]

            opt[idx] = y[0]

        return opt

    @staticmethod
    def analog_bode(num, den, freq):
        freq_response = np.zeros(np.shape(freq), dtype=complex)
        for idx, k in np.ndenumerate(freq):
            num_response = FilterModule.cal_analog_response(num, k)
            den_response = FilterModule.cal_analog_response(den, k)
            freq_response[idx] = num_response / den_response
        return freq_response

    @staticmethod
    def cal_analog_response(coe, k):        # k = 2 * np.pi * frequency
        response = 0 + 0 * 1j
        num_coe = len(coe)
        if k != 0:
            for i in range(num_coe):
                response += coe[i]
                response = response * 1j * k

            response = response / 1j / k
        else:
            response = coe[-1]

        return response

    """ not verification """
    @staticmethod
    def digital_bode(num, den, freq):
        freq_response = np.zeros(np.shape(freq), dtype=complex)
        for idx, k in np.ndenumerate(freq):
            num_response = FilterModule.cal_digital_response(num, k)
            den_response = FilterModule.cal_digital_response(den, k)
            freq_response[idx] = num_response / den_response
        return freq_response

    @staticmethod
    def cal_digital_response(coe, k):       # k = np.exp(jw)
        response = 0 + 0 * 1j
        num_coe = len(coe)
        if k != 0:
            for i in range(num_coe):
                response += coe[i]
                response = response * k

            response = response / (k ** num_coe)
        else:
            response = coe[-1]
        return response


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy import signal

    """ S Transform """
    K = 1
    omegan = 2 * np.pi * 275.0
    zeta = 0.3
    num_2ndpole = [K * omegan * omegan]
    den_2ndpole = [1, 2 * zeta * omegan, omegan * omegan]
    secondpole_h = signal.lti(num_2ndpole, den_2ndpole)
    w, mag, phase = signal.bode(secondpole_h, np.arange(100000))

    secondpole_h_z = secondpole_h.to_discrete(dt=200.0e-6, method='tustin')

    print(w)
    print(mag)
    print(phase)
    print(num_2ndpole, den_2ndpole)
    print(secondpole_h)
    print(secondpole_h_z)

    freq_response = FilterModule.analog_bode(num_2ndpole, den_2ndpole, np.arange(100000))

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

    """ z Transform and Filtering """
    t_duration = 1.0
    t_step = 1.0e-6
    no_of_data = int(t_duration / t_step)
    time_array = np.arange(no_of_data) * t_step
    frequency = 50.0
    omega = 2 * np.pi * frequency
    omega_noise = 2 * np.pi * 550.0
    inp_mag = np.sqrt(2) * 240.0
    ip_voltage_signal = (np.sin(time_array * omega) + 0.3 * np.sin(time_array * omega_noise))
    t_sample = 200.0e-6
    no_of_skip = int(t_sample / t_step)
    tsample_array = time_array[::no_of_skip]
    ip_voltage_samples = ip_voltage_signal[::no_of_skip]
    op_voltage_samples = np.zeros(ip_voltage_samples.size)
    u = np.zeros(3)
    y = np.zeros(3)

    freq_response = FilterModule.digital_bode(secondpole_h_z.num, secondpole_h_z.den,
                                              np.exp(1j * 2 * np.pi * np.arange(-0.5, 0.5, 0.001)))
    fig_mag = plt.figure(figsize=(7, 4), dpi=100)
    ax_mag = fig_mag.add_subplot()
    ax_mag.set_title('Magnitude of Frequency Response', fontsize=12)
    ax_mag.set_xlabel('log w', fontsize=10)
    ax_mag.set_ylabel('Magnitude - dB', fontsize=10)
    ax_mag.plot(np.arange(-0.5, 0.5, 0.001), 20 * np.log10(np.abs(freq_response)), label='filter module')
    plt.legend()

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

    opt = FilterModule.digital_filter(secondpole_h_z.num, secondpole_h_z.den, ip_voltage_samples)

    fig_filter = plt.figure(figsize=(7, 4), dpi=100)
    ax_filter = fig_filter.add_subplot()
    ax_filter.plot(tsample_array, ip_voltage_samples, label='input', ds='steps')
    ax_filter.plot(tsample_array, op_voltage_samples, label='auto', ds='steps')
    ax_filter.plot(tsample_array, opt, label='filter module', ds='steps')
    ax_filter.set_title('Input versus output', fontsize=12)
    ax_filter.set_xlabel('Time', fontsize=10)
    ax_filter.set_ylabel('Volts', fontsize=10)
    plt.legend()

    plt.show()