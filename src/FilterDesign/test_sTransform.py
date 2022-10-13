import numpy as np
import matplotlib . pyplot as plt
from scipy import signal

if __name__ == '__main__':
    import FilterModule

    K = 1
    omegan = 2 * np.pi * 275.0
    zeta = 0.1
    num_2ndpole = [K * omegan * omegan]
    den_2ndpole = [1, 2 * zeta * omegan, omegan* omegan]
    secondpole_h = signal.lti(num_2ndpole, den_2ndpole)

    w, mag, phase = signal.bode(secondpole_h, np.arange(100000))

    print(w)
    print(mag)
    print(phase)
    print(num_2ndpole, den_2ndpole)
    print(secondpole_h)

    freq_response = FilterModule.FilterModule.analog_bode(num_2ndpole, den_2ndpole, np.arange(100000))

    fig_mag = plt.figure(figsize=(7, 4), dpi=100)
    ax_mag = fig_mag.add_subplot()
    ax_mag.set_title('Magnitude of Frequency Response', fontsize=12)
    ax_mag.set_xlabel('log w', fontsize=10)
    ax_mag.set_ylabel('Magnitude - dB', fontsize=10)
    ax_mag.semilogx(w/2/np.pi, mag, label='auto')
    ax_mag.semilogx(w/2/np.pi, 20 * np.log10(np.abs(freq_response)), label='filter module')
    plt.legend()

    fig_ang = plt.figure(figsize=(7, 4), dpi=100)
    ax_ang = fig_ang.add_subplot()
    ax_ang.set_title('Phase of Frequency Response', fontsize=10)
    ax_ang.set_xlabel('log w', fontsize=10)
    ax_ang.set_ylabel('Magnitude - dB', fontsize=10)
    ax_ang.semilogx(w/2/np.pi, phase, label='auto')
    ax_ang.semilogx(w/2/np.pi, np.angle(freq_response) / np.pi * 180, label='filter module')
    plt.legend()

    plt.show()