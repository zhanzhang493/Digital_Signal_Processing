if __name__ == '__main__':
    import NuFFT
    import numpy as np
    import matplotlib.pyplot as plt

    # Const
    GHz = 1e9
    MHz = 1e6
    KHz = 1e3
    ms = 1e-3
    us = 1e-6
    ns = 1e-9
    fs = 40 * MHz
    Ts = 1 / fs

    # Parameter setting
    numSnap = 2 ** 8
    N = numSnap

    numFreq_factor = 4  # 2, 4, 8, 16
    M = numFreq_factor * N

    numTx = 4

    NdivbyM = numTx / numFreq_factor

    R = 2
    M_sp = 3
    tau = (1 / (M ** 2)) * (np.pi * M_sp) / (R * (R - 0.5))

    numTx = 4

    # Data vector
    amp_1 = 2
    amp_2 = 1

    f_1 = 16 * MHz
    f_2 = 11 * MHz

    # Non-uniform position vector
    t = np.arange(0, numTx * 10 * N, 1) * Ts / 10
    s = 1 / N * (amp_1 * np.sin(2 * np.pi * f_1 * t) + amp_2 * np.sin(2 * np.pi * f_2 * t))

    nx = np.arange(0, N, 1) * numTx + np.random.randint(0, numTx, N)
    tx = nx * Ts
    f = 1 / N * (amp_1 * np.sin(2 * np.pi * f_1 * tx) + amp_2 * np.sin(2 * np.pi * f_2 * tx))

    f_zero = np.zeros(N * numTx)
    f_zero[nx] = f

    x = nx / numTx / N * 2 * np.pi

    # if NdivbyM < 1:
    #     x = x * NdivbyM

    k = np.arange(-M // 2, M // 2, 1)

    # Direct summation
    F_ds = NuFFT.NuFFTModule.direct_summation(f, x, M, NdivbyM)
    F_ds = np.abs(F_ds[M // 2:])

    # direct FFT
    F_fft = np.fft.fft(f, M)
    F_fft = np.abs(F_fft[:M // 2])

    # Nufft
    F_nufft = NuFFT.NuFFTModule.nufft_1d(f, x, M, R, M_sp, tau, NdivbyM)
    F_nufft = np.abs(F_nufft[M // 2:])

    # zero-padding FFT
    F_zeros = np.fft.fft(f_zero, M)
    F_zeros = np.abs(F_zeros[:M // 2])

    # error
    error = NuFFT.NuFFTModule.relative_error_norm(F_nufft, F_ds)
    print(error)

    freq = k[(M // 2):] / M * fs

    fig1 = plt.figure(figsize=(7, 4), dpi=100)
    ax1 = fig1.add_subplot()
    ax1.set_title("Original signal and its non-uniform samples", fontsize=6)
    ax1.set_xlabel('t (micro-sec)', fontsize=6)
    ax1.set_ylabel('y (amplitude)', fontsize=6)
    ax1.plot(t / us, s, linewidth=1, color='k')
    ax1.plot(tx / us, f, linewidth=0.5, color='r', marker='x')

    fig2 = plt.figure(figsize=(7, 4), dpi=100)
    ax2 = fig2.add_subplot()
    ax2.set_title("FFT of non-uniform samples", fontsize=10)
    ax2.set_xlabel('freq (MHz)', fontsize=8)
    ax2.set_ylabel('$\mathdefault{20\log_{10}(F)}$ (dB)', fontsize=8)
    ax2.plot(freq / MHz, 20 * np.log10(F_ds), linewidth=1, color='k', label='DS')
    ax2.plot(freq / MHz, 20 * np.log10(F_nufft), linestyle='--', linewidth=0.5,
             color='r', marker='x', label='NuFFT')
    ax2.plot(freq / MHz, 20 * np.log10(F_fft), linestyle='--', linewidth=0.5,
             color='g', marker='o', label='FFT')
    ax2.plot(freq / MHz, 20 * np.log10(F_zeros), linestyle='-.', linewidth=0.2,
             color='y', marker='.', label='FFT_zeroPadding')
    plt.legend(loc='best', shadow=True, fontsize='x-small')
    plt.tick_params(labelsize=8)

    plt.show()