import numpy as np
import matplotlib.pyplot as plt


class NuFFTModule:
    name = 'NuFFT Module'

    def __init__(self):
        pass

    @staticmethod
    def direct_summation(f, x, M, NdivbyM):
        N = len(f)
        F = np.zeros(M, dtype=complex)
        idx = 0
        for k in range(-M // 2, M // 2, 1):
            sum = 0 + 0j
            for n in range(0, N, 1):
                sum = sum + f[n] * np.exp(-1j * k * NdivbyM * x[n])

            F[idx] = sum
            idx = idx + 1
        return F
    '''
    Author: zhan.zhang493@qq.com
    Date: 2022-08-22
    Name: direct_summation
    PURPOSE: Computes the discrete Fourier transform of the uniform or non-uniform positioned data f by direct summation.

    INPUT:  f = [f_1; f_2; ... ;f_N]   1-dimensional input data
            x = [x_1; x_2; ... ;x_N]   uniform or non-uniform positions
                                       in [0,2*pi]
            M                          number of frequencies k s.t.
                                       -M/2 <= k < M/2

    OUTPUT: F = [F_1; F_2; ... ;F_M]  Fourier coefficients '''


    @staticmethod
    def nufft_1d(f, x, M, R, M_sp, tau, NdivbyM):
        N = len(f)

        # Num of samples in oversampled grid
        M_r = R * M

        if NdivbyM < 1:
            x = x * NdivbyM

        # Frequencies
        k = np.arange(-M // 2, M // 2, 1)

        # Finde upsampled grid in [0, 2*\pi]
        xi = np.linspace(0, 2 * np.pi, M_r)

        # Step size in upsampled grid
        h = 2 * np.pi / M_r

        # Find indices in xi that are the closet to x_n
        idx = NuFFTModule.closet_grid(x, xi)

        # Convolve f with g_tau
        f_tau = np.zeros(M_r, dtype=complex)

        for n in range(0, N, 1):
            for l in range(-M_sp, M_sp + 1, 1):
                f_tau[np.mod(idx[n] + l, M_r)] = \
                    f_tau[np.mod(idx[n] + l, M_r)] \
                    + f[n] * np.exp(-(x[n] - h * idx[n] - h * l) ** 2 / (4 * tau))

        fig_inter = plt.figure(figsize=(7, 4), dpi=100)
        ax_inter = fig_inter.add_subplot()
        ax_inter.set_title("Interpolation of non-uniform samples", fontsize=6)
        ax_inter.set_xlabel('n (GHz)', fontsize=6)
        ax_inter.set_ylabel('f (amplitude)', fontsize=6)
        ax_inter.plot(np.abs(f_tau), linewidth=1, color='k')

        # Standard FFT
        F_tau = np.fft.fftshift(np.fft.fft(f_tau, M_r))

        # Compute Fourier coefficients
        if NdivbyM > 1:
            F = np.sqrt(np.pi / tau) * np.exp(k ** 2 * tau) \
                * F_tau[int(M_r // 2 - (M * NdivbyM) // 2): int(M_r // 2 + (M * NdivbyM) // 2): int(NdivbyM)] / M_r
        else:
            F = np.sqrt(np.pi / tau) * np.exp(k ** 2 * tau) \
                * F_tau[int(M_r // 2 - M // 2): int(M_r // 2 + M // 2): 1] / M_r
        return F

    '''
    Name: nufft_1d
    PURPOSE: Approximates the 1-dimensional non-uniform discrete Fourier transform 
             using the non-uniform fast Fourier transform (NUFFT) algorithm. See 
             Greengard and Lee (2004) for more details.
    INPUT: f = [f_1; f_2; ... ;f_N]   1-dimensional input data
            x = [x_1; x_2; ... ;x_N]   uniform or non-uniform positions
                                       in [0,2*pi]
            M                          number of frequencies k s.t.
                                       -M/2 <= k < M/2
            R                          oversampling factor
            M_sp                       number of neighbours to spread data to
            tau                        spreading factor of Gaussian kernel
    OUTPUT: F = [F_1; F_2; ... ;F_M]  Fourier coefficients '''

    @staticmethod
    def closet_grid(x, xi):
        nx = len(x)
        nxi = len(xi)
        idx = np.zeros(nx, dtype=int)
        for i in range(nx):
            distance = np.zeros(nxi, dtype=float)
            for j in range(nxi):
                distance[j] = np.abs(x[i] - xi[j])
            idx[i] = np.argmin(distance)
        return idx

    @staticmethod
    def plot_Gaussian_kernal_func(tau):
        t = np.arange(-5, 5, 0.01)
        tau = 1
        g_k = np.exp(-t ** 2 / (4 * tau))
        fig_gk = plt.figure(figsize=(7, 4), dpi=100)
        ax_gk = fig_gk.add_subplot()
        ax_gk.set_title("Gaussian kernal function", fontsize=6)
        ax_gk.set_xlabel('t (sec)', fontsize=6)
        ax_gk.set_ylabel('y (amplitude)', fontsize=6)
        ax_gk.plot(t, g_k, linewidth=1, color='b')

    @staticmethod
    def relative_error_norm(x_hat, x):
        return np.linalg.norm(x_hat - x)


if __name__ == '__main__':
    # Const
    np.random.seed(0)

    GHz = 1e9
    MHz = 1e6
    us = 1e-6
    ns = 1e-9
    fs = 1 * GHz
    Ts = 1 / fs

    # Parameter setting
    numSnap = 2 ** 8
    N = numSnap
    M = N
    NdivbyM = N / M

    R = 2
    M_sp = 4
    tau = (1 / (M ** 2)) * (np.pi * M_sp) / (R * (R - 0.5))

    a = 0
    b = 2 * np.pi

    nx = np.arange(0, N, 1) + 0.6 * np.random.rand(N)
    x = nx / N * 2 * np.pi

    t = np.arange(0, 10 * N, 1) / N / 10 * 2 * np.pi

    # Data vector
    amp_1 = 2
    amp_2 = 0.1

    f_1 = 25
    f_2 = 60
    f = 1 / N * (amp_1 * np.sin(f_1 * x) + amp_2 * np.sin(f_2 * x))
    s = 1 / N * (amp_1 * np.sin(f_1 * t) + amp_2 * np.sin(f_2 * t))

    k = np.arange(-M // 2, M // 2, 1)

    # Direct summation
    F_ds = NuFFTModule.direct_summation(f, x, M, NdivbyM)

    f_ds = np.fft.ifft(np.fft.fftshift(F_ds))

    F_ds = np.abs(F_ds[M // 2:])

    # Nufft
    F_nufft = NuFFTModule.nufft_1d(f, x, M, R, M_sp, tau, NdivbyM)
    F_nufft = np.abs(F_nufft[M // 2:])

    # FFT
    F_fft = np.fft.fft(f, M)
    F_fft = np.abs(F_fft[0:M // 2])

    freq = k[(M // 2):] / M * fs

    fig1 = plt.figure(figsize=(5, 3), dpi=150)
    ax1 = fig1.add_subplot()
    ax1.set_title("Original signal and its non-uniform samples", fontsize=6)
    ax1.set_xlabel('t (nano-sec)', fontsize=6)
    ax1.set_ylabel('y (amplitude)', fontsize=6)
    ax1.plot(t / 2 / np.pi * N * Ts / ns, s, linewidth=1, color='k')
    ax1.plot(x / 2 / np.pi * N * Ts / ns, f, linewidth=0.5, color='r', marker='x')

    fig2 = plt.figure(figsize=(5, 3), dpi=150)
    ax2 = fig2.add_subplot()
    ax2.set_title("FFT of non-uniform samples", fontsize=6)
    ax2.set_xlabel('freq (GHz)', fontsize=6)
    ax2.set_ylabel('y (amplitude)', fontsize=6)
    ax2.plot(freq / GHz, 20 * np.log10(F_ds) - np.max(20 * np.log10(F_ds)), linewidth=1, color='k', label='DS')
    ax2.plot(freq / GHz, 20 * np.log10(F_nufft) - np.max(20 * np.log10(F_nufft)), linewidth=1, linestyle=':',
             color='r', label='NuFFT')
    ax2.plot(freq / GHz, 20 * np.log10(F_fft) - np.max(20 * np.log10(F_fft)), linestyle='-.',
             linewidth=0.5, color='b', marker='o', label='FFT')
    plt.legend()

    NuFFTModule.plot_Gaussian_kernal_func(1)

    plt.show()