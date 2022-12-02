import numpy as np
import matplotlib.font_manager as fm
font_times = fm.FontProperties(family='Times New Roman', stretch=0)


class WindowModule:
    name = 'Window Module'

    def __init__(self):
        pass

    @staticmethod
    def black_man(num_order, ctrl='symmetric'):
        if ctrl == 'periodic':
            num_point = num_order + 1
        elif ctrl == 'symmetric':
            num_point = num_order
        else:
            num_point = num_order
            print('window ctrl is wrong.\n')
        n = np.arange(num_point)
        win_all = 0.42 - 0.5 * np.cos(2 * np.pi * n / (num_point - 1)) + 0.08 * np.cos(4 * np.pi * n / (num_point - 1))
        win = win_all[:num_order]

        return win

    @staticmethod
    def hamming(num_order, ctrl='symmetric'):
        if ctrl == 'periodic':
            num_point = num_order + 1
        elif ctrl == 'symmetric':
            num_point = num_order
        else:
            num_point = num_order
            print('window ctrl is wrong.\n')

        n = np.arange(num_point)
        win_all = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_point - 1))
        win = win_all[:num_order]
        return win

    @staticmethod
    def hanning(num_order, ctrl='symmetric'):
        if ctrl == 'periodic':
            num_point = num_order + 1
        elif ctrl == 'symmetric':
            num_point = num_order
        else:
            num_point = num_order
            print('window ctrl is wrong.\n')

        n = np.arange(num_point)
        win_all = 0.5 * (1 - np.cos(2 * np.pi * n / (num_point - 1)))
        win = win_all[:num_order]

        return win


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    NUM_ORDER = 11
    WIN_HANNING_S = WindowModule.hanning(NUM_ORDER)
    WIN_HANNING_P = WindowModule.hanning(NUM_ORDER, 'periodic')
    print(WIN_HANNING_S)
    print(WIN_HANNING_P)

    WIN_HAMMING_S = WindowModule.hamming(NUM_ORDER)
    WIN_HAMMING_P = WindowModule.hamming(NUM_ORDER, 'periodic')
    print(WIN_HAMMING_S)
    print(WIN_HAMMING_P)

    WIN_BLACKMAN_S = WindowModule.black_man(NUM_ORDER)
    WIN_BLACKMAN_P = WindowModule.black_man(NUM_ORDER, 'periodic')
    print(WIN_BLACKMAN_S)
    print(WIN_BLACKMAN_P)
