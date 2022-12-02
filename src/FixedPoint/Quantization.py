import math

import numpy as np
import math
import matplotlib.font_manager as fm
font_times = fm.FontProperties(family='Times New Roman', stretch=0)


class Quantizer:
    name = 'Quantizer'

    def __init__(self):
        pass

    @staticmethod
    def float_to_fix_single_point(x, bit_width, integer, signed, q_mode, o_mode):
        pass

    """##############################################################################################################"""
    """Quantization modes for ac_fixed"""
    @staticmethod
    def trn(x):
        return math.floor(x)

    @staticmethod
    def trn_zero(x):
        return int(x)

    @staticmethod
    def rnd_zero(x):
        return round(x)

    """##############################################################################################################"""
    """Overflow modes for ac_fixed"""
    @staticmethod
    def sat_sym(x, bit_width, integer, signed=True):
        assert bit_width >= integer >= 0
        if not signed:
            max_value = (1 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = 0
        elif signed:
            max_value = (0.5 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = -max_value
        else:
            print('Sign info is wrong.\n')
            max_value = (0.5 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = -max_value

        if x > max_value:
            y = max_value
        elif x < min_value:
            y = min_value
        else:
            y = x

        return y


if __name__ == '__main__':
    X = -8
    print(X, bin(X & 0xf))

    X = 2.1
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = -2.1
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = 2.5
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = -2.5
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = 2.6
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = -2.6
    print(int(X), round(X), math.floor(X), math.ceil(X))

    X = 16
    Y = Quantizer.sat_sym(X, 5, 5)
    print(Y)

