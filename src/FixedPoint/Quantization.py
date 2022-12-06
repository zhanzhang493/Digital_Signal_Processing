import math
import re
import numpy as np
import math
import matplotlib.font_manager as fm
font_times = fm.FontProperties(family='Times New Roman', stretch=0)


class Quantizer:
    name = 'Quantizer'

    def __init__(self):
        pass

    """##############################################################################################################"""
    """ Hex or Binary to Decimal"""
    @staticmethod
    def hex_to_dec(x, bit_width, integer, signed):
        num_point = len(x)
        y = np.zeros(num_point, dtype=float)
        for k in range(num_point):
            y[k] = Quantizer.hex_to_dec_single_point(x[k], bit_width, integer, signed)
        return y

    @staticmethod
    def hex_to_dec_single_point(x, bit_width, integer, signed):
        assert re.match(r'(0)(x)([\d a-fA-F]+)', x)
        m = re.match(r'(0)(x)([\d a-fA-F]+)', x)
        assert bit_width >= integer >= 0
        fraction = bit_width - integer
        if not signed:
            y = eval('0' + 'x' + m.group(3)) / (2**fraction)
        else:
            tmp = eval('0' + 'x' + m.group(3))
            if tmp >= 2 ** (bit_width - 1):
                y = (tmp - 2 ** bit_width) / (2 ** fraction)
            else:
                y = tmp / (2**fraction)
        return y

    @staticmethod
    def bin_to_dec(x, bit_width, integer, signed):
        num_point = len(x)
        y = np.zeros(num_point, dtype=float)
        for k in range(num_point):
            y[k] = Quantizer.bin_to_dec_single_point(x[k], bit_width, integer, signed)
        return y

    @staticmethod
    def bin_to_dec_single_point(x, bit_width, integer, signed):
        assert re.match(r'(0)(b)([0-1]+)', x)
        m = re.match(r'(0)(b)([0-1]+)', x)
        assert len(m.group(3)) <= bit_width
        assert bit_width >= integer >= 0
        fraction = bit_width - integer
        if not signed:
            y = eval('0' + 'b' + m.group(3)) / (2**fraction)
        else:
            tmp = eval('0' + 'b' + m.group(3))
            if tmp >= 2 ** (bit_width - 1):
                y = (tmp - 2 ** bit_width) / (2**fraction)
            else:
                y = tmp / (2**fraction)
        return y

    """##############################################################################################################"""
    """ float-point to fixed-point"""
    @staticmethod
    def float_to_fix_1d(x, bit_width, integer, signed, q_mode='rnd_zero', o_mode='sat_sym'):
        num_point = len(x)
        y = np.zeros(num_point, dtype=float)
        z = list()
        for k in range(num_point):
            _, y[k], fix_point = Quantizer.float_to_fix_single_point(x[k], bit_width, integer, signed, q_mode, o_mode)
            z.append(fix_point)
        return x, y, z

    @staticmethod
    def float_to_fix_single_point(x, bit_width, integer, signed, q_mode='rnd_zero', o_mode='sat_sym'):
        assert isinstance(bit_width, int) and isinstance(integer, int)
        if o_mode == 'sat_sym':
            x1 = Quantizer.sat_sym(x, bit_width, integer, signed)

        fraction = bit_width - integer
        x2 = x1 * (2**fraction)

        if q_mode == 'rnd_zero':
            x3 = Quantizer.rnd_zero(x2)

        y = x3 / (2**fraction)
        z = hex(x3 & (2**bit_width - 1))

        return x, y, z

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
    def sat(x, bit_width, integer, signed=True):
        assert isinstance(bit_width, int) and isinstance(integer, int)
        assert bit_width >= integer >= 0
        if not signed:
            max_value = (1 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = 0
        elif signed:
            max_value = (0.5 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = -0.5 * (2 ** integer)
        else:
            print('Sign info is wrong.\n')
            max_value = (0.5 - (2 ** (-bit_width))) * (2 ** integer)
            min_value = -0.5 * (2 ** integer)

        if x > max_value:
            y = max_value
        elif x < min_value:
            y = min_value
        else:
            y = x

        return y

    @staticmethod
    def sat_sym(x, bit_width, integer, signed=True):
        assert isinstance(bit_width, int) and isinstance(integer, int)
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

    X = -0.388190791409149
    X, Y, Z = Quantizer.float_to_fix_single_point(X, 8, 2, True)
    print(X, Y, Z)

    X = 2.5
    X, Y, Z = Quantizer.float_to_fix_single_point(X, 8, 2, True)
    print(X, Y, Z)
    print(type(Z))

    a1 = [-0.388190791409149, -0.285741152113574, -0.138801927534664, -0.025907440076305]
    X, Y, Z = Quantizer.float_to_fix_1d(a1, 8, 2, True)
    print(X)
    print(Y)
    print(Z)

    a2 = [0.057398668463065, 0.186286541864827, 0.4230987127079, 0.76740478730534]
    X, Y, Z = Quantizer.float_to_fix_1d(a2, 8, 0, False)
    print(X)
    print(Y)
    print(Z)

    b1 = [1.917977679727313, 1.419457762233652, 0.897984940087441, 0.615899604416527]
    X, Y, Z = Quantizer.float_to_fix_1d(b1, 8, 2, True)
    print(X)
    print(Y)
    print(Z)

    X = 7.5
    Y = Quantizer.sat_sym(X, 5, 4)
    Z = Quantizer.sat(X, 5, 4)
    print(X, Y, Z)

    X = -8
    Y = Quantizer.sat_sym(X, 5, 4)
    Z = Quantizer.sat(X, 5, 4)
    print(X, Y, Z)

    X = ['0xf', '0x30', '0x6c', '0xc4']
    Y = Quantizer.hex_to_dec(X, 8, 0, False)
    print(Y)

    X = ['0xe7', '0xee', '0xf7', '0xfe']
    Y = Quantizer.hex_to_dec(X, 8, 2, True)
    print(Y)

    X = ['0b11100111', '0b11101110', '0b11110111', '0b11111110']
    Y = Quantizer.bin_to_dec(X, 8, 2, True)
    print(Y)

