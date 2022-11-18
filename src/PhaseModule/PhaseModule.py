import numpy as np


class PhaseModule:
    name = 'Phase Module'

    def __init__(self):
        pass

    @staticmethod
    def phase_wrapping(ang_np):
        num_point = len(ang_np)
        ang_wrap = np.zeros(num_point, dtype=float)
        for k in range(num_point):
            ang_wrap[k] = PhaseModule.phase_wrapping_single_point(ang_np[k])

        return ang_wrap

    """ -pi, pi
    @staticmethod
    def phase_wrapping_single_point(ang):
        while abs(ang) > np.pi:
            if ang > np.pi:
                ang = ang - 2 * np.pi
            elif ang < -np.pi:
                ang = ang + 2 * np.pi

        return ang
    """

    """ -0.5, 0.5
    @staticmethod
    def phase_wrapping_single_point(ang):
        while ang >= 0.5 or ang < -0.5:
            if ang >= 0.5:
                ang = ang - 1
            elif ang < -0.5:
                ang = ang + 1
    
        return ang
    """

    @staticmethod
    def phase_wrapping_single_point(ang):
        while ang >= 1 or ang < 0:
            if ang >= 1:
                ang = ang - 1
            elif ang < 0:
                ang = ang + 1

        ang = PhaseModule.zero_one(ang)

        return ang

    """ -pi, pi
    @staticmethod
    def phase_unwrapping(ang_np):
        num_point = len(ang_np)

        ang_unwrap = ang_np.copy()  # a big hole
        for k in range(1, num_point):
            diff = ang_np[k] - ang_np[k - 1]
            if diff > np.pi:
                ang_unwrap[k:] = ang_unwrap[k:] - 2 * np.pi
            elif diff < -np.pi:
                ang_unwrap[k:] = ang_unwrap[k:] + 2 * np.pi

        return ang_unwrap
    """

    @staticmethod
    def phase_unwrapping(ang_np):
        num_point = len(ang_np)

        ang_unwrap = ang_np.copy()   # a big hole
        for k in range(1, num_point):
            diff = ang_np[k] - ang_np[k - 1]
            if diff > 0.5:
                ang_unwrap[k:] = ang_unwrap[k:] - 1
            elif diff < -0.5:
                ang_unwrap[k:] = ang_unwrap[k:] + 1

        return ang_unwrap

    @staticmethod
    def phase_estimator(i_np, q_np):
        assert len(i_np) == len(q_np)
        num_point = len(i_np)
        ang_np = np.zeros(num_point, dtype=float)
        for k in range(num_point):
            # ang_np[k] = PhaseModule.phase_estimator_single_point(i_np[k], q_np[k])
            # ang_np[k] = PhaseModule.phase_estimator_single_point_cordic(i_np[k], q_np[k], 12)
            ang_np[k] = PhaseModule.phase_estimator_single_point_cordic_normalized(i_np[k], q_np[k], 16)

        return ang_np

    """ -pi, pi
    @staticmethod
    def phase_estimator_single_point(i, q):
        if i == 0:
            if q > 0:
                ang = np.pi / 2
            elif q < 0:
                ang = -np.pi / 2
            elif q == 0:
                ang = 0.0
        elif i != 0:
            ang = np.arctan(q / i)
            if (i < 0) and (q >= 0):
                ang = np.pi + ang
            elif i < 0 and q < 0:
                ang = ang - np.pi
        return ang
    """

    """ -0.5, 0.5
    @staticmethod 
    def phase_estimator_single_point(i, q):
        if i == 0:
            if q > 0:
                ang = np.pi / 2
            elif q < 0:
                ang = -np.pi / 2
            elif q == 0:
                ang = 0.0
        elif i != 0:
            ang = np.arctan(q / i)
            if (i < 0) and (q >= 0):
                ang = np.pi + ang
            elif i < 0 and q < 0:
                ang = ang - np.pi

        ang = ang / 2 / np.pi
        return ang
    """

    """ 0, 1"""
    @staticmethod
    def phase_estimator_single_point(i, q):
        if i == 0:
            if q > 0:
                ang = 0.25
            elif q < 0:
                ang = 0.75
            else:
                ang = 0.0
        elif i != 0:
            ang = np.arctan(q / i) / 2 / np.pi
            if (i > 0) and (q >= 0):
                pass
            elif i < 0:
                ang = ang + 0.5
            else:
                ang = ang + 1

            ang = PhaseModule.zero_one(ang)
        return ang

    """ 0, 1"""
    @staticmethod
    def phase_estimator_single_point_cordic(i, q, n):
        n_axis = np.arange(n)
        tan_table = (1/2) ** n_axis
        angle_lut = np.arctan(tan_table)
        amp = np.ones(n, dtype=float)
        for k in range(n):
            amp[k] = 1/np.sqrt(1 + 2 ** (-2 * k))

        mag = np.sqrt(i ** 2 + q ** 2)
        if mag == 0:
            x = i
            y = q
        else:
            x = i / mag
            y = q / mag

        angle_accumulate = 0

        """ CorDic approach"""
        if (x == 0) and (y == 0):
            radian_out = 0
        else:
            if x > 0:
                phase_shift = 0
            elif y < 0:
                phase_shift = -np.pi
                x = -x
                y = -y
            else:
                phase_shift = np.pi
                x = -x
                y = -y

            for k in range(n):
                x_tmp = x
                if y < 0:
                    x = amp[k] * (x_tmp - y * 2 ** (-k))
                    y = amp[k] * (y + x_tmp * 2 ** (-k))
                    angle_accumulate = angle_accumulate - angle_lut[k]
                else:
                    x = amp[k] * (x_tmp + y * 2 ** (-k))
                    y = amp[k] * (y - x_tmp * 2 ** (-k))
                    angle_accumulate = angle_accumulate + angle_lut[k]

            radian_out = angle_accumulate + phase_shift

        angle_out = radian_out
        angle_out = PhaseModule.phase_to_zero_one(angle_out)

        return angle_out

    @staticmethod
    def phase_estimator_single_point_cordic_normalized(i, q, n):
        n_axis = np.arange(n)
        tan_table = (1/2) ** n_axis
        angle_lut = np.arctan(tan_table) / np.pi / 2
        amp = np.ones(n, dtype=float)
        for k in range(n):
            amp[k] = 1/np.sqrt(1 + 2 ** (-2 * k))

        mag = np.sqrt(i ** 2 + q ** 2)
        if mag == 0:
            x = i
            y = q
        else:
            x = i / mag
            y = q / mag

        """ CorDic approach"""
        if (x == 0) and (y == 0):
            radian_out = 0
        else:
            if x > 0:
                if y <= 0:
                    phase_shift = 1
                else:
                    phase_shift = 0
            else:
                phase_shift = 0.5
                x = -x
                y = -y

            angle_accumulate = phase_shift

            for k in range(n):
                x_tmp = x
                if y < 0:
                    x = amp[k] * (x_tmp - y * 2 ** (-k))
                    y = amp[k] * (y + x_tmp * 2 ** (-k))
                    angle_accumulate = angle_accumulate - angle_lut[k]
                else:
                    x = amp[k] * (x_tmp + y * 2 ** (-k))
                    y = amp[k] * (y - x_tmp * 2 ** (-k))
                    angle_accumulate = angle_accumulate + angle_lut[k]

            radian_out = angle_accumulate

        angle_out = radian_out

        return angle_out

    @staticmethod
    def zero_one(ang):
        if (ang > 0.9999) and (ang < 1.0001):
            ang = 0
        if (ang > - 0.0001) and (ang < 0.0001):
            ang = 0
        return ang

    @staticmethod
    def phase_to_zero_one(ang):
        val = ang / 2 / np.pi
        if val < 0:
            val = val + 1
        return val


if __name__ == '__main__':
    import matplotlib.font_manager as fm
    import os

    font_times = fm.FontProperties(family='Times New Roman', stretch=0)
    current_path = os.path.dirname(__file__)

    I = 1
    Q = 1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 16))


    I = 1
    Q = -1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = -1
    Q = 1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = -1
    Q = -1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = -1
    Q = 0
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = 0
    Q = -1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = 1
    Q = 0
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = 0
    Q = 1
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    I = 0.4
    Q = 0.9
    COMPLEX = I + 1j * Q
    print("==========================")
    print(PhaseModule.phase_to_zero_one(np.angle(COMPLEX)))
    print(PhaseModule.phase_estimator_single_point(I, Q))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 12))
    print(PhaseModule.phase_estimator_single_point_cordic(I, Q, 14))
    print(PhaseModule.phase_estimator_single_point_cordic_normalized(I, Q, 12))

    """ test -pi, pi
    print(PhaseModule.phase_wrapping_single_point(3 * np.pi))
    print(PhaseModule.phase_wrapping_single_point(np.pi + 0.1))
    print(PhaseModule.phase_wrapping_single_point(-2 * np.pi))
    print(PhaseModule.phase_wrapping_single_point(-2 * np.pi + 0.1))
    print(PhaseModule.phase_wrapping_single_point(-2 * np.pi - 0.1))
    print(PhaseModule.phase_wrapping_single_point(-3 * np.pi))
    """

    """ test 0, 1 """
    print("==========================")
    print("Test phase wrapping:")
    print(PhaseModule.phase_wrapping_single_point(3.3))
    print(PhaseModule.phase_wrapping_single_point(-3.3))
    print(PhaseModule.phase_wrapping_single_point(-0.1))
    print(PhaseModule.phase_wrapping_single_point(0.1))
    print(PhaseModule.phase_wrapping_single_point(0.9))
    print(PhaseModule.phase_wrapping_single_point(1))
