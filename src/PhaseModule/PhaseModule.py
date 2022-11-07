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

    @staticmethod
    def phase_wrapping_single_point(ang):
        while abs(ang) > np.pi:
            if ang > np.pi:
                ang = ang - 2 * np.pi
            elif ang < -np.pi:
                ang = ang + 2 * np.pi

        return ang

    # if abs(ang) > np.pi:
    #     mul = int((abs(ang) - np.pi) / 2 / np.pi) + 1
    #     if ang > 0:
    #         ang = ang - mul * 2 * np.pi
    #     elif ang < 0:
    #         ang = ang + mul * 2 * np.pi

    @staticmethod
    def phase_unwrapping(ang_np):
        num_point = len(ang_np)

        ang_unwrap = ang_np
        for k in range(1, num_point):
            diff = ang_np[k] - ang_np[k - 1]
            if diff > np.pi:
                ang_unwrap[k:] = ang_unwrap[k:] - 2 * np.pi
            elif diff < -np.pi:
                ang_unwrap[k:] = ang_unwrap[k:] + 2 * np.pi

        return ang_unwrap

    @staticmethod
    def phase_estimator(i_np, q_np):
        assert len(i_np) == len(q_np)
        num_point = len(i_np)
        ang_np = np.zeros(num_point, dtype=float)
        for k in range(num_point):
            ang_np[k] = PhaseModule.phase_estimator_single_point(i_np[k], q_np[k])

        return ang_np

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


if __name__ == '__main__':
    import matplotlib.font_manager as fm
    import os
    import sys

    font_times = fm.FontProperties(family='Times New Roman', stretch=0)
    current_path = os.path.dirname(__file__)

    I = 1
    Q = 1
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    I = 1
    Q = -1
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    I = -1
    Q = 1
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    I = -1
    Q = -1
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    I = -1
    Q = 0
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    I = 0
    Q = -1
    x = I + 1j * Q
    print(np.angle(x))
    print(PhaseModule.phase_estimator_single_point(I, Q))

    print(PhaseModule.phase_wrapping_single_point(3 * np.pi))
    print(PhaseModule.phase_wrapping_single_point(np.pi + 0.1))
    print(PhaseModule.phase_wrapping_single_point(-2 * np.pi + 0.1))
    print(PhaseModule.phase_wrapping_single_point(-2 * np.pi - 0.1))
    print(PhaseModule.phase_wrapping_single_point(-3 * np.pi))
    print(np.fix([-3.5, 3.5]))