import numpy as np


def simulate(x):
    time = np.linspace(450, 500, 500)

    x0, h1, x1, h2 = x
    stddev = 10
    # first gaussian peak
    peak1 = h1 * np.exp(-0.5 * ((time - x0) / stddev) ** 2)
    # second gaussian peak
    peak2 = h2 * np.exp(-0.5 * ((time - x1) / stddev) ** 2)

    desorption = peak1 + peak2

    return desorption, time
