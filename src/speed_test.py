from data import get_Cy
from scipy.interpolate import pchip_interpolate, RegularGridInterpolator
import timeit


def test(x=0.2):
    cy = get_Cy(x, 0, 0, 6, 0, 300, 3.5, 0)  # [-2.34376138]
    return cy


print(timeit.timeit("test()", number=500, setup="from __main__ import test"))
