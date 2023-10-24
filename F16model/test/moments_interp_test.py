from scipy.interpolate import RegularGridInterpolator
import numpy as np


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
data = f(xg, yg, zg)

x_int = (x, y, z)
interp = RegularGridInterpolator(x_int, data, bounds_error=True, fill_value=False)

input_values = (0.9, 6.2, 100)

clip input value

clipped_input = np.clip(
    input_values, a_min=[np.min(i) for i in x_int], a_max=[np.max(i) for i in x_int]
)


print(clipped_input)

print(interp(clipped_input))
