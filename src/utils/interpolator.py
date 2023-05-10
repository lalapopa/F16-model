from scipy.interpolate import pchip_interpolate, RegularGridInterpolator, interp1d
from csaps import csaps


class Interpolator:
    def __init__(self, x_values, y_values, method="linear"):
        self.x = x_values
        self.y = y_values
        self.method = method

    def get_value(self, x_int, smooth=1.0 - 10**-5):
        if self.method == "linear":
            if isinstance(self.x, tuple):
                return self._interp_linear(x_int)
            else:
                return self._interp_linear1d(x_int)
        if self.method == "csaps":
            return self._interp_csaps(x_int, smooth)

    def _interp_linear(self, x_int):
        interp_y = RegularGridInterpolator(
            self.x,
            self.y,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        return interp_y(x_int)

    def _interp_linear1d(self, x_int):
        interp_y = interp1d(
            self.x,
            self.y,
            kind="linear",
            bounds_error=False,
            fill_value=None,
        )
        return interp_y(x_int)

    def _interp_csaps(self, x_int, smooth):
        return csaps(
            self.x,
            self.y,
            x_int,
            smooth=smooth,
        ).flatten()
