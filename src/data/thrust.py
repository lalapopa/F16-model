from scipy.interpolate import pchip_interpolate, RegularGridInterpolator

import data._engine as engine


def get_thrust(H, M, Pa):
    interpP = RegularGridInterpolator(
        (engine.Pa1, engine.H1, engine.M1),
        engine.Pt1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interpP((Pa, H, M))
