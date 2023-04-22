from csaps import csaps

import data._engine as engine


def get_thrust(H, M, Pa):
    P = csaps(
        [engine.Pa1, engine.H1, engine.M1],
        engine.Pt1,
        [Pa, H, M],
        smooth=1.0 - 10**-5,
    ).flatten()
    return P
