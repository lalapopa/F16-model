import data._aerodynamics as aerodynamics
from math import radians

from csaps import csaps
from scipy.interpolate import pchip_interpolate

smooth_6 = 1.0 - pow(10, -6)
smooth_5 = 1.0 - pow(10, -5)


def get_Cy(alpha, beta, fi, dnos, Wz, V, ba, sb):
    Cy = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Cy1,
        [fi, alpha, beta],
        smooth=smooth_6,
    ).flatten()
    Cy0 = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Cy1,
        [0, alpha, beta],
        smooth=smooth_6,
    ).flatten()
    Cy_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.Cy_nos1,
        [alpha, beta],
        smooth=smooth_6,
    ).flatten()
    Cywz = pchip_interpolate(
        aerodynamics.alpha1, aerodynamics.Cywz1, alpha
    ) + pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCywz_nos1, alpha) * (
        dnos / radians(25)
    )
    dCy_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dCy_sb1, alpha)
    dCy_nos = Cy_nos - Cy0
    return (
        Cy
        + dCy_nos * (dnos / radians(25))
        + Cywz * ((Wz * ba) / (2 * V))
        + dCy_sb * (sb / radians(60))
    )


def get_Cx(alpha, beta, fi, dnos, Wz, V, ba, sb):
    Cx = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Cx1,
        [fi, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Cx0 = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Cx1,
        [0, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Cx_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.Cy_nos1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Cxwz = pchip_interpolate(
        aerodynamics.alpha1, aerodynamics.Cxwz1, alpha
    ) + pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCxwz_nos1, alpha) * (
        dnos / radians(25)
    )
    dCx_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dCx_sb1, alpha)
    dCx_nos = Cx_nos - Cx0
    return (
        Cx
        + dCx_nos * (dnos / radians(25))
        + Cxwz * ((Wz * ba) / (2 * V))
        + dCx_sb * (sb / radians(60))
    )


def get_Cz(alpha, beta, drn, dail, dnos, Wx, Wy, V, l):
    Cz = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Cz1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Cz_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.Cz_nos1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Czdel = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Czdel20,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Czdel_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.Czdel20_nos,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Czdrn = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.Czdrn30,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    Czwx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.Czwx1, alpha)
    Czwx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCzwx_nos1, alpha)
    Czwy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.Czwy1, alpha)
    Czwy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCzwy_nos1, alpha)
    Czwx = Czwx1 + Czwx2 * (dnos / radians(25))
    Czwy = Czwy1 + Czwy2 * (dnos / radians(25))
    dCz_nos = Cz_nos - Cz
    dCzdel = Czdel - Cz
    dCzdel_nos = Czdel_nos - Cz_nos - dCzdel
    dCzdrn = Czdrn - Cz
    return (
        Cz
        + dCz_nos * (dnos / radians(25))
        + (dCzdel + dCzdel_nos * (dnos / radians(25))) * (dail / radians(20))
        + dCzdrn * (drn / radians(-30))
        + Czwx * ((Wx * l) / (2 * V))
        + Czwy * ((Wy * l) / (2 * V))
    )


def get_Mx(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l):
    mx = csaps(
        [aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mx1,
        [fi, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mx0 = csaps(
        [aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mx1,
        [0, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mx_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.mx_nos1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mxdel = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mxdel20,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mxdel_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.mxdel20_nos,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mxdrn = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mxdrn30,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    dmxbt = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmxbt1, alpha)
    mxwx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mxwx1, alpha)
    mxwx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmxwx_nos1, alpha)
    mxwy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mxwy1, alpha)
    mxwy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmxwy_nos1, alpha)
    mxwx = mxwx1 + mxwx2 * (dnos / radians(25))
    mxwy = mxwy1 + mxwy2 * (dnos / radians(25))
    dmx_nos = mx_nos - mx0
    dmxdel = mxdel - mx0
    dmxdel_nos = mxdel_nos - mx_nos - dmxdel
    dmxdrn = mxdrn - mx0
    return (
        mx
        + dmx_nos * (dnos / radians(25))
        + (dmxdel + dmxdel_nos * (dnos / radians(25))) * (dail / radians(20))
        + dmxdrn * (drn / radians(-30))
        + mxwx * ((Wx * l) / (2 * V))
        + mxwy * ((Wy * l) / (2 * V))
        + dmxbt * beta
    )


def get_My(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l):
    my = csaps(
        [aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.my1,
        [fi, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    my0 = csaps(
        [aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.my1,
        [0, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    my_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.my_nos1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mydel = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mydel20,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mydel_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.mydel20_nos,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mydrn = csaps(
        [aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mydrn30,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    dmybt = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmybt1, alpha)
    mywx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mywx1, alpha)
    mywx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmywx_nos1, alpha)
    mywy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mywy1, alpha)
    mywy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmywy_nos1, alpha)
    mywx = mywx1 + mywx2 * (dnos / radians(25))
    mywy = mywy1 + mywy2 * (dnos / radians(25))
    dmy_nos = my_nos - my0
    dmydel = mydel - my0
    dmydel_nos = mydel_nos - my_nos - dmydel
    dmydrn = mydrn - my0
    return (
        my
        + dmy_nos * (dnos / radians(25))
        + (dmydel + dmydel_nos * (dnos / radians(25))) * (dail / radians(20))
        + dmydrn * (drn / radians(-30))
        + mywx * ((Wx * l) / (2 * V))
        + mywy * ((Wy * l) / (2 * V))
        + dmybt * beta
    )


def get_Mz(alpha, beta, fi, dnos, Wz, V, ba, sb):
    mz = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mz1,
        [fi, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mz0 = csaps(
        [aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1],
        aerodynamics.mz1,
        [0, alpha, beta],
        smooth=smooth_5,
    ).flatten()
    mz_nos = csaps(
        [aerodynamics.alpha2, aerodynamics.beta1],
        aerodynamics.mz_nos1,
        [alpha, beta],
        smooth=smooth_5,
    ).flatten()
    dmz = csaps(
        aerodynamics.alpha1,
        aerodynamics.dmz1,
        alpha,
        smooth=smooth_5,
    ).flatten()
    dmz_ds = csaps(
        [aerodynamics.alpha1, aerodynamics.fi3],
        aerodynamics.dmz_ds1,
        [alpha, fi],
        smooth=smooth_5,
    ).flatten()
    mzwz1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mzwz1, alpha)
    mzwz2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmzwz_nos1, alpha)
    dmz_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmz_sb1, alpha)
    eta_fi = pchip_interpolate(aerodynamics.fi1, aerodynamics.eta_fi1, fi)
    mzwz = mzwz1 + mzwz2 * (dnos / radians(25))
    dmz_nos = mz_nos - mz0
    return (
        mz * eta_fi
        + dmz_nos * (dnos / radians(25))
        + dmz
        + mzwz * ((Wz * ba) / (2 * V))
        + dmz_sb * (sb / radians(60))
        + dmz_ds
    )


# cy = get_Cy(3, 0.1, 0.1, 0, 0.5, 500, 3.45, 0)
# cx = get_Cx(3, 0.1, 0.1, 0, 0.5, 500, 3.45, 0)
# cz = get_Cz(3, 0.1, 0.1, 0.05, 0, 0.02, 0.04, 500, 5.4)
# mx = get_Mx(0.1, 0.1, 0.1, 0.04, 0.0125, 0, 0.0034, 0.932, 432, 5.4)
# my = get_My(0.1, 0.1, 0.1, 0.04, 0.0125, 0, 0.0034, 0.932, 432, 5.4)
# mz = get_Mz(0.1, 0.13, 0.032, 0, 0.32, 343, 5.4, 100)
