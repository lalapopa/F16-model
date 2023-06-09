import numpy as np
from scipy.interpolate import pchip_interpolate

import F16model.data._aerodynamics as aerodynamics
from F16model.utils.interpolator import Interpolator


def get_Cy(alpha, beta, fi, dnos, Wz, V, ba, sb, interp_method="linear"):
    interpCy = Interpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cy1,
        method=interp_method,
    )
    Cy = interpCy.get_value((fi, alpha, beta), smooth=1.0 - 10**-6)
    Cy0 = interpCy.get_value((0, alpha, beta), smooth=1.0 - 10**-6)
    interpCy_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cy_nos1,
        method=interp_method,
    )
    Cy_nos = interpCy_nos.get_value((alpha, beta), smooth=1.0 - 10**-6)
    Cywz = pchip_interpolate(
        aerodynamics.alpha1, aerodynamics.Cywz1, alpha
    ) + pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCywz_nos1, alpha) * (
        dnos / np.radians(25)
    )
    dCy_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dCy_sb1, alpha)
    dCy_nos = Cy_nos - Cy0
    return (
        Cy
        + dCy_nos * (dnos / np.radians(25))
        + Cywz * ((Wz * ba) / (2 * V))
        + dCy_sb * (sb / np.radians(60))
    )


def get_Cx(alpha, beta, fi, dnos, Wz, V, ba, sb, interp_method="linear"):
    interpCx = Interpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cx1,
        method=interp_method,
    )
    Cx = interpCx.get_value((fi, alpha, beta))
    Cx0 = interpCx.get_value((0, alpha, beta))
    interpCx_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cx_nos1,
        method=interp_method,
    )
    Cx_nos = interpCx_nos.get_value((alpha, beta))
    Cxwz = pchip_interpolate(
        aerodynamics.alpha1, aerodynamics.Cxwz1, alpha
    ) + pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCxwz_nos1, alpha) * (
        dnos / np.radians(25)
    )
    dCx_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dCx_sb1, alpha)
    dCx_nos = Cx_nos - Cx0
    return (
        Cx
        + dCx_nos * (dnos / np.radians(25))
        + Cxwz * ((Wz * ba) / (2 * V))
        + dCx_sb * (sb / np.radians(60))
    )


def get_Cz(alpha, beta, drn, dail, dnos, Wx, Wy, V, l, interp_method="linear"):
    interpCz = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cz1,
        method=interp_method,
    )
    Cz = interpCz.get_value((alpha, beta))
    interpCz_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cz_nos1,
        method=interp_method,
    )
    Cz_nos = interpCz_nos.get_value((alpha, beta))
    interpCzdel = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Czdel20,
        method=interp_method,
    )
    Czdel = interpCzdel.get_value((alpha, beta))
    interpCzdel_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Czdel20_nos,
        method=interp_method,
    )
    Czdel_nos = interpCzdel_nos.get_value((alpha, beta))
    interpCzdrn = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Czdrn30,
        method=interp_method,
    )
    Czdrn = interpCzdrn.get_value((alpha, beta))
    Czwx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.Czwx1, alpha)
    Czwx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCzwx_nos1, alpha)
    Czwy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.Czwy1, alpha)
    Czwy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dCzwy_nos1, alpha)
    Czwx = Czwx1 + Czwx2 * (dnos / np.radians(25))
    Czwy = Czwy1 + Czwy2 * (dnos / np.radians(25))
    dCz_nos = Cz_nos - Cz
    dCzdel = Czdel - Cz
    dCzdel_nos = Czdel_nos - Cz_nos - dCzdel
    dCzdrn = Czdrn - Cz
    return (
        Cz
        + dCz_nos * (dnos / np.radians(25))
        + (dCzdel + dCzdel_nos * (dnos / np.radians(25))) * (dail / np.radians(20))
        + dCzdrn * (drn / np.radians(-30))
        + Czwx * ((Wx * l) / (2 * V))
        + Czwy * ((Wy * l) / (2 * V))
    )


def get_Mx(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l, interp_method="linear"):
    interpmx = Interpolator(
        (aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mx1,
        method=interp_method,
    )
    mx = interpmx.get_value((fi, alpha, beta))
    mx0 = interpmx.get_value((0, alpha, beta))
    interpmx_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mx_nos1,
        method=interp_method,
    )
    mx_nos = interpmx_nos.get_value((alpha, beta))
    interpmxdel = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mxdel20,
        method=interp_method,
    )
    mxdel = interpmxdel.get_value((alpha, beta))
    interpmxdel_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mxdel20_nos,
        method=interp_method,
    )
    mxdel_nos = interpmxdel_nos.get_value((alpha, beta))
    interpmxdrn = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mxdrn30,
        method=interp_method,
    )
    mxdrn = interpmxdrn.get_value((alpha, beta))
    dmxbt = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmxbt1, alpha)
    mxwx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mxwx1, alpha)
    mxwx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmxwx_nos1, alpha)
    mxwy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mxwy1, alpha)
    mxwy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmxwy_nos1, alpha)
    mxwx = mxwx1 + mxwx2 * (dnos / np.radians(25))
    mxwy = mxwy1 + mxwy2 * (dnos / np.radians(25))
    dmx_nos = mx_nos - mx0
    dmxdel = mxdel - mx0
    dmxdel_nos = mxdel_nos - mx_nos - dmxdel
    dmxdrn = mxdrn - mx0
    return (
        mx
        + dmx_nos * (dnos / np.radians(25))
        + (dmxdel + dmxdel_nos * (dnos / np.radians(25))) * (dail / np.radians(20))
        + dmxdrn * (drn / np.radians(-30))
        + mxwx * ((Wx * l) / (2 * V))
        + mxwy * ((Wy * l) / (2 * V))
        + dmxbt * beta
    )


def get_My(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l, interp_method="linear"):
    interpmy = Interpolator(
        (aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.my1,
        method=interp_method,
    )
    my = interpmy.get_value((fi, alpha, beta))
    my0 = interpmy.get_value((0, alpha, beta))
    interpmy_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.my_nos1,
        method=interp_method,
    )
    my_nos = interpmy_nos.get_value((alpha, beta))
    interpmydel = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mydel20,
        method=interp_method,
    )
    mydel = interpmydel.get_value((alpha, beta))
    interpmydel_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mydel20_nos,
        method=interp_method,
    )
    mydel_nos = interpmydel_nos.get_value((alpha, beta))
    interpmydrn = Interpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mydrn30,
        method=interp_method,
    )
    mydrn = interpmydrn.get_value((alpha, beta))
    dmybt = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmybt1, alpha)
    mywx1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mywx1, alpha)
    mywx2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmywx_nos1, alpha)
    mywy1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mywy1, alpha)
    mywy2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmywy_nos1, alpha)
    mywx = mywx1 + mywx2 * (dnos / np.radians(25))
    mywy = mywy1 + mywy2 * (dnos / np.radians(25))
    dmy_nos = my_nos - my0
    dmydel = mydel - my0
    dmydel_nos = mydel_nos - my_nos - dmydel
    dmydrn = mydrn - my0
    return (
        my
        + dmy_nos * (dnos / np.radians(25))
        + (dmydel + dmydel_nos * (dnos / np.radians(25))) * (dail / np.radians(20))
        + dmydrn * (drn / np.radians(-30))
        + mywx * ((Wx * l) / (2 * V))
        + mywy * ((Wy * l) / (2 * V))
        + dmybt * beta
    )


def get_Mz(alpha, beta, fi, dnos, Wz, V, ba, sb, interp_method="linear"):
    interpmz = Interpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mz1,
        method=interp_method,
    )
    mz = interpmz.get_value((fi, alpha, beta))
    mz0 = interpmz.get_value((0, alpha, beta))
    interpmz_nos = Interpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mz_nos1,
        method=interp_method,
    )
    mz_nos = interpmz_nos.get_value((alpha, beta))
    inerpdmz = Interpolator(
        aerodynamics.alpha1, aerodynamics.dmz1, method=interp_method
    )
    dmz = inerpdmz.get_value(alpha)
    interpdmz_ds = Interpolator(
        (aerodynamics.alpha1, aerodynamics.fi3),
        aerodynamics.dmz_ds1,
        method=interp_method,
    )
    dmz_ds = interpdmz_ds.get_value((alpha, fi))
    mzwz1 = pchip_interpolate(aerodynamics.alpha1, aerodynamics.mzwz1, alpha)
    mzwz2 = pchip_interpolate(aerodynamics.alpha2, aerodynamics.dmzwz_nos1, alpha)
    dmz_sb = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmz_sb1, alpha)
    eta_fi = pchip_interpolate(aerodynamics.fi1, aerodynamics.eta_fi1, fi)
    mzwz = mzwz1 + mzwz2 * (dnos / np.radians(25))
    dmz_nos = mz_nos - mz0
    return (
        mz * eta_fi
        + dmz_nos * (dnos / np.radians(25))
        + dmz
        + mzwz * ((Wz * ba) / (2 * V))
        + dmz_sb * (sb / np.radians(60))
        + dmz_ds
    )

