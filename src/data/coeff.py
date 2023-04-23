import data._aerodynamics as aerodynamics
import numpy as np
from scipy.interpolate import pchip_interpolate, RegularGridInterpolator


def get_Cy(alpha, beta, fi, dnos, Wz, V, ba, sb):
    interpCy = RegularGridInterpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cy1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cy = interpCy((fi, alpha, beta))
    Cy0 = interpCy((0, alpha, beta))
    interpCy_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cy_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cy_nos = interpCy_nos((alpha, beta))
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


def get_Cx(alpha, beta, fi, dnos, Wz, V, ba, sb):
    interpCx = RegularGridInterpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cx1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cx = interpCx((fi, alpha, beta))
    Cx0 = interpCx((0, alpha, beta))
    interpCx_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cy_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cx_nos = interpCx_nos((alpha, beta))
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


def get_Cz(alpha, beta, drn, dail, dnos, Wx, Wy, V, l):
    interpCz = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Cz1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cz = interpCz((alpha, beta))
    interpCz_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Cz_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Cz_nos = interpCz_nos((alpha, beta))
    interpCzdel = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Czdel20,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Czdel = interpCzdel((alpha, beta))
    interpCzdel_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.Czdel20_nos,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Czdel_nos = interpCzdel_nos((alpha, beta))
    interpCzdrn = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.Czdrn30,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    Czdrn = interpCzdrn((alpha, beta))
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


def get_Mx(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l):
    interpmx = RegularGridInterpolator(
        (aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mx1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mx = interpmx((fi, alpha, beta))
    interpmx0 = RegularGridInterpolator(
        (aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mx1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mx0 = interpmx0((0, alpha, beta))
    interpmx_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mx_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mx_nos = interpmx_nos((alpha, beta))
    interpmxdel = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mxdel20,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mxdel = interpmxdel((alpha, beta))
    interpmxdel_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mxdel20_nos,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mxdel_nos = interpmxdel_nos((alpha, beta))
    interpmxdrn = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mxdrn30,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mxdrn = interpmxdrn((alpha, beta))
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


def get_My(alpha, beta, fi, drn, dail, dnos, Wx, Wy, V, l):
    interpmy = RegularGridInterpolator(
        (aerodynamics.fi2, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.my1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    my = interpmy((fi, alpha, beta))
    my0 = interpmy((0, alpha, beta))
    interpmy_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.my_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    my_nos = interpmy_nos((alpha, beta))
    interpmydel = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mydel20,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mydel = interpmydel((alpha, beta))
    interpmydel_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mydel20_nos,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mydel_nos = interpmydel_nos((alpha, beta))
    interpmydrn = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mydrn30,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mydrn = interpmydrn((alpha, beta))
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


def get_Mz(alpha, beta, fi, dnos, Wz, V, ba, sb):
    interpmz = RegularGridInterpolator(
        (aerodynamics.fi1, aerodynamics.alpha1, aerodynamics.beta1),
        aerodynamics.mz1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mz = interpmz((fi, alpha, beta))
    mz0 = interpmz((0, alpha, beta))
    interpmz_nos = RegularGridInterpolator(
        (aerodynamics.alpha2, aerodynamics.beta1),
        aerodynamics.mz_nos1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    mz_nos = interpmz_nos((alpha, beta))
    dmz = pchip_interpolate(aerodynamics.alpha1, aerodynamics.dmz1, -0)
    interpdmz_ds = RegularGridInterpolator(
        (aerodynamics.alpha1, aerodynamics.fi3),
        aerodynamics.dmz_ds1,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    dmz_ds = interpdmz_ds((alpha, fi))
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


cy = get_Cy(1, 0.1, 0.1, 0, 0.5, 500, 3.45, 0)
cx = get_Cx(1, 0.1, 0.1, 0, 0.5, 500, 3.45, 0)
cz = get_Cz(1, 0.1, 0.1, 0.05, 0, 0.02, 0.04, 500, 5.4)
mx = get_Mx(0.1, 0.1, 0.1, 0.04, 0.0125, 0, 0.0034, 0.932, 432, 5.4)
my = get_My(0.1, 0.1, 0.1, 0.04, 0.0125, 0, 0.0034, 0.932, 432, 5.4)
mz = get_Mz(0.1, 0.13, 0.032, 0, 0.32, 343, 5.4, 100)
print(f"cy = {cy} ")
print(f"cx = {cx} ")
print(f"cz = {cz} ")
print(f"mx = {mx} ")
print(f"my = {my} ")
print(f"mz = {mz} ")
