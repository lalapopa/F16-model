from math import atan, asin, sin, cos, sin, sqrt


def body2wind(Vx, Vy, Vz):
    V = sqrt(Vx**2 + Vy**2 + Vz**2)
    alpha = -atan(Vy / Vx)
    beta = asin(Vz / V)
    return V, alpha, beta


def wind2body(V, alpha, beta):
    Vx = V * cos(alpha) * cos(beta)
    Vy = -V * sin(alpha) * cos(beta)
    Vz = V * sin(beta)
    return (Vx, Vy, Vz)
