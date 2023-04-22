def engine_power_level(Pa, throttle_act):
    """Find speed difference of thrust level with respect to:
    :Pa: current thrust %
    :throttle_act: control signal
    :returns: derivative Pa
    """
    if throttle_act <= 0.77:
        Pc = 64.94 * throttle_act
    else:
        Pc = 217.38 * throttle_act - 117.38

    if Pc >= 50 and Pa < 50:
        Pc = 60
    elif Pc < 50 and Pa >= 50:
        Pc = 40

    dP = Pc - Pa

    if dP <= 25:
        w_eng = 1.0
    elif dP >= 50:
        w_eng = 0.1
    else:
        w_eng = 1.9 - 0.036 * dP

    if Pa >= 50:
        w_eng = 5

    return w_eng * dP
