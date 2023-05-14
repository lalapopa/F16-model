import numpy as np

Pt1 = np.zeros((3, 6, 6))

H1 = np.arange(0, 50000+10000,10000)*0.3048
M1 = np.arange(0,1+0.2, 0.2)
Pa1 = np.array([0, 50, 100])

Pt1[0] = [
    [1060.0, 635.0, 60.0, -1020.0, -2700.0, -3600.0],
    [670.0, 425.0, 25.0, -710.0, -1900.0, -1400.0],
    [880.0, 690.0, 345.0, -300.0, -1300.0, -595.0],
    [1140.0, 1010.0, 755.0, 350.0, -247.0, -342.0],
    [1500.0, 1330.0, 1130.0, 910.0, 600.0, -200.0],
    [1860.0, 1700.0, 1525.0, 1360.0, 1100.0, 700.0],
]

Pt1[1] = [
    [12680.0, 12680.0, 12610.0, 12640.0, 12390.0, 11680.0],
    [9150.0, 9150.0, 9312.0, 9839.0, 10176.0, 9848.0],
    [6200.0, 6313.0, 6610.0, 7090.0, 7750.0, 8050.0],
    [3950.0, 4040.0, 4290.0, 4660.0, 5320.0, 6100.0],
    [2450.0, 2470.0, 2600.0, 2840.0, 3250.0, 3800.0],
    [1400.0, 1400.0, 1560.0, 1660.0, 1930.0, 2310.0],
]
Pt1[2] = [
    [20000.0, 21420.0, 22700.0, 24240.0, 26070.0, 28886.0],
    [15000.0, 15700.0, 16860.0, 18910.0, 21075.0, 23319.0],
    [10800.0, 11225.0, 12250.0, 13760.0, 15975.0, 18300.0],
    [7000.0, 7323.0, 8154.0, 9285.0, 11115.0, 13484.0],
    [4000.0, 4435.0, 5000.0, 5700.0, 6860.0, 8642.0],
    [2500.0, 2600.0, 2835.0, 3215.0, 3950.0, 5057.0],
]

Pt1 = Pt1 * 4.4482216


