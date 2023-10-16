import numpy as np
from numpy import cos as c
from numpy import sin as s

# Base Rotation DCMs
def R3(t):
    return np.array([[c(t), s(t), 0],
                     [-1*s(t), c(t), 0],
                     [0, 0, 1]])
def R2(t):
    return np.array([[c(t), 0, -s(t)],
                     [0, 1, 0],
                     [s(t), 0, c(t)]])
def R1(t):
    return np.array([[1, 0, 0],
                     [0, c(t), s(t)],
                     [0,-1*s(t), c(t)]])

# EULER ANGLES
def EAtoDCM(ax1, ax2, ax3, ang1, ang2, ang3):
    # Find sequence
    match ax1:
        case 1:
            rot1 = R1(ang1)
        case 2:
            rot1 = R2(ang1)
        case 3:
            rot1 = R3(ang1)
    match ax2:
        case 1:
            rot2 = R1(ang1)
        case 2:
            rot2 = R2(ang1)
        case 3:
            rot2 = R3(ang1)
    match ax3:
        case 1:
            rot3 = R1(ang1)
        case 2:
            rot3 = R2(ang1)
        case 3:
            rot3 = R3(ang1)

    # compute DCM
    DCM = np.matmul(rot3, np.matmul(rot2, rot1))
    return DCM
