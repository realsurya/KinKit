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

# Euler Angle to DCM function
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
            rot2 = R1(ang2)
        case 2:
            rot2 = R2(ang2)
        case 3:
            rot2 = R3(ang2)
    match ax3:
        case 1:
            rot3 = R1(ang3)
        case 2:
            rot3 = R2(ang3)
        case 3:
            rot3 = R3(ang3)

    # compute DCM
    DCM = np.matmul(rot3, np.matmul(rot2, rot1))
    return DCM

# DCM to Euler Angle Functions
def DCMtoEA121(DCM):
    return np.array([np.arctan2(DCM[1][0], DCM[2][0]),
                     np.arccos(DCM[0][0]),
                     np.arctan2(DCM[0][1], -DCM[0][2])])
def DCMtoEA123(DCM):
    return np.array([np.arctan2(DCM[1][2], DCM[2][2]),
                     -np.arcsin(DCM[0][2]),
                     np.arctan2(DCM[0][1], DCM[0][0])])
def DCMtoEA131(DCM):
    return np.array([np.arctan2(DCM[2][0], -DCM[1][0]),
                     np.arccos(DCM[0][0]),
                     np.arctan2(DCM[0][2], DCM[0][1])])
def DCMtoEA132(DCM):
    return np.array([np.arctan2(-DCM[2][1], DCM[1][1]),
                     np.arcsin(DCM[0][1]),
                     np.arctan2(-DCM[0][2], DCM[0][0])])
def DCMtoEA212(DCM):
    return np.array([np.arctan2(DCM[0][1], -DCM[2][1]),
                     np.arccos(DCM[1][1]),
                     np.arctan2(DCM[1][0], DCM[1][2])])
def DCMtoEA213(DCM):
    return np.array([np.arctan2(-DCM[0][2], DCM[2][2]),
                     np.arcsin(DCM[1][2]),
                     np.arctan2(-DCM[1][0], DCM[1][1])])
def DCMtoEA231(DCM):
    return np.array([np.arctan2(DCM[2][0], DCM[0][0]),
                     -np.arcsin(DCM[1][0]),
                     np.arctan2(DCM[1][2], DCM[1][1])])
def DCMtoEA232(DCM): #IN PROG
    return np.array([np.arctan2(DCM[2][0], DCM[0][0]),
                     -np.arcsin(DCM[1][0]),
                     np.arctan2(DCM[1][2], DCM[1][1])])