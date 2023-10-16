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
def EAtoDCM(seq, ang1, ang2, ang3):
    ax1 = int(str(seq)[0])
    ax2 = int(str(seq)[1])
    ax3 = int(str(seq)[2])
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
def DCMtoEA(sequence, DCM):
    match sequence:
        case 121:
            return np.array([np.arctan2(DCM[1][0], DCM[2][0]),
                             np.arccos(DCM[0][0]),
                             np.arctan2(DCM[0][1], -DCM[0][2])])
        case 123:
            return np.array([np.arctan2(DCM[1][2], DCM[2][2]),
                             -np.arcsin(DCM[0][2]),
                             np.arctan2(DCM[0][1], DCM[0][0])])
        case 131 :
            return np.array([np.arctan2(DCM[2][0], -DCM[1][0]),
                             np.arccos(DCM[0][0]),
                             np.arctan2(DCM[0][2], DCM[0][1])])
        case 132:
            return np.array([np.arctan2(-DCM[2][1], DCM[1][1]),
                             np.arcsin(DCM[0][1]),
                             np.arctan2(-DCM[0][2], DCM[0][0])])
        case 212:
            return np.array([np.arctan2(DCM[0][1], -DCM[2][1]),
                             np.arccos(DCM[1][1]),
                             np.arctan2(DCM[1][0], DCM[1][2])])
        case 213:
            return np.array([np.arctan2(-DCM[0][2], DCM[2][2]),
                             np.arcsin(DCM[1][2]),
                             np.arctan2(-DCM[1][0], DCM[1][1])])
        case 231:
            return np.array([np.arctan2(DCM[2][0], DCM[0][0]),
                             -np.arcsin(DCM[1][0]),
                             np.arctan2(DCM[1][2], DCM[1][1])])
        case 232:
            return np.array([np.arctan2(DCM[2][1], DCM[0][1]),
                             np.arccos(DCM[1][1]),
                             np.arctan2(DCM[1][2], -DCM[1][0])])
        case 312:
            return np.array([np.arctan2(DCM[0][1], DCM[1][1]),
                             -np.arcsin(DCM[2][1]),
                             np.arctan2(DCM[2][0], DCM[2][2])])
        case 313:
            return np.array([np.arctan2(DCM[0][2], DCM[1][2]),
                             np.arccos(DCM[2][2]),
                             np.arctan2(DCM[2][0], -DCM[2][1])])
        case 321:
            return np.array([np.arctan2(-DCM[1][0], DCM[0][0]),
                             np.arcsin(DCM[2][0]),
                             np.arctan2(-DCM[2][1], DCM[2][2])])
        case 313:
            return np.array([np.arctan2(DCM[1][2], -DCM[0][2]),
                             np.arccos(DCM[2][2]),
                             np.arctan2(DCM[2][1], DCM[2][0])])