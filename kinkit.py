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

# Euler Angle Mappings
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

def DCMtoEA(sequence, DCM):
    match sequence:
        case 121:
            return np.flip(np.array([np.arctan2(DCM[1][0], DCM[2][0]),
                             np.arccos(DCM[0][0]),
                             np.arctan2(DCM[0][1], -DCM[0][2])]))
        case 123:
            return np.flip(np.array([np.arctan2(DCM[1][2], DCM[2][2]),
                             -np.arcsin(DCM[0][2]),
                             np.arctan2(DCM[0][1], DCM[0][0])]))
        case 131:
            return np.flip(np.array([np.arctan2(DCM[2][0], -DCM[1][0]),
                             np.arccos(DCM[0][0]),
                             np.arctan2(DCM[0][2], DCM[0][1])]))
        case 132:
            return np.flip(np.array([np.arctan2(-DCM[2][1], DCM[1][1]),
                             np.arcsin(DCM[0][1]),
                             np.arctan2(-DCM[0][2], DCM[0][0])]))
        case 212:
            return np.flip(np.array([np.arctan2(DCM[0][1], -DCM[2][1]),
                             np.arccos(DCM[1][1]),
                             np.arctan2(DCM[1][0], DCM[1][2])]))
        case 213:
            return np.flip(np.array([np.arctan2(-DCM[0][2], DCM[2][2]),
                             np.arcsin(DCM[1][2]),
                             np.arctan2(-DCM[1][0], DCM[1][1])]))
        case 231:
            return np.flip(np.array([np.arctan2(DCM[2][0], DCM[0][0]),
                             -np.arcsin(DCM[1][0]),
                             np.arctan2(DCM[1][2], DCM[1][1])]))
        case 232:
            return np.flip(np.array([np.arctan2(DCM[2][1], DCM[0][1]),
                             np.arccos(DCM[1][1]),
                             np.arctan2(DCM[1][2], -DCM[1][0])]))
        case 312:
            return np.flip(np.array([np.arctan2(DCM[0][1], DCM[1][1]),
                             -np.arcsin(DCM[2][1]),
                             np.arctan2(DCM[2][0], DCM[2][2])]))
        case 313:
            return np.flip(np.array([np.arctan2(DCM[0][2], DCM[1][2]),
                             np.arccos(DCM[2][2]),
                             np.arctan2(DCM[2][0], -DCM[2][1])]))
        case 321:
            return np.flip(np.array([np.arctan2(-DCM[1][0], DCM[0][0]),
                             np.arcsin(DCM[2][0]),
                             np.arctan2(-DCM[2][1], DCM[2][2])]))
        case 323:
            return np.flip(np.array([np.arctan2(DCM[1][2], -DCM[0][2]),
                             np.arccos(DCM[2][2]),
                             np.arctan2(DCM[2][1], DCM[2][0])]))

# PRP Mappings
def DCMtoPRP(DCM):
    theta = np.arccos(0.5*(DCM[0][0]+DCM[1][1]+DCM[2][2]-1))
    lamHat = (1/(2*np.sin(theta))) * np.array([DCM[1][2]-DCM[2][1],
                                              DCM[2][0]-DCM[0][2],
                                              DCM[0][1]-DCM[1][0]])
    return (theta, lamHat)

def PRPtoDCM(PRP):
    th, lh = PRP

    DCM = np.array([[((lh[0] ** 2) * (1 - c(th)) + c(th)), ((lh[0] * lh[1]) * (1 - c(th)) + (lh[2] * s(th))),
                     ((lh[0] * lh[2]) * (1 - c(th)) - (lh[1] * s(th)))],
                    [((lh[1] * lh[0]) * (1 - c(th)) - (lh[2] * s(th))), ((lh[1] ** 2) * (1 - c(th)) + c(th)),
                     ((lh[1] * lh[2]) * (1 - c(th)) + (lh[0] * s(th)))],
                    [((lh[2] * lh[0]) * (1 - c(th)) + (lh[1] * s(th))),
                     ((lh[2] * lh[1]) * (1 - c(th)) - (lh[0] * s(th))), ((lh[2] ** 2) * (1 - c(th)) + c(th))]])
    return DCM

# CRP Mappings
def DCMtoCRP(DCM):
    theta, lamHat = DCMtoPRP(DCM)
    rho = lamHat*np.tan(theta/2)
    return rho

def CRPtoDCM(CRP):
    eps4 = 1/np.sqrt(1 + np.dot(CRP, CRP))
    eps13 = CRP/np.sqrt(1 + np.dot(CRP, CRP))
    return EPtoDCM(np.array([eps13[0], eps13[1], eps13[2], eps4]))

# MRP Mappings
def DCMtoMRP(DCM):
    theta, lamHat = DCMtoPRP(DCM)
    sigma = lamHat*np.tan(theta/4)
    return sigma

def MRPtoDCM(MRP):
    eps4 = (1-np.dot(MRP, MRP))/(1+np.dot(MRP, MRP))
    eps13 = (2*MRP)/(1+np.dot(MRP, MRP))
    return EPtoDCM(np.array([eps13[0], eps13[1], eps13[2], eps4]))

# Euler Parameter (Quaternion) mappings
def DCMtoEP(DCM):  # Sheppard's method
    e1sq = (1 / 4) * (1 + (2 * DCM[0][0]) - np.trace(DCM))
    e2sq = (1 / 4) * (1 + (2 * DCM[1][1]) - np.trace(DCM))
    e3sq = (1 / 4) * (1 + (2 * DCM[2][2]) - np.trace(DCM))
    e4sq = (1 / 4) * (1 + np.trace(DCM))

    if np.max([e1sq, e2sq, e3sq, e4sq]) == e1sq:
        eps = (1 / (4 * np.sqrt(e1sq))) * np.array(
            [4 * e1sq, DCM[0][1] + DCM[1][0], DCM[2][0] + DCM[0][2], DCM[1][2] - DCM[2][1]])
    elif np.max([e1sq, e2sq, e3sq, e4sq]) == e2sq:
        eps = (1 / (4 * np.sqrt(e2sq))) * np.array(
            [DCM[0][1] + DCM[1][0], 4 * e2sq, DCM[1][2] + DCM[2][1], DCM[2][0] - DCM[0][2]])
    elif np.max([e1sq, e2sq, e3sq, e4sq]) == e3sq:
        eps = (1 / (4 * np.sqrt(e3sq))) * np.array(
            [DCM[2][0] + DCM[0][2], DCM[1][2] + DCM[2][1], 4 * e3sq, DCM[0][1] - DCM[1][0]])
    elif np.max([e1sq, e2sq, e3sq, e4sq]) == e4sq:
        eps = (1 / (4 * np.sqrt(e4sq))) * np.array(
            [DCM[1][2] - DCM[2][1], DCM[2][0] - DCM[0][2], DCM[0][1] - DCM[1][0], 4 * e4sq])

    if eps[3] < 0:
        return -1 * eps
    else:
        return eps

def EPtoDCM(E):
    DCM = np.array([[1-(2*E[1]**2) - (2*E[2]**2), 2*(E[0]*E[1] + E[2]*E[3]), 2*(E[0]*E[2] - E[1]*E[3])],
                  [2*(E[0]*E[1] - E[2]*E[3]), 1-(2*E[0]**2) - (2*E[2]**2), 2*(E[1]*E[2] + E[0]*E[3])],
                  [2*(E[0]*E[2] + E[1]*E[3]), 2*(E[1]*E[2] - E[0]*E[3]), 1-(2*E[0]**2) - (2*E[1]**2)]])
    return DCM


def DCMtoEP_standard(DCM):
    theta, lamHat = DCMtoPRP(DCM)
    eps13 = lamHat * np.sin(theta / 2)
    eps4 = np.cos(theta / 2)

    if eps4 < 0:
        return -1 * np.array([eps13[0], eps13[1], eps13[2], eps4])
    else:
        return np.array([eps13[0], eps13[1], eps13[2], eps4])

def dwdt_Bframe(t, omega, I, L):
    omedot = [((-(I[2]-I[1])*omega[1]*omega[2]) + (L[0]))/I[0],
              ((-(I[0]-I[2])*omega[0]*omega[2]) + (L[1]))/I[1],
              ((-(I[1]-I[0])*omega[0]*omega[1]) + (L[2]))/I[2]]
    return omedot

def EP_KDE(t, epsilon, omega):
    omegaA = np.array([omega[0], omega[1], omega[2], 0])
    magic = np.array([[EP[3], -EP[2], EP[1], EP[0]],
                      [EP[2], EP[3], -EP[0], EP[1]],
                      [-EP[1], EP[0], EP[3], EP[2]],
                      [-EP[0], -EP[1], -EP[2], EP[3]]])
    epsilonDot = (0.5)*np.matmul(magic, omegaA)
    return epsilonDot

def MRP_KDE_std(t, MRP, omega):
    ss = np.dot(MRP, MRP)
    omegaA = np.array([omega[0], omega[1], omega[2]])
    magic = np.array([[1-ss+2*(MRP[0]**2), 2*(MRP[0]*MRP[1] - MRP[2]), 2*(MRP[0]*MRP[2] + MRP[1])],
                      [2*(MRP[1]*MRP[0] + MRP[2]), 1-ss+2*(MRP[1]**2), 2*(MRP[1]*MRP[2] - MRP[0])],
                      [2*(MRP[2]*MRP[0] - MRP[1]), 2*(MRP[2]*MRP[1] + MRP[0]), 1-ss+2*(MRP[2]**2)]])
    sigmaDot = (0.25)*np.matmul(magic, omegaA)
    return sigmaDot