# KinKit
Simple but useful toolbox for kinematics and attitude dynamics in Python. Written with numpy only.

Currently Supporting:

Forward Mapping:
- Euler Angle Sequence to DCM
- DCM to Principal Rotation Parameters (PRP)
- DCM to Classical Rodrigues Parameters (CRP)
- DCM to Modified Rodrigues Parameters (MRP)
- DCM to Euler Parameter (Quarternions) using Sheppard's Method

Inverse Mappimng:
- DCM to EA Sequence
- PRP to DCM
- CRP to DCM
- MRP to DCM
- EP to DCM

Kinematic Differential Equations:
- Euler Parameter KDE
- Modified Rodrigues Parameter KDE

Additional Functions:
- Axisymmetric Body Equations of Motion function (`dwdt_Bframe`).

Check `demo.ipynb` for a demonstration of provided functions.
