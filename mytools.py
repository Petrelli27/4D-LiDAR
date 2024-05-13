import math
import numpy as np

def rotm2quat(R):
    assert(abs(np.linalg.det(R)-1)<1e-6)
    qw = 0.5*math.sqrt(1+R[0,0]+R[1,1]+R[2,2])
    qx = (R[2,1]-R[1,2])/(4*qw)
    qy = (R[0,2]-R[2,0])/(4*qw)
    qz = (R[1,0]-R[0,1])/(4*qw)
    return np.array([qw,qx,qy,qz])

def quat2rotm(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = np.array([[2*(qw**2+qx**2)-1, 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                  [2*(qx*qy+qw*qz), 2*(qw**2+qy**2)-1, 2*(qy*qz - qw*qx)],
                  [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 2*(qw**2+qz**2)-1]])
    return R

def normalize_quat(q):
    q_norm = np.linalg.norm(q)
    q_unit = q/q_norm
    return q_unit

def sigmoid(x, a=0.9, k=7):
    s= 1/(1+np.exp(k*(x-a)))
    y = -2/math.pi *x + 1
    if y > 1:
        y = 1
    elif y < 0:
        y = 0
    return s

def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1