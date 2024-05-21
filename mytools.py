import math
import numpy as np

def rotm2quat(R):
    # assert(abs(np.linalg.det(R)-1)<1e-6)
    tr = np.trace(R)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.flatten()

    if tr > 0:
        S = np.sqrt(tr+1.0) * 2  # S=4*qw 
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) & (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


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
    return q_unit if q_unit[0] >= 0 else -q_unit

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

def padding_nan(data):
    maxlen = max(len(l) for l in data)
    for i, l in enumerate(data):
        if maxlen - len(l) == 0:
            continue
        data[i] = np.concatenate([l, [np.nan] * (maxlen - len(l))])
    return np.array(data)