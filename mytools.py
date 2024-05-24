import math
import numpy as np


def rotm2quat(R):
    # assert(abs(np.linalg.det(R)-1)<1e-6)
    tr = np.trace(R)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = R.flatten()

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
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
    return q_unit #if q_unit[0] >= 0 else -q_unit

def similar_quat(q1, q_ref):
    q2 = -q1
    if np.linalg.norm(q1 - q_ref) < np.linalg.norm(q2 - q_ref):
        return q1
    else:
        return q2

def quat_angle_diff(q1, q2):
    R1 = quat2rotm(q1)
    R2 = quat2rotm(q2)
    R_diff = R1.T @ R2
    angle_diff = np.arccos((np.trace(R_diff) - 1) / 2)
    return abs(angle_diff)

def quat_multiply(q1, q2):
    q1_w = q1[0]
    q1_v = q1[1:4]
    q2_w = q2[0]
    q2_v = q2[1:4]
    result = np.hstack([q1_w*q2_w- np.dot(q1_v,q2_v),
             (q1_w*q2_v + q2_w*q1_v + np.cross(q1_v, q2_v))])
    return result

def exp_to_quat(omega, dt):
    angle = np.linalg.norm(omega)
    axis = omega/np.linalg.norm(omega)
    return np.hstack([np.cos(angle/2), axis*np.sin(angle/2)])

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

def R_to_axis_angle(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    axis = 1/(2*np.sin(angle))*np.array([R[2,1]-R[1,2],
                                       R[0,2]-R[2,0],
                                       R[1,0]-R[0,1]])
    return (axis, angle)