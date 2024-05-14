import numpy as np
from math import trunc
from mytools import *
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# some utility functions
def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color):
    # z1 plane boundary
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)  # W
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color)
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color='g')

    # z1 plane boundary
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color)  # W
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color)
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color)
    ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color='g')

    # Connecting
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color='g')  # W
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color)
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color)
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color='g')

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='b')

def tilde(v):
    vx = v[0,0]
    vy = v[1,0]
    vz = v[2,0]
    v_tilde = np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]])
    return v_tilde

def getR(x,y,z):
    # we want to find the rotation matrix that takes [x,y,z] to [0,0,1]
    p = np.array([x,y,z]) # position vector, but also z-axis of b frame
    z_L = np.array([0,0,1]) # z-axis of L frame
    z_B = p/np.linalg.norm(p)
    e = (np.cross(z_L, z_B))[:,np.newaxis] # 3x1 axis of rotation
    phi = np.arccos(np.dot(z_B, z_L)) # z_B and z_L are already unit vectors
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + tilde(e)*np.sin(phi)
    return R.T

def nearest_search(pi_k, z_pi_k, z_c_k):
    """ Identify a set of points from the target set that is closest to the
        source set, based on nearest neighbour search with the L2 norm.
    """

    z_pi_k = z_pi_k.T - z_c_k
    ec_dist_i = []
    ec_dists = np.zeros([8,8])
    z_pis = np.zeros([8,3])

    for i, pi_k_i in enumerate(pi_k):

        # Compute the L2 norm between these two points
        ec_dist = np.sqrt(sum((i - j) ** 2 for i, j in zip(z_pi_k.T, pi_k_i)))
        ec_dists[i, :] = ec_dist

    print(ec_dists)
    print("Hello")
    for i, ec_dist in enumerate(ec_dists):
        #print(ec_dists)
        all_idx = np.argmin(ec_dists)
        print(all_idx)
        pi_idx = trunc(all_idx/8)
        z_idx = all_idx % 8
        #print(z_idx)
        z_pis[pi_idx, :] = z_pi_k[z_idx, :]
        ec_dists[pi_idx, :] = 1000
        ec_dists[:, z_idx] = 1000

    z_p1_k = z_pis[0, :] + np.array(z_c_k)
    z_p2_k = z_pis[1, :] + np.array(z_c_k)
    z_p3_k = z_pis[2, :] + np.array(z_c_k)
    z_p4_k = z_pis[3, :] + np.array(z_c_k)
    z_p5_k = z_pis[4, :] + np.array(z_c_k)
    z_p6_k = z_pis[5, :] + np.array(z_c_k)
    z_p7_k = z_pis[6, :] + np.array(z_c_k)
    z_p8_k = z_pis[7, :] + np.array(z_c_k)

    return z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k

def mahalonobis_association(pi_k, z_pi_k, z_c_k):

    z_pi_k = z_pi_k.T - z_c_k  # translate to origin

    return

def rotation_association(q_kp1, R_1):
    # Orientation association
    # R_1 is obtained from bounding box
    predicted_R = quat2rotm(q_kp1)
    possible_Rs = []
    angle_diffs = []
    possible_xs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    possible_ys = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    for x_axis in possible_xs:
        for y_axis in possible_ys:
            if (x_axis == y_axis).all() or (np.cross(x_axis, y_axis)==0).all():
                continue  # skip over case where x and y axis overlap
            else:
                z_axis = np.cross(x_axis, y_axis)
                possible_R = R_1@np.array([x_axis, y_axis, z_axis])
                possible_Rs.append(possible_R)
                rotation_diff = (predicted_R.T) @ (possible_R)
                angle_diff = np.arccos((np.trace(rotation_diff) - 1) / 2)
                angle_diffs.append(angle_diff)
    angle_errors = np.array(np.abs(angle_diffs)) # take abs value and convert to np array
    R_indices = np.argsort(angle_errors)
    threshold = np.deg2rad(32)
    Q_list = []
    weights = []
    counter = 0
    for i in R_indices:
        if counter > 1:
            break
        angle_error = angle_errors[i]
        # if angle_error  <= threshold:
        Q_list.append((rotm2quat(possible_Rs[i]))) # auto sorted
        weights.append(sigmoid(angle_error))
        counter += 1
    min_error = min(abs(angle_errors))
    bad_attitude_measurement_flag = (min_error > threshold)
    # bad_attitude_measurement_flag = False
    weights_normed = np.array(weights)/np.sum(weights)

    # Q = (np.array(Q_list)).T
    # Q = (weights_normed) * Q
    # QQT = Q@(Q.T)
    # eigenvalues, eigenvectors = np.linalg.eig(QQT)
    # max_eigval_index = np.argmax(eigenvalues)
    # q_avg = eigenvectors[max_eigval_index]
    # if q_avg[0] < 0:
    #     q_avg = -q_avg
    # return Q_list[0], bad_attitude_measurement_flag
    return slerp(Q_list[0], Q_list[1], weights_normed[1]), bad_attitude_measurement_flag, min_error