import copy
import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan2
import boundingbox
from stl import mesh
from estimateOmega import estimate_LLS, estimate_kabsch, estimate_rotation_B
from associationdata import nearest_search
from associationdata import rotation_association
from mytools import *
import pickle
import open3d as o3d

from mpl_toolkits import mplot3d
from matplotlib import pyplot

def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color, linewidth):
    # z1 plane boundary
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=color, linewidth=linewidth)

    # z1 plane boundary
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color, linewidth=linewidth)
    ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=color, linewidth=linewidth)

    # Connecting
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color, linewidth=linewidth)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='#9b42f5')

def skew(vector):
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def getR(x,y,z):
    # we want to find the rotation matrix that takes [x,y,z] to [0,0,1]
    p = np.array([x,y,z]) # position vector, but also z-axis of b frame
    z_L = np.array([0,0,1]) # z-axis of L frame
    z_B = p/np.linalg.norm(p)
    e = (np.cross(z_L, z_B))[:,np.newaxis] # 3x1 axis of rotation
    phi = np.arccos(np.dot(z_B, z_L)) # z_B and z_L are already unit vectors
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + skew(e)*np.sin(phi)
    return R.T

def rodrigues(omega, dt):
    
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R

def verticeupdate(dt, x_k):

    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]

    # Rotation matrix - rodrigues formula
    R_k_kp1 = rodrigues(omega_k, dt)
    R_k_kp1 = R_k_kp1

    # Translate vertex to origin
    p1_ko = p1_k - p_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1, p1_ko.reshape(len(p1_ko), 1))

    # Translate vertex back to new expected origin
    p1_kp1 = (p1_kp1o.T + p_k + v_k * dt).ravel()

    return p1_kp1, R_k_kp1

def orientationupdate(dt, x_k):

    # Decompose the state vector
    omega_k = x_k[6:9]
    q_k = x_k[12:16]
    qw = q_k[0]; qx = q_k[1]; qy = q_k[2]; qz = q_k[3]

    # hamilton = np.array([-omega_k[0] * q_k[1] - omega_k[1] * q_k[2] - omega_k[2] * q_k[3],
    #             omega_k[0] * q_k[0] + omega_k[2] * q_k[2] - omega_k[1] * q_k[3],
    #             omega_k[1] * q_k[0] - omega_k[2] * q_k[1] + omega_k[0] * q_k[3],
    #             omega_k[2] * q_k[0] + omega_k[1] * q_k[1] - omega_k[0] * q_k[2]])
    
    dqkdt = 0.5*np.array([[-qx, -qy, -qz],
                          [qw, qz, -qy],
                          [-qz, qw, qx],
                          [qy, -qx, qw]]) @ omega_k
    
    q_kp1 = similar_quat(normalize_quat(dqkdt*dt + q_k), q_k)
    return q_kp1

def F_matrix(dt, R, x_k):
    """

    :param dt:
    :param R:
    :param p_k:
    :param p1_k:
    :param p2_k:
    :param p3_k:
    :param p4_k:
    :param p5_k:
    :param p6_k:
    :param p7_k:
    :param p8_k:
    :return:
    """

    p_k = x_k[:3]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]
    q_k = x_k[12:]

    F = np.eye(16)

    # dp_k/dvT_k
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # dp1_k/dp_k
    F[9:12, 0:3] = np.eye(3) - R

    # dp1_k/dvT_k
    F[9, 3] = dt
    F[10, 4] = dt
    F[11, 5] = dt

    # dp1_k/domega_k
    F[9:12, 6:9] = skew(p1_k - p_k)

    # dp1_k/dp1_k
    F[9:12, 9:12] = R

    # dq_k/domega_k
    q_123k = q_k[1:]
    q_0k = q_k[0]
    bottom = np.array(skew(q_123k) + q_0k * np.eye(3))
    F[12:, 6:9] = 0.5 * dt * np.concatenate((-np.array([q_123k]), bottom), axis=0)

    # dq_k/dq_k
    F_temp = np.zeros((4, 4))
    F_temp[0, 1:] = -np.array(omega_k)
    F_temp[1:, 0] = omega_k
    F_temp[1:, 1:] = skew(omega_k)
    F[12:, 12:] = F_temp + np.eye(4)

    return F

def get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini):

    Rot_0 = quat2rotm(q_ini)
    Rot_0 = np.eye(3)
    # print(Rot_0)
    q_s = []
    for i in range(len(debris_pos)):

        # get rotation matrix for that timestep
        Rot_i = rodrigues(omega_true, dt * i)
        q_i = rotm2quat(Rot_i @ Rot_0)
        if i == 0:
            q_s.append(q_i)
        else:
            q_i_alt = -q_i
            q_prev = q_s[i-1]
            if np.linalg.norm(q_i - q_prev) < np.linalg.norm(q_i_alt - q_prev):
                q_s.append(q_i)
            else:
                q_s.append(q_i_alt)
    return q_s



# initialize debris position, velocity and orientation
O_B = np.array([0,0,0])
O_L = np.array([0,0,0])

with open('sim_kompsat_neg_om_longer.pickle', 'rb') as sim_data:
# with open('sim_kompsat_neg_om_longer.pickle', 'rb') as sim_data:
# with open('sim_new_conditions.pickle', 'rb') as sim_data:
    data = pickle.load(sim_data)
XBs = data[0]
YBs = data[1]
ZBs = data[2]
PBs = data[3]
VBs = data[4]

debris_pos = data[5]
debris_vel = data[6]
Rot_L_to_B = data[7]
Rot_B_to_L = [np.transpose(r) for r in Rot_L_to_B]
omega_L = data[8]
dt = data[9]


# Estimation Loop
XLs = []  # store point cloud x in L
YLs = []
ZLs = []
PLs = []  # store x, y, z point cloud in L
VLs = VBs  # store velocity point cloud
x_s = []  # store states over time
z_s = []  # store measurements over time
P_s = []  # store covariances in time
errors = [0]
nframes = len(VBs)

# Running the simulation - Initializations

# Initializations in L Frame
vT_0 = [1., 1., 1.]  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = [1., 1., 1.]
omega_true = [1., 1., 1.]
q_ini = [1., 0., 0., 0.]
q_true = np.array(get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini))
q_true_alt = -q_true

# Initial covariance
P_0 = np.diag([0.25, 0.5, 0.25, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25])  # Initial Covariance matrix
P_k = P_0.copy()  # covariance matrix

# Process noise covariance matrix
qp = 0.0000001
qv = 0.0000005
qom = 0.005
qp1 = 0.05
qq = 0.000005
Q = np.diag([qp, qp, qp, qv, qv, qv, qom, qom, qom, qp1, qp1, qp1, qq, qq, qq, qq])

# Measurement noise covariance matrix
p = 0.00025
om = .05
p1 = 1
q = 0.0004
R1 = np.diag([p, p, p, om, om, om, p1, p1, p1, q, q, q, q])
R2 = np.diag([p, p, p, om, om, om, p1, p1, p1])


# Measurement matrix
H1 = np.zeros([len(P_0)-3, len(P_0)])  # no measuring of velocity
H1[0:3,0:3] = np.eye(3)
H1[3:,6:] = np.eye(10)
bad_attitude_measurement_flag = False

H2 = np.zeros([9,16])
H2[0:3,0:3] = np.eye(3)
H2[3:6,6:9] = np.eye(3)
H2[6:, 9:12] = np.eye(3)

# Kabsch estimation parameters
n_moving_average = 40
settling_time = 500
# Record keeping for angular velocity estimate
omegas_kabsch_b = np.zeros((nframes, 3))
omegas_lls_b = np.zeros((nframes, 3))
omega_kabsch_b_box = np.zeros((n_moving_average,3))

q_kp1s =[]
metrics = []
z_s_all = []
for i in range(nframes):
    visualize_flag = i>0 and i%300==1
    # Use first measurements for initializations of states
    if i > 0:
        # Decompose the state vector
        p_k = x_k[:3]
        v_k = x_k[3:6]
        omega_k = x_k[6:9]
        p1_k = x_k[9:12]
        q_k = x_k[12:16]

        ##############
        # Prediction
        ##############

        # Position update
        p_kp1 = v_k * dt + p_k

        # Velocity update
        v_kp1 = v_k.copy()

        # Angular velocity update
        omega_kp1 = omega_k.copy()

        # Vertex update
        p1_kp1, R_k_kp1 = verticeupdate(dt, x_k)

        # Orientation Update
        if i == 1:
            q_kp1s.append(q_ini)
        # x_k[12:] = q_true[i - 1]
        q_kp1 = orientationupdate(dt, x_k)
        q_kp1s.append(q_kp1)

        # plt.figure()
        # array_qkp1s = np.array(q_kp1s)
        # array_qtrue = np.array(q_true)
        # plt.plot(np.arange(len(array_qkp1s[:, 0])) * dt, array_qkp1s[:, 3], color='orange')
        # plt.plot(np.arange(len(array_qtrue[:i+1, 0])) * dt, array_qtrue[0:i+1, 3], color='green')
        # plt.show()

        # Compute Jacobian
        F_kp1 = F_matrix(dt, R_k_kp1, x_k)

        # Update Covariance
        P_kp1 = np.matmul(F_kp1, np.matmul(P_k, F_kp1.T)) + Q

        # Make updated State vector
        x_kp1 = np.hstack([p_kp1, v_kp1, omega_kp1, p1_kp1, q_kp1])

    #######################
    # Measurements
    #######################

    PLs.append((Rot_L_to_B[i].T @ (PBs[i]).T).T)
    # find bounding box from points
    XLs.append(PLs[i][:, 0])
    YLs.append(PLs[i][:, 1])
    ZLs.append(PLs[i][:, 2])
    X_i = XLs[i]
    Y_i = YLs[i]
    Z_i = ZLs[i]


    # Return bounding box and centroid estimate of bounding box
    z_pi_k_1, z_p_k_1, R_1 = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
    z_pi_k_2, z_p_k_2, R_1_2, normal_vecs, ranking = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, True, visualize_flag)
    # R_1_2 = R_1_2.T
    # z_p_k = debris_pos[i, :]
    # Orientation association
    # R_1 is obtained from bounding box
    if i == 0:
        # z_q_k = rotm2quat(R_1)
        z_q_k_1 = rotm2quat(R_1 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]) )  # this rotation is to set initial orientation to match with true
        z_q_k_2 = rotm2quat(R_1_2 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]))  # this rotation is to set initial orientation to match with true
        z_q_k = z_q_k_1.copy()
        z_pi_k = z_pi_k_1.copy()
        z_p_k = z_p_k_1.copy()
        perfect_metric = False
        is_z_q_k_good = True
        is_z_q_k_2_good = True
    else:
        z_q_k_1, _, error = rotation_association(q_kp1, R_1)
        z_q_k_2, bad_attitude_measurement_flag_2, error_2 = rotation_association(q_kp1, R_1_2)
        if quat_angle_diff(z_q_k_1, q_true[i,:]) > np.deg2rad(25):
            perfect_metric = True
        else:
            perfect_metric = False

        if np.max(R_1.T @ R_1_2) < np.cos(np.deg2rad(35)):
            # bad z_q_k
            is_z_q_k_good = False
        else:
            is_z_q_k_good = True
        if  np.max(ranking) < 85:
            is_z_q_k_2_good = False
        else:
            is_z_q_k_2_good = True
        # z_q_k_2 = rotm2quat(R_1_2)
        # z_q_k = q_true[i]
        # errors.append(np.rad2deg(error))
        if is_z_q_k_good and is_z_q_k_2_good:
            # everything is good, so just use z_q_k
            bad_attitude_measurement_flag = False
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
        elif (i<100) or (is_z_q_k_good and (not is_z_q_k_2_good)):
            # ransac bbox bad because there were no orthogonal planes
            bad_attitude_measurement_flag = False
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
        elif (not is_z_q_k_good) and (is_z_q_k_2_good):
            # pca is bad, but ransac results are good
            bad_attitude_measurement_flag = False
            z_q_k = z_q_k_2.copy()
            z_pi_k = z_pi_k_2.copy()
            z_p_k = z_p_k_2.copy()
        elif (not is_z_q_k_good) and (not is_z_q_k_2_good):
            # both are bad
            bad_attitude_measurement_flag = True
    metrics.append([int(perfect_metric), int(not is_z_q_k_good)])        
    if i < 100: bad_attitude_measurement_flag = False # don't skip things until 5 seconds in
    if i>0:
        LWD = 2*quat2rotm(q_kp1).T @ (p_kp1 - p1_kp1)
        L = LWD[0]; W = LWD[1]; D = LWD[2]
        predictedBbox = boundingbox.from_params(p_kp1, q_kp1, L, W, D)# just use the predicted box instead
    if i==0 or (not perfect_metric):
        # first use q from R_1 to get L,W,D
        # then use z_q_k (not perfectly aligned) to get 
        associatedBbox_1, Lm_1, Wm_1, Dm_1 = boundingbox.associated(z_q_k_1, z_pi_k_1, z_p_k_1, R_1)  # L: along x-axis, W: along y-axis D: along z-axis
        z_p1_k_1 = associatedBbox_1[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)
        # associatedBbox_2, Lm_2, Wm_2, Dm_2 = boundingbox.associated(z_q_k_2, z_pi_k_2, z_p_k_2,
        #                                                     R_1_2)  # L: along x-axis, W: along y-axis D: along z-axis
        # z_p1_k_2 = associatedBbox_2[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)
        # associatedBbox, Lm, Wm, Dm = boundingbox.associated(z_q_k, z_pi_k , z_p_k , R_1)

        # verifying metric usefulness
        associatedBbox, Lm, Wm, Dm = associatedBbox_1, Lm_1, Wm_1, Dm_1
        z_q_k = z_q_k_1.copy()
        z_pi_k = z_pi_k_1.copy()
        z_p_k = z_p_k_1.copy()
        z_p1_k = associatedBbox[:, 0]
    else:
        print(f"bad attitude at t={i*dt}")
        associatedBbox = predictedBbox
        z_p_k = z_p_k_1.copy()
        z_p1_k = associatedBbox[:,0]
        


    if visualize_flag:
    # if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.title.set_text(f'Time={i*dt}s' + '\n' + f'Pred. Length={round(L,2)}m ' + f'Width={round(W, 2)}m ' + f'Height={round(D, 2)}m' + '\n' + f'Meas. Length={round(Lm,2)}m ' + f'Width={round(Wm, 2)}m ' + f'Height={round(Dm, 2)}m')
        # width = orange to green, blue to green
        # length = orange to cyan, blue to cyan
        # height = orange to magenta, blue to magenta
        ax.scatter(X_i, Y_i, Z_i, color='black', marker='o', s=2)
        # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], marker='o', color='r')
        ax.set_aspect('equal', 'box')

        # print(x_kp1)
        # drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'orange', 1)
        drawrectangle(ax, associatedBbox_1[:, 0], associatedBbox_1[:, 1], associatedBbox_1[:, 2], associatedBbox_1[:, 3],
                      associatedBbox_1[:, 4], associatedBbox_1[:, 5], associatedBbox_1[:, 6], associatedBbox_1[:, 7], 'b', 2)

        # drawrectangle(ax, associatedBbox_2[:, 0], associatedBbox_2[:, 1], associatedBbox_2[:, 2], associatedBbox_2[:, 3],
        #           associatedBbox_2[:, 4], associatedBbox_2[:, 5], associatedBbox_2[:, 6], associatedBbox_2[:, 7], 'orange', 2)
        
        drawrectangle(ax, associatedBbox[:, 0], associatedBbox[:, 1], associatedBbox[:, 2], associatedBbox[:, 3],
                  associatedBbox[:, 4], associatedBbox[:, 5], associatedBbox[:, 6], associatedBbox[:, 7], 'orange', 2)

    # drawrectangle(ax, z_pi_k[:, 0], z_pi_k[:, 1], z_pi_k[:, 2], z_pi_k[:, 3],
        #               z_pi_k[:, 4], z_pi_k[:, 5], z_pi_k[:, 6], z_pi_k[:, 7], 'r', 1)
        # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], color='b', s=20)
        # drawrectangle(ax, predictedBbox[:, 0], predictedBbox[:, 1], predictedBbox[:, 2], predictedBbox[:, 3],
        #               predictedBbox[:, 4], predictedBbox[:, 5], predictedBbox[:, 6], predictedBbox[:, 7], 'r', 1)

        # ax.scatter(predictedBbox[0, 0], predictedBbox[1, 0], predictedBbox[2, 0], color='orange', label='Vertex 1 Pred.')
        # ax.scatter(associatedBbox[0, 0], associatedBbox[1, 0], associatedBbox[2, 0], color='blue',
        #            label='Vertex 1 Meas.')
        ax.legend()
        
        Rot_measured = quat2rotm(z_q_k)

        Rot_measured_2 = quat2rotm(z_q_k_2)
        Rot_measured_2 = R_1_2
        # normal_vecs = normal_vecs.T

        R_estimated = quat2rotm(q_kp1)

        R_true = quat2rotm(q_true[i,:])


        # plot measured
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 0]], [z_p_k[2], z_p_k[2] + Rot_measured[2, 0]],
        color='blue', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 1]], [z_p_k[2], z_p_k[2] + Rot_measured[2, 1]],
        color='blue', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 2]], [z_p_k[2], z_p_k[2] + Rot_measured[2, 2]],
        color='b', linewidth=4)

        # plot measured
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 0]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 0]],
                color='red', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 1]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 1]],
                color='red', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 2]],
                [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 2]],
                color='red', linewidth=4)

        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 0]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 0]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 0]],
        #         color='blue', linewidth=4)
        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 1]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 1]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 1]],
        #         color='blue', linewidth=4)
        # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 2]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 2]],
        #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 2]],
        #         color='blue', linewidth=4)
        #

        # plot current estimate of ekf
        # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 0]], [z_p_k[1], z_p_k[1] + R_estimated[1, 0]],
        #         [z_p_k[2], z_p_k[2] + R_estimated[2, 0]],
        #         color='orange', linewidth=4)
        # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 1]], [z_p_k[1], z_p_k[1] + R_estimated[1, 1]],
        #         [z_p_k[2], z_p_k[2] + R_estimated[2, 1]],
        #         color='orange', linewidth=4)
        # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 2]], [z_p_k[1], z_p_k[1] + R_estimated[1, 2]],
        #         [z_p_k[2], z_p_k[2] + R_estimated[2, 2]],
        #         color='orange', linewidth=4)

        # plot true
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 0]], [z_p_k[1], z_p_k[1] + R_true[1, 0]],
                [z_p_k[2], z_p_k[2] + R_true[2, 0]],
                color='green', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 1]], [z_p_k[1], z_p_k[1] + R_true[1, 1]],
                [z_p_k[2], z_p_k[2] + R_true[2, 1]],
                color='green', linewidth=4)
        ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 2]], [z_p_k[1], z_p_k[1] + R_true[1, 2]],
                [z_p_k[2], z_p_k[2] + R_true[2, 2]],
                color='green', linewidth=4)

        # plot b_frame
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 0]], [0., 0. + Rot_B_to_L[i][1, 0]],
        #         [0., 0. + Rot_B_to_L[i][2, 0]],
        #         color='r', linewidth=1)
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 1]], [0., 0. + Rot_B_to_L[i][1, 1]],
        #         [0., 0. + Rot_B_to_L[i][2, 1]],
        #         color='g', linewidth=1)
        # ax.plot([0., 0. + Rot_B_to_L[i][0, 2]], [0., 0. + Rot_B_to_L[i][1, 2]],
        #         [0., 0. + Rot_B_to_L[i][2, 2]],
        #         color='b', linewidth=1)

        # black is axis of rotation
        # ax.plot([z_p_k[0], z_p_k[0] + 1], [z_p_k[1], z_p_k[1] + 1],
        #         [z_p_k[2], z_p_k[2] + 1],
        #         color='black', linewidth=4)




        # outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # Visualize the inliers (plane) and outliers
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red plane
        # outlier_cloud.paint_uniform_color([0.0, 1, 0])  # Green remaining points
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        plt.show()

        # ax.scatter(x_k[0], x_k[1], x_k[2], color='r' )
        # ax.scatter(z_p_k[0], z_p_k[1], z_p_k[2], color='b')
        # ax.scatter(debris_pos[i,0], debris_pos[i,1], debris_pos[i,2], color='g')



    # find angular velocity from LOS velocities
    if i > 0:
        # 1. Linear Least Squares
        omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i]@z_p_k, Rot_L_to_B[i]@v_k, VBs[i])
        omega_LLS = Rot_B_to_L[i] @ omega_LLS_B

    # 2. Rotation of B Frame
    omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)

    # 3. Kabsch
    ################ to use Kabsch you need i > 0, to wait for state initializations?
    if i==0:
        omega_los_L = np.array([0,0,0])
        prev_box_L = np.transpose(copy.deepcopy(associatedBbox))
        prev_box_B = (Rot_L_to_B[i] @ prev_box_L.T).T
    else:
        cur_box_L = np.transpose(copy.deepcopy(associatedBbox))
        cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
        # rotate previous box with everything else
        # prev_box_B = (rodrigues((omega_LLS + omega_L_to_B), dt) @ prev_box_B.T).T
        omega_los_B = estimate_kabsch(prev_box_B, cur_box_B, dt)
        prev_box_B = cur_box_B # for next iteration

        # using moving average to smooth out omega_los_B
        omega_kabsch_b_box[i%n_moving_average] = omega_los_B
        if i < n_moving_average:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i+1], axis=0)
        else:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
        omega_los_L = Rot_B_to_L[i]@omega_los_B_averaged
    
    # 3b Kabsch alternate
    # this should be done in the B frame
    # omega_los_Ls = np.zeros([n_moving_average, 3])
    # if i==0:
    #     omega_los_L = np.array([0,0,0])
    # else:
    #     prev_q = z_s[i-1][9:]
    #     cur_q = z_q_k
    #     prev_R_in_B = Rot_L_to_B[i] @ quat2rotm(prev_q)
    #     cur_R_in_B = Rot_L_to_B[i] @ quat2rotm(cur_q)
    #     prev_R_in_B_after_LLS = prev_R_in_B @ rodrigues(omega_LLS_B, dt)
    #     net_R_in_B = (prev_R_in_B_after_LLS.T) @ (cur_R_in_B)
    #     rotation_axis_bbox, rotation_angle_bbox = R_to_axis_angle(net_R_in_B)
    #     omega_bbox = rotation_axis_bbox * rotation_angle_bbox / dt
    #     # take only the z_component (along LOS) of omega_bbox
    #     omega_los_B_q = np.array([0,0,omega_bbox[2]])
    #     omega_los_L_q = (Rot_B_to_L[i] @ omega_los_B_q.reshape([3,1])).reshape(3)
    #     omega_los_Ls[i%n_moving_average] = omega_los_L_q
    #     if i < n_moving_average:
    #         omega_los_L_averaged = np.mean(omega_los_Ls[0:i+1], axis=0)
    #     else:
    #         omega_los_L_averaged = np.mean(omega_los_Ls, axis=0)
        

    # Combine angular velocity estimates
    if i == 0:
        z_omega_k = omega_0
    elif i <= settling_time:
        z_omega_k = omega_LLS + omega_L_to_B # ignores kabsch
    else:
        z_omega_k = omega_LLS + omega_L_to_B + omega_los_L

    # z_omega_k = np.array([1, 1, 1])
    #################################

    # Compute Measurement Vector
    # if False:
    if perfect_metric:
        z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k])
    else:
        z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k])

    # Set initial states to measurements
    if i == 0:
        x_k = np.hstack([np.array([-170., -350., -20.]), vT_0, z_omega_k,z_p1_k, q_ini])

    ##############
    # Update - Combine Measurement and Estimates
    ##############

    # allow for some time for states to settle
    if i > 0:
        # if False:
        if perfect_metric:
            H = H2
            R = R2
        # if abs(np.linalg.norm(z_p_k - p_kp1)) > 0.7:
        #     pass
        else:
            # Calculate the Kalman gain
            H = H1
            R = R1
        K_kp1 = np.matmul(P_kp1, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(P_kp1, H.T)) + R)))

        # Calculate Residual
        res_kp1 = z_kp1 - np.matmul(H, x_kp1)

        # Update State
        x_kp1 = x_kp1 + np.matmul(K_kp1, res_kp1)
        x_kp1[12:] = similar_quat(normalize_quat(x_kp1[12:]), x_k[12:16])

        # Update Covariance
        P_kp1 = np.matmul(np.eye(len(K_kp1)) - K_kp1 @ H, P_kp1)

        # for debugging purposes
        if (np.isnan(P_kp1)).any():
            print('something went wrong')

    # Transfer states and covariance from kp1 to k
    if i>0:
        P_k = P_kp1.copy()
        x_k = x_kp1.copy()

    # Append for analysis
    z_s.append(z_kp1)
    P_s.append(P_k)
    x_s.append(x_k)




##############
# Plot relevant figures
##############
#print(z_p_s)
#print(debris_pos)

z_s = padding_nan(z_s)
x_s = np.array(x_s)

plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.legend()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.scatter(debris_pos[1,0], debris_pos[1,1], debris_pos[1,2], color='orange', marker='o', s=20)
ax.scatter(debris_pos[-1,0], debris_pos[-1,1], debris_pos[-1,2], color='k', marker='o', s=20)
ax.scatter(z_s[:,0], z_s[:,1], z_s[:,2], color='b', s=0.3, linewidths=0)
ax.plot(debris_pos[:,0], debris_pos[:,1], debris_pos[:,2], color='g')
plt.legend(['Start','End','Computed Centroid Positions', 'True Centroid Positions'])
plt.xlim([-170.5,-167.5])
plt.ylim([-351,-306])
ax.set_zlim(-20,-9)

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 3], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 6], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_x$ (rad/s)')
#plt.title('$\displaystyle\Omega_x$')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:,4], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,7], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_y$ (rad/s)')
#plt.title('Omega Y')


fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:,5], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,8], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()

plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_z$ (rad/s)')
#plt.title('Omega Z')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 6] - omega_true[0], label='Error $\displaystyle\Omega_x$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 7] - omega_true[1], label='Error $\displaystyle\Omega_y$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 8] - omega_true[2], label='Error $\displaystyle\Omega_z$', linewidth=2)
# plt.plot(np.arange(0, dt*nframes, dt), np.zeros([nframes,1]), linewidth = 1) # draw line at zero
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity Error (rad/s)')
#plt.title('Angular Velocity Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 0] - debris_pos[:nframes,0], label='Error $\displaystyle p_x$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 1] - debris_pos[:nframes,1], label='Error $\displaystyle p_y$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 2] - debris_pos[:nframes,2], label='Error $\displaystyle p_x$', linewidth=2)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
#plt.title('Position Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 0], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 0], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:nframes,0], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_x$ (m)')
#plt.title('X Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 1], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 1], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:nframes,1], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_y$ (m)')
#plt.title('Y Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 2], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 2], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:nframes,2], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_z$ (m)')
#plt.title('Z Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,3] - debris_vel[:nframes,0], label='Error $\displaystyle v_{Tx}$', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,4] - debris_vel[:nframes,1], label='Error $\displaystyle v_{Ty}$', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,5] - debris_vel[:nframes,2], label='Error $\displaystyle v_{Tz}$', linewidth=1)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
#plt.title('Velocity Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 3], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:nframes, 0], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Tx}$ (m/s)')
#plt.title('Velocity in X')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,4], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:nframes,1], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Ty}$ (m/s)')
#plt.title('Velocity in y')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:,5], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:nframes,2], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Tz}$ (m/s)')
#plt.title('Velocity in z')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 9], label='$\displaystyle p_{1x}$')
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 10], label='$\displaystyle p_{1y}$')
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 11], label='$\displaystyle p_{1z}$')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Vertex $\displaystyle p_{1}$ Position (m)')
#plt.title('Position of Vertice P1 overt time')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 9], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 12], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), q_true[:nframes, 0], label='True', linewidth=1, linestyle='dashed')
plt.plot(np.arange(0, dt*nframes, dt), q_true_alt[:nframes, 0], label='True equiv.', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_0$')
#plt.title('Orientation $\displaystyle q_0$')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 10], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 13], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), q_true[:nframes,1], label='True', linewidth=1, linestyle='dashed')
plt.plot(np.arange(0, dt*nframes, dt), q_true_alt[:nframes, 1], label='True equiv.', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_1$')
#plt.title('Orientation $\displaystyle q_1$')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 11], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 14], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), q_true[:nframes, 2], label='True', linewidth=1, linestyle='dashed')
plt.plot(np.arange(0, dt*nframes, dt), q_true_alt[:nframes, 2], label='True equiv.', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_2$')
#plt.title('Orientation $\displaystyle q_2$')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_s[:, 12], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:, 15], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), q_true[:nframes, 3], label='True', linewidth=1, linestyle='dashed')
plt.plot(np.arange(0, dt*nframes, dt), q_true_alt[:nframes, 3], label='True equiv.', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle q_3$')
#plt.title('Orientation $\displaystyle q_3$')

metrics = np.array(metrics)
fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), metrics[:,0], label='Perfect Metric', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), metrics[:,1], label='Our Metric', linewidth=1)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Metric')

"""
fig = plt.figure()
true_b = []
for i in range(nframes):
    true_b.append(Rot_L_to_B[i] @ [1,1,1])
true_b = np.array(true_b)
plt.plot(np.arange(0, dt*nframes, dt), true_b[:,2], label='True')

plt.plot(np.arange(0, dt*nframes, dt), omega_kabsch_b[:,2], label='Computed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('$\displaystyle {}^B \Omega_z$ from Kabsch')
"""

plt.show()