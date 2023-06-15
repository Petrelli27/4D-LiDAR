import copy
import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import linalg
import math
import lidarScan2
import boundingbox
from stl import mesh
from estimateOmega import estimate_LLS, estimate_kabsch, estimate_rotation_B
from associationdata import nearest_search
from associationdata import rotation_association
import pickle

from mpl_toolkits import mplot3d
from matplotlib import pyplot

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

    hamilton = [-omega_k[0] * q_k[1] - omega_k[1] * q_k[2] - omega_k[2] * q_k[3],
                omega_k[0] * q_k[0] + omega_k[2] * q_k[2] - omega_k[1] * q_k[3],
                omega_k[1] * q_k[0] - omega_k[2] * q_k[1] + omega_k[0] * q_k[3],
                omega_k[2] * q_k[0] + omega_k[1] * q_k[1] - omega_k[0] * q_k[2]]

    return 0.5 * dt * hamilton + q_k

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

# initialize debris position, velocity and orientation
O_B = np.array([0,0,0])
O_L = np.array([0,0,0])

with open('sim_kompsat670.pickle', 'rb') as sim_data:
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
nframes = len(VBs)


# Running the simulation - Initializations

# Initializations in L Frame
vT_0 = [1., 1., 1.]  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous

# Initial covariance
P_0 = np.diag([0.25, 0.5, 0.25, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25])  # Initial Covariance matrix
P_k = P_0.copy()  # covariance matrix

# Process noise covariance matrix
qp = 0.000025
qv = 0.0000005
qom = 0.00005
qp1 = 0.00005
qq = 0.0005
Q = np.diag([qp, qp, qp, qv, qv, qv, qom, qom, qom, qp1, qp1, qp1, qq, qq, qq, qq])

# Measurement noise covariance matrix
p = 500
om = 0.25
p1 = 0.05
q = 0.01
R = np.diag([p, p, p, om, om, om, p1, p1, p1, q, q, q, q])

# Measurement matrix
H = np.zeros([len(P_0)-3, len(P_0)])  # no measuring of velocity
H[0:3,0:3] = np.eye(3)
H[3:,6:] = np.eye(10)

# Kabsch estimation parameters
n_moving_average = 100
settling_time = 500
# Record keeping for angular velocity estimate
omegas_kabsch_b = np.zeros((nframes, 3))
omegas_lls_b = np.zeros((nframes, 3))
omega_kabsch_b_box = np.zeros((n_moving_average,3))

for i in range(nframes):

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
        q_kp1 = orientationupdate(dt, x_k)

        # Compute Jacobian
        F_kp1 = F_matrix(dt, R_k_kp1, x_k)

        # Update Covariance
        P_kp1 = np.matmul(F_kp1, np.matmul(P_k, F_kp1.T)) + Q

        # Make updated State vector
        x_kp1 = np.array([p_kp1, v_kp1, omega_kp1, p1_kp1, q_kp1]).ravel()

    #######################
    # Measurements
    #######################

    PLs.append((linalg.inv(Rot_L_to_B[i]) @ (PBs[i]).T).T)
    # find bounding box from points
    XLs.append(PLs[i][:, 0])
    YLs.append(PLs[i][:, 1])
    ZLs.append(PLs[i][:, 2])
    X_i = XLs[i]
    Y_i = YLs[i]
    Z_i = ZLs[i]

    # Return bounding box and centroid estimate of bounding box
    z_pi_k, z_p_k, R_1 = boundingbox.bbox3d(X_i, Y_i, Z_i, True) # unassociated bbox

    # Orientation association
    # R_1 is obtained from bounding box
    if i == 0:
        z_q_k = Rotation.as_quat(Rotation.from_matrix(R_1))
    else:
        z_q_k = rotation_association(q_kp1, R_1)

    associatedBbox, L, W, H = boundingbox.associated(z_q_k, z_pi_k, z_p_k)
    z_p1_k = associatedBbox[:,1]
    # find angular velocity from LOS velocities
    # 1. Linear Least Squares
    omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i]@p_k, Rot_L_to_B[i]@v_k, VBs[i])
    omega_LLS = Rot_B_to_L[i] @ omega_LLS_B

    ################################ I think to use Kabsch you need i > 0, to wait for state initializations?
    # 2. Rotation of B Frame
    omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)

    # 3. Kabsch
    if i==0:
        omega_los_L = np.array([0,0,0])
    else:
        cur_box_L = np.transpose(copy.deepcopy(associatedBbox))
        cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
        # rotate previous box with everything else
        prev_box_B = (rodrigues((omega_LLS + omega_L_to_B), dt) @ prev_box_B.T).T
        omega_los_B = estimate_kabsch(prev_box_B, cur_box_B, dt)
        prev_box_B = cur_box_B # for next iteration

        # using moving average to smooth out omega_los_B
        omega_kabsch_b_box[i%n_moving_average] = omega_los_B
        if i < n_moving_average:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i+1], axis=0)
        else:
            omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
        omega_los_L = Rot_B_to_L[i]@omega_los_B_averaged

    # Combine angular velocity estimates
    if i <= settling_time:
        z_omega_k = omega_LLS + omega_L_to_B # ignores kabsch
    else:
        z_omega_k = omega_LLS + omega_L_to_B + omega_los_L
    #################################

    # Compute Measurement Vector
    z_kp1 = np.array([z_p_k, z_omega_k, z_p1_k, z_q_k]).ravel()

    # Set initial states to measurements
    if i == 0:
        x_k = np.array([z_p_k, z_omega_k, vT_0, z_p1_k, z_q_k]).ravel()

    ##############
    # Update - Combine Measurement and Estimates
    ##############

    # allow for some time for states to settle
    if i > 300:
        if abs(np.linalg.norm(z_p_k - p_kp1)) > 0.3:
            pass
        else:
            # Calculate the Kalman gain
            K_kp1 = np.matmul(P_kp1, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(P_kp1, H.T)) + R)))

            # Calculate Residual
            res_kp1 = z_kp1 - np.matmul(H, x_kp1)

            # Update State
            x_kp1 = x_kp1 + np.matmul(K_kp1, res_kp1)

            # Update Covariance
            P_kp1 = np.matmul(np.eye(len(K_kp1)) - K_kp1 @ H, P_kp1)

    # Transfer states and covariance from kp1 to k
    P_k = P_kp1.copy()
    x_k = x_kp1.copy()

    # Append for analysis
    z_s.append(z_p_k)
    P_s.append(P_k)
    x_s.append(x_k)


##############
# Plot relevant figures
##############