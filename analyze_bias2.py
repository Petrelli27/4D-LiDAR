import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan2
import boundingbox
from stl import mesh
from estimateOmega import estimate_LLS as estimate
from associationdata import nearest_search
import pickle

from mpl_toolkits import mplot3d
from matplotlib import pyplot

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

def drawrectangle2(ax, p1, p2, p3, p4, p5, p6, p7, p8, color):
    # z1 plane boundary
    #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)  # W
    #ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color)
    #ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color)
    #ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=color)

    # z1 plane boundary
    #ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color)  # W
    #ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color)
    #ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color)
    #ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=color)

    # Connecting
    #ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=color)  # W
    #ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color)
    #ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color)
    #ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='b')

# some utility functions
def tilde(v):
    vx = v[0,0]
    vy = v[1,0]
    vz = v[2,0]
    v_tilde = np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]])
    return v_tilde

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
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + tilde(e)*np.sin(phi)
    return R.T

def rodrigues(omega, dt):
    
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R

def F_matrix(dt, R, p_k, p1_k, p2_k, p3_k, p4_k, p5_k, p6_k, p7_k, p8_k):
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
    F = np.eye(36)

    # p_k - dv/dp_k
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # p1_k - dv/dp1_k
    F[9, 3] = dt
    F[10, 4] = dt
    F[11, 5] = dt

    # dp1_k/dp1_k
    F[9:12, 9:12] = R

    # dp1_k/dp_k
    F[9:12, 0:3] = 1 - R

    # dp1_k/domega
    F[9:12, 6:9] = skew(p1_k - p_k)

    # p2_k
    F[12, 3] = dt
    F[13, 4] = dt
    F[14, 5] = dt
    F[12:15, 12:15] = R
    F[12:15, 0:3] = 1 - R
    F[12:15, 6:9] = skew(p2_k - p_k)

    # p3_k
    F[15, 3] = dt
    F[16, 4] = dt
    F[17, 5] = dt
    F[15:18, 15:18] = R
    F[15:18, 0:3] = 1 - R
    F[15:18, 6:9] = skew(p3_k - p_k)

    # p4_k
    F[18, 3] = dt
    F[19, 4] = dt
    F[20, 5] = dt
    F[18:21, 18:21] = R
    F[18:21, 0:3] = 1 - R
    F[18:21, 6:9] = skew(p4_k - p_k)

    # p5_k
    F[21, 3] = dt
    F[22, 4] = dt
    F[23, 5] = dt
    F[21:24, 21:24] = R
    F[21:24, 0:3] = 1 - R
    F[21:24, 6:9] = skew(p5_k - p_k)

    # p6_k
    F[24, 3] = dt
    F[25, 4] = dt
    F[26, 5] = dt
    F[24:27, 24:27] = R
    F[24:27, 0:3] = 1 - R
    F[24:27, 6:9] = skew(p6_k - p_k)

    # p7_k
    F[27, 3] = dt
    F[28, 4] = dt
    F[29, 5] = dt
    F[27:30, 27:30] = R
    F[27:30, 0:3] = 1 - R
    F[27:30, 6:9] = skew(p7_k - p_k)

    # p8_k
    F[30, 3] = dt
    F[31, 4] = dt
    F[32, 5] = dt
    F[30:33, 30:33] = R
    F[30:33, 0:3] = 1 - R
    F[30:33, 6:9] = skew(p8_k - p_k)

    return F

def verticeupdate(dt, x_k):

    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]
    p2_k = x_k[12:15]
    p3_k = x_k[15:18]
    p4_k = x_k[18:21]
    p5_k = x_k[21:24]
    p6_k = x_k[24:27]
    p7_k = x_k[27:30]
    p8_k = x_k[30:33]

    # Rotation matrix - rodrigues formula
    e_omega = omega_k / np.linalg.norm(omega_k)  # Unit vector along omega
    phi = np.linalg.norm(omega_k) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R_k_kp1 = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)

    c_k = (p1_k + p2_k + p3_k + p4_k + p5_k + p6_k + p7_k + p8_k)/8

    # Translate vertices to origin
    p1_ko = p1_k - c_k
    p2_ko = p2_k - c_k
    p3_ko = p3_k - c_k
    p4_ko = p4_k - c_k
    p5_ko = p5_k - c_k
    p6_ko = p6_k - c_k
    p7_ko = p7_k - c_k
    p8_ko = p8_k - c_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1, p1_ko.reshape(len(p1_ko), 1))
    p2_kp1o = np.matmul(R_k_kp1, p2_ko.reshape(len(p2_ko), 1))
    p3_kp1o = np.matmul(R_k_kp1, p3_ko.reshape(len(p3_ko), 1))
    p4_kp1o = np.matmul(R_k_kp1, p4_ko.reshape(len(p4_ko), 1))
    p5_kp1o = np.matmul(R_k_kp1, p5_ko.reshape(len(p5_ko), 1))
    p6_kp1o = np.matmul(R_k_kp1, p6_ko.reshape(len(p6_ko), 1))
    p7_kp1o = np.matmul(R_k_kp1, p7_ko.reshape(len(p7_ko), 1))
    p8_kp1o = np.matmul(R_k_kp1, p8_ko.reshape(len(p8_ko), 1))

    # Translate them back to new expected origin
    p1_kp1 = (p1_kp1o.T + p_k + v_k * dt).ravel()
    p2_kp1 = (p2_kp1o.T + p_k + v_k * dt).ravel()
    p3_kp1 = (p3_kp1o.T + p_k + v_k * dt).ravel()
    p4_kp1 = (p4_kp1o.T + p_k + v_k * dt).ravel()
    p5_kp1 = (p5_kp1o.T + p_k + v_k * dt).ravel()
    p6_kp1 = (p6_kp1o.T + p_k + v_k * dt).ravel()
    p7_kp1 = (p7_kp1o.T + p_k + v_k * dt).ravel()
    p8_kp1 = (p8_kp1o.T + p_k + v_k * dt).ravel()

    return p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, R_k_kp1

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
Rot_B_to_L = [r.T for r in Rot_L_to_B]
omega_L = data[8]
dt = data[9]
#debris = data[10]
# Estimation Loop
XLs = []
YLs = []
ZLs = []
PLs = []
VLs = VBs


# Running the simulation
nframes = len(VBs)
# Estimation Loop
XLs = []
YLs = []
ZLs = []
PLs = []
VLs = VBs


# Running the simulation
# Initializations in L Frameeeee
L_0 = 2  # Initial Length of box - x
D_0 = 2  # Initial Width of box - y
H_0 = 2  # Initial Height of box - z
p_0 = [-170., -350., -20.]#np.array(debris_pos[0])  # Guess of initial position of debris - *need to formulate guess*
v_0 = [1., 1., 1.]#np.array(debris_vel[0])  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = [5.,-5.,4.] #np.array(omega_L)  # Initial guess of angular velocities - *need to formulate guess*

# For the initializations, imagine a bounding box bottom right = p1, bottom left = p2, TP = p3, TR = p4, p5, p6, p7, p8 are
# the same but in the back, centered at p0
p1_0 = p_0 + np.array([L_0/2, -D_0/2, -H_0/2])
p2_0 = p_0 + np.array([-L_0/2, -D_0/2, -H_0/2])
p3_0 = p_0 + np.array([-L_0/2, -D_0/2, H_0/2])
p4_0 = p_0 + np.array([L_0/2, -D_0/2, H_0/2])
p5_0 = p_0 + np.array([L_0/2, D_0/2, -H_0/2])
p6_0 = p_0 + np.array([-L_0/2, D_0/2, -H_0/2])
p7_0 = p_0 + np.array([-L_0/2, D_0/2, H_0/2])
p8_0 = p_0 + np.array([L_0/2, D_0/2, H_0/2])
# bz_0 = [0.05, 0.05, 0.0]  # measurement bias
bz_0 = [0., 0., 0.]  # measurement bias
x_0 = np.array([p_0, v_0, omega_0, p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0, bz_0]).ravel()  # Initial State vector

P_0 = np.diag([0.25, 0.5, 0.25, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25,
               0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.1, 0.1, 0.1])  # Initial Covariance matrix
# Process noise covariance matrix
qpixz = 0.00005
qpyz = 0.000025
qpxyz = 0.0000001
qv = 0.0000005
qom = 0.00005
qbz = 0.000000000005  # measurement bias covariance
Q = np.diag([qpxyz, qpxyz, qpxyz, qv, qv, qv, qom, qom, qom, qpixz, qpyz, qpixz, qpixz, qpyz, qpixz, qpixz, qpyz, qpixz, qpixz, qpyz, qpixz,
               qpixz, qpyz, qpixz, qpixz, qpyz, qpixz, qpixz, qpyz, qpixz, qpixz, qpyz, qpixz, qbz, qbz, qbz])
# Measurement noise covariance matrix
pxz = 500
py = 500
om = 0.25
vn = 0.01
pxyz = 0.05
pyy = 0.05

# adaptive quantities
pxyz_res_geq_0 = 0.5
qxyz_res_geq_0 = 0.00005
pxyz_res_seq_0 = 0.00005
qxyz_res_seq_0 = 0.5

R = np.diag([pxyz, pyy, pxyz, om, om, om, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz,
               pxz, py, pxz, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz])
#R = np.diag([pxz, py, pxz, vn, vn, vn, om, om, om, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz,
 #              pxz, py, pxz, pxz, py, pxz, pxz, py, pxz, pxz, py, pxz])

H = np.zeros([len(x_0)-3-3, len(x_0)])  # take away 3 additional one for the measurement bias
h_1 = np.eye(3)
h_2 = np.eye(27)
H[0:3,0:3] = h_1
H[3:, 6:33] = h_2
H[:3, 33:] = h_1
# H[6:9, 33:] = h_1
# H[9:12, 33:] = h_1
# H[12:15, 33:] = h_1
# H[15:18, 33:] = h_1
# H[18:21, 33:] = h_1
# H[21:24, 33:] = h_1
# H[24:27, 33:] = h_1
# H[27:30, 33:] = h_1

#H = np.eye(33)

# print(debris_pos[0])
# print(debris_vel[0])
# Current states
x_k = x_0.copy()  # State vector
P_k = P_0.copy()  # covariance matrix

# Get Final measurement vectors
z_p_s = [p_0]
zv_mags = []
z_omegas = []
z_v_s = []
z_s = []
x_s = [x_0]
P_s = []
reses = []

#print(i)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.legend()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

n_moving_average = 100
settling_time = 500
omega_kabsch_b = np.zeros((nframes, 3))
omega_lls_b = np.zeros((nframes, 3))
omega_kabsch_b_box = np.zeros((n_moving_average,3))

# convert points and LOS velocities to {L}
for i in range(nframes):

    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]
    p2_k = x_k[12:15]
    p3_k = x_k[15:18]
    p4_k = x_k[18:21]
    p5_k = x_k[21:24]
    p6_k = x_k[24:27]
    p7_k = x_k[27:30]
    p8_k = x_k[30:33]
    bz_k = x_k[33:]  # meas bias

    #ax.scatter(x_k[0], x_k[1], x_k[2], color='r')

    z_v_s.append(v_k)
    ##############
    # Update state
    ##############

    # Centroid from vertices
    c_k = (p1_k + p2_k + p3_k + p4_k + p5_k + p6_k + p7_k + p8_k) / 8.

    # Position update
    p_kp1 = v_k * dt + p_k

    # Velocity update
    v_kp1 = v_k.copy()

    # Angular velocity update
    omega_kp1 = omega_k.copy()

    # Vertice updates
    p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, R_k_kp1 = verticeupdate(dt, x_k)

    if i > 300:
        # measurement bias
        bz_kp1 = bz_k.copy()
    else:
        bz_kp1 = bz_0.copy()

    # Final box
    #drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'g')

    # Compute Jacobian
    F_kp1 = F_matrix(dt, R_k_kp1, p_k, p1_k, p2_k, p3_k, p4_k, p5_k, p6_k, p7_k, p8_k)

    # Update Covariance
    P_kp1 = np.matmul(F_kp1, np.matmul(P_k, F_kp1.T)) + Q


    # Make updated State vector
    x_kp1 = np.array([p_kp1, v_kp1, omega_kp1, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, bz_kp1]).ravel()

    #######################
    # Make measurements
    #######################
    PLs.append((np.linalg.inv(Rot_L_to_B[i]) @ (PBs[i]).T).T)
    # find bounding box from points
    XLs.append(PLs[i][:, 0])
    YLs.append(PLs[i][:, 1])
    ZLs.append(PLs[i][:, 2])
    X_i = XLs[i]
    Y_i = YLs[i]
    Z_i = ZLs[i]

    # Return bounding box and centroid estimate of bounding box
    z_pi_k, z_p_k = boundingbox.bbox3d(X_i, Y_i, Z_i)

    # Vertice association
    pi_pk1 = [p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1]
    z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k = nearest_search(pi_pk1-p_k-v_k*dt, z_pi_k, z_p_k)



    #ax.scatter(debris_pos[i, 0], debris_pos[i, 1], debris_pos[i, 2], color='g')
    #ax.scatter(X_i, Y_i, Z_i)
    #ax.scatter(z_p_k[0], z_p_k[1], z_p_k[2], color='b')
    # ax.add_collection3d(mplot3d.art3d.Poly3DCollection(debris.vectors, alpha=0.3))
    # drawrectangle(ax, z_pi_k[:,0], z_pi_k[:,1], z_pi_k[:,2], z_pi_k[:,3], z_pi_k[:,4], z_pi_k[:,5], z_pi_k[:,6], z_pi_k[:,7], 'b')
    if False:
    # if i>1000 and i%10 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.scatter(X_i, Y_i, Z_i, color='black', marker='o', s=0.5)
        drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'r')
        drawrectangle(ax, z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k, 'b')
        ax.scatter(x_k[0], x_k[1], x_k[2], color='r' )
        ax.scatter(z_p_k[0], z_p_k[1], z_p_k[2], color='b')
        ax.scatter(debris_pos[i,0], debris_pos[i,1], debris_pos[i,2], color='g')

    
        plt.show()
    # Estimate linear velocity
    #z_v_k1 = (np.array(z_p_k) - np.array(z_p_s[i-1]))/dt

    #z_v_k_mag = np.mean(VBs[i])
    #zv_mags.append(z_v_k_mag)
    #print(debris_vel[i])
    #print(z_v_k_mag)
    #z_v_k = z_v_k_mag*np.array(-(debris_pos[i]))/np.linalg.norm(debris_pos[i])
    #z_v_k = z_v_k_mag*-(Rot_L_to_B[i].T@np.array([0,0,1]))
    #z_v_k = np.dot(z_v_k, debris_vel[i]/np.linalg.norm(debris_vel[i]))*(debris_vel[i]/np.linalg.norm(debris_vel[i]))
    #z_v_s.append(z_v_k)
    #z_v_k = debris_vel[i]

    #print(z_v_k1)
    #print(z_v_k)
    #print(debris_vel[i])

    # find angular velocity from LOS velocities
    #c = debris_pos[i]
    #v_c = debris_vel[i]
    # z_omega_k = estimate(X_i, Y_i, Z_i, p_k, v_k, VBs[i])
    z_omega_k_B = estimate(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i]@p_k, Rot_L_to_B[i]@v_k, VBs[i])
    z_omega_k = Rot_B_to_L[i] @ z_omega_k_B
    # z_omega_k = estimate(X_i, Y_i, Z_i, c, v_c, VBs[i])
    # want to rotate pi_pk1 about u_los (z axis in B frame or z_p_k (debris_pos) in L frame) to best match z_pi_k, and then find that angle
    # uses Kabsch algorithm
    if i==0:
        prev_box_L = np.array([z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k])
        prev_box_B = (Rot_L_to_B[i] @ prev_box_L.T).T
    cur_box_L = np.array([z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k])
    # cur_box_L = np.array(pi_pk1)
    cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
    prev_box_B_centroid = np.mean(prev_box_B, axis=0)
    cur_box_B_centroid = np.mean(cur_box_B, axis=0)
    prev_box_B -= prev_box_B_centroid # line up centroids
    cur_box_B -= cur_box_B_centroid
    # apply rotation obtained using linear least squares (everything except about z_b)
    prev_box_B = (rodrigues(z_omega_k_B, dt) @ prev_box_B.T).T 
    prev_rec_B = prev_box_B[:,0:2] # remove z since it doesn't matter
    cur_rec_B = cur_box_B[:,0:2]
    
    cov_rec_B = prev_rec_B.T @ cur_rec_B # covariance
    prev_box_B = cur_box_B # for next iteration

    d = 1 if np.linalg.det(cov_rec_B) >= 0 else -1
    U, S, Vh = np.linalg.svd(cov_rec_B)
    Rot_los = Vh.T @ np.array([[1,0],[0,d]]) @ U.T
    # Rot_los = Rot_los.T # why did we want a transpose here? should remove this line
    angle_rot_los = np.arctan2(Rot_los[1,0],Rot_los[0,0])
    omega_los_B = np.array([0,0,angle_rot_los / dt])

    # using moving average to smooth out omega_los_B
    omega_kabsch_b_box[i%n_moving_average] = omega_los_B
    if i < n_moving_average:
        omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i+1], axis=0)
    else:
        omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
    omega_los_L = Rot_B_to_L[i]@omega_los_B_averaged
    # if angle_rot_los/dt > 10: print(i, angle_rot_los/dt)

    # angular velocity of B to L
    if i == 0:
        omega_L_to_B = np.array([0,0,0])
    else:
        Rlb = Rot_L_to_B[i-1].T @ Rot_L_to_B[i]  # shorthand
        # Rlb = (Rot_L_to_B[i] - Rot_L_to_B[i-1])/dt
        angle_B_to_B = np.arccos((np.trace(Rlb) - 1)/2)
        axis_B_to_B = 1./(2*np.sin(angle_B_to_B))*np.array([Rlb[2,1]-Rlb[1,2],Rlb[0,2]-Rlb[2,0],Rlb[1,0]-Rlb[0,1]])
        axis_B_to_B = Rot_B_to_L[i] @ axis_B_to_B / np.linalg.norm(axis_B_to_B)
        omega_L_to_B = (angle_B_to_B * axis_B_to_B)/dt
        # print(omega_L_to_B, angle_B_to_B)
    if i <= settling_time:
        z_omega_k = z_omega_k + omega_L_to_B # ignores kabsch
    else:
        z_omega_k = z_omega_k + omega_los_L + omega_L_to_B
    omega_lls_b[i,:] = z_omega_k_B
    omega_kabsch_b[i,:] = omega_los_B_averaged
    # z_omega_k = [1., 1., 1.]
    z_omegas.append(z_omega_k)

    # Get Measurement Vector
    #z_kp1 = np.array(
     #   [z_p_k, z_v_k, z_omega_k, z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k]).ravel()


    z_kp1 = np.array(
        [z_p_k, z_omega_k, z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k]).ravel()

    ####################

    #if False:
    # if i > 300 and abs(np.linalg.norm(z_p_k - p_kp1)) > 0.3:
    #     pass
    # else:
        # Calculate the Kalman gain
        # print(P_kp1.shape)
        # print(H.shape)
        # print(R.shape)
    K_kp1 = np.matmul(P_kp1, np.matmul(H.T, np.linalg.inv(np.matmul(H, np.matmul(P_kp1, H.T)) + R)))



    # Calculate Residual
    res_kp1 = z_kp1 - np.matmul(H, x_kp1)

    if res_kp1[0] > 0:
        pass
    else:
        R[0, 0] = 1e-3 * R[0, 0]
    if res_kp1[1] > 0:
        pass
    else:
        R[1, 1] = 1e-3 * R[1, 1]
    if res_kp1[2] > 0:
        pass
    else:
        R[2, 2] = 1e-3 * R[2, 2]

    print(i)
    print(K_kp1[0, :])
    print(z_kp1[:3])
    print(x_kp1[:3])
    print(res_kp1[:3])
    print(x_kp1[-3:])


    reses.append(res_kp1[0])


    # Update State
    x_kp1 = x_kp1 + np.matmul(K_kp1, res_kp1)
    print(x_kp1[:3])
    print(x_kp1[-3:])
    print('\n')
    # Update Covariance
    P_kp1 = np.matmul(np.eye(len(K_kp1)) - K_kp1@H, P_kp1)

    # Transfer states and covariance from kp1 to k
    P_k = P_kp1.copy()
    x_k = x_kp1.copy()

    # Append for analysis
    z_p_s.append(z_p_k)
    P_s.append(P_k)
    x_s.append(x_k)
    #z_s.append(z_kp1)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.scatter(X_i, Y_i, Z_i)
        true_omega_B = Rot_L_to_B[i] @ [1,1,1]
        #true_omega_B_z = np.array([0,0,true_omega_B[2]])
        true_Rot = rodrigues(true_omega_B, dt)
        true_rec_B = (true_Rot[0:2,0:2] @ prev_rec_B.T).T

        prev_rotated = (Rot_los.T @ prev_rec_B.T).T
        
        ax.scatter(prev_rec_B[:,0], prev_rec_B[:,1], color='b' , label='previous predicted')
        ax.scatter(cur_rec_B[:,0], cur_rec_B[:,1], color='r' ,label='current predicted')
        ax.scatter(prev_rotated[:,0], prev_rotated[:,1], color = 'g', label = 'kabsch from previous to current')
        ax.scatter(true_rec_B[:,0], true_rec_B[:,1], color='y', label='true from previous to current')
        ax.legend()
        # print(angle_rot_los)
        plt.show()

#print(z_p_s)
#print(debris_pos)
z_p_s = np.array(z_p_s)
z_omegas = np.array(z_omegas)
z_v_s = np.array((z_v_s))
x_s = np.array(x_s)

x_rmse = np.sqrt(np.mean(np.square(debris_pos[:,0] - z_p_s[:len(debris_pos),0])))
y_rmse = np.sqrt(np.mean(np.square(debris_pos[:,1] - z_p_s[:len(debris_pos),1])))
z_rmse = np.sqrt(np.mean(np.square(debris_pos[:,2] - z_p_s[:len(debris_pos),2])))
stdx = np.std(debris_pos[:,0] - z_p_s[:len(debris_pos),0])
stdy = np.std(debris_pos[:,1] - z_p_s[:len(debris_pos),1])
stdz = np.std(debris_pos[:,2] - z_p_s[:len(debris_pos),2])

omegax_rmse = np.sqrt(np.mean(np.square(z_omegas[:,0] - 1)))
omegay_rmse = np.sqrt(np.mean(np.square(z_omegas[:,1] - 1)))
omegaz_rmse = np.sqrt(np.mean(np.square(z_omegas[:,2] - 1)))
stdomegax = np.std(z_omegas[:,0] - 1)
stdomegay = np.std(z_omegas[:,1] - 1)
stdomegaz = np.std(z_omegas[:,2] - 1)

vx_rmse = np.sqrt(np.mean(np.square(z_v_s[:,0] - debris_vel[:,0])))
vy_rmse = np.sqrt(np.mean(np.square(z_v_s[:,1] - debris_vel[:,1])))
vz_rmse = np.sqrt(np.mean(np.square(z_v_s[:,2] - debris_vel[:,2])))
stdvx = np.std(z_v_s[:,0] - debris_vel[:,0])
stdvy = np.std(z_v_s[:,1] - debris_vel[:,1])
stdvz = np.std(z_v_s[:,2] - debris_vel[:,2])

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
ax.scatter(z_p_s[:,0], z_p_s[:,1], z_p_s[:,2], color='b', s=0.3, linewidths=0)
ax.plot(debris_pos[:,0], debris_pos[:,1], debris_pos[:,2], color='g')
plt.legend(['Start', 'End', 'Computed Centroid Positions', 'True Centroid Positions'])
plt.xlim([-170.5,-167.5])
plt.ylim([-351,-306])
ax.set_zlim(-20,-9)
# ax.set_aspect('equal')


m1 = len(x_s)
# print(m1)
"""
fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_omegas[:,0], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,6], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_x$ (rad/s)')
#plt.title('$\displaystyle\Omega_x$')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_omegas[:,1], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,7], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_y$ (rad/s)')
#plt.title('Omega Y')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_omegas[:,2], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,8], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), np.ones([nframes,1]), label='True', linewidth=1, linestyle='dashed')
plt.legend()

plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle\Omega_z$ (rad/s)')
#plt.title('Omega Z')
"""
fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,6] - 1, label='Error $\displaystyle\Omega_x$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,7] - 1, label='Error $\displaystyle\Omega_y$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,8] - 1, label='Error $\displaystyle\Omega_z$', linewidth=2)
# plt.plot(np.arange(0, dt*nframes, dt), np.zeros([nframes,1]), linewidth = 1) # draw line at zero
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity Error (rad/s)')
#plt.title('Angular Velocity Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,0] - debris_pos[:,0], label='Error $\displaystyle p_x$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,1] - debris_pos[:,1], label='Error $\displaystyle p_y$', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,2] - debris_pos[:,2], label='Error $\displaystyle p_z$', linewidth=2)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position Error (m)')
#plt.title('Position Errors')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_p_s[1:,0], label='Computed', linewidth=1)

plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,0], label='Estimated', linewidth=2)

plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,0], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_x$ (m)')
#plt.title('X Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_p_s[1:,1], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,1], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,1], label='True', linewidth=1, linestyle='dashed')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_y$ (m)')
#plt.title('Y Position')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_p_s[1:,2], label='Computed', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,2], label='Estimated', linewidth=2)
plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,2], label='True', linewidth=1, linestyle='dashed')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle p_z$ (m)')
#plt.title('Z Position')


fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:,0] - debris_vel[:,0], label='Error $\displaystyle v_{Tx}$', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:,1] - debris_vel[:,1], label='Error $\displaystyle v_{Ty}$', linewidth=1)
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:,2] - debris_vel[:,2], label='Error $\displaystyle v_{Tz}$', linewidth=1)

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Velocity Error (m/s)')
#plt.title('Velocity Errors')

"""
fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:, 0], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:, 0], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Tx}$ (m/s)')
#plt.title('Velocity in X')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:,1], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:,1], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Ty}$ (m/s)')
#plt.title('Velocity in y')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), z_v_s[:,2], label='Estimated')
plt.plot(np.arange(0, dt*nframes, dt), debris_vel[:,2], label='True')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('$\displaystyle v_{Tz}$ (m/s)')
#plt.title('Velocity in z')

fig = plt.figure()
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,9], label='$\displaystyle p_{1x}$')
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,10], label='$\displaystyle p_{1y}$')
plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,11], label='$\displaystyle p_{1z}$')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Vertex $\displaystyle p_{1}$ Position (m)')
#plt.title('Position of Vertice P1 overt time')

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

fig = plt.figure()
plt.plot(np.arange(0, dt * nframes, dt), x_s[:m1 - 1, 33], label='xbias')
plt.plot(np.arange(0, dt * nframes, dt), reses, label='resx')
# plt.plot(np.arange(0, dt * nframes, dt), x_s[:m1 - 1, 34], label='ybias')
# plt.plot(np.arange(0, dt * nframes, dt), x_s[:m1 - 1, 35], label='zbias')
plt.legend()
plt.show()