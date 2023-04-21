import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan
import boundingbox

def skew(vector):
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


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
    F = np.eye(33)

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

def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color):
    # z1 plane boundary
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color)  # W
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color)
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=color)

    # z1 plane boundary
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color)  # W
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color)
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color)
    ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=color)

    # Connecting
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=color)  # W
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color)
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color)
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='b')

# Initializations in L Frameeeee
L_0 = 2  # Initial Length of box - x
D_0 = 2  # Initial Width of box - y
H_0 = 2  # Initial Height of box - z
p_0 = np.array([0., 0., 400.])  # Guess of initial position of debris - *need to formulate guess*
v_0 = np.array([0., 0., 1.])  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = np.array([0., np.pi/4., 0.])  # Initial guess of angular velocities - *need to formulate guess*
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
x_0 = np.array([p_0, v_0, omega_0, p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0]).ravel()  # Initial State vector
P_0 = np.diag([5., 5., 5., 1., 1., 1., 0.9, 0.9, 0.9, 2.1, 2.1, 2.1, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.4, 2.4, 2.4,
               2.5, 2.5, 2.5, 2.6, 2.6, 2.6, 2.7, 2.7, 2.7, 2.8, 2.8, 2.8])  # Initial Covariance matrix
# Process noise covariance matrix
Q = np.diag([5., 5., 5., 1., 1., 1., 0.9, 0.9, 0.9, 2.1, 2.1, 2.1, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.4, 2.4, 2.4,
               2.5, 2.5, 2.5, 2.6, 2.6, 2.6, 2.7, 2.7, 2.7, 2.8, 2.8, 2.8])
# Measurement noise covariance matrix
R = np.diag([5., 5., 5., 1., 1., 1., 0.9, 0.9, 0.9, 2.1, 2.1, 2.1, 2.2, 2.2, 2.2, 2.3, 2.3, 2.3, 2.4, 2.4, 2.4,
               2.5, 2.5, 2.5, 2.6, 2.6, 2.6, 2.7, 2.7, 2.7, 2.8, 2.8, 2.8])

rate = 1  # Hz
dt = 1/rate  # Rate in time
endtime = 4  # Endtime of simulation in seconds
nframes = rate*endtime

# Current states
x_k = x_0.copy()  # State vector
P_k = P_0.copy()  # covariance matrix

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Initial box
drawrectangle(ax, x_k[9:12], x_k[12:15], x_k[15:18], x_k[18:21], x_k[21:24], x_k[24:27], x_k[27:30], x_k[30:33], 'b')

# Just for simulations
poses = np.ones([3, nframes])
vels = 2*np.ones([3, nframes])
omegas = 3*np.ones([3, nframes])
p1s = 4*np.ones([3, nframes])
p2s = 5*np.ones([3, nframes])
p3s = 6*np.ones([3, nframes])
p4s = 7*np.ones([3, nframes])
p5s = 8*np.ones([3, nframes])
p6s = 9*np.ones([3, nframes])
p7s = 10*np.ones([3, nframes])
p8s = 11*np.ones([3, nframes])

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

    ##############
    # Update state
    ##############

    # Centroid from vertices
    c_k = (p1_k + p2_k + p3_k + p4_k + p5_k + p6_k + p7_k + p8_k)/8.

    # Position update
    p_kp1 = v_k*dt + (p_k + c_k)/2

    # Velocity update
    v_kp1 = v_k.copy()

    # Angular velocity update
    omega_kp1 = omega_k.copy()

    #######
    # Vertice updates
    #######

    # Rotation matrix - rodrigues formula
    e_omega = omega_k/np.linalg.norm(omega_k)  # Unit vector along omega
    phi = np.linalg.norm(omega_k)*dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R_k_kp1 = ee_t + (np.eye(len(e_omega)) - ee_t)*np.cos(phi) + e_tilde*np.sin(phi)

    # Translate vertices to origin
    p1_ko = p1_k - p_k
    p2_ko = p2_k - p_k
    p3_ko = p3_k - p_k
    p4_ko = p4_k - p_k
    p5_ko = p5_k - p_k
    p6_ko = p6_k - p_k
    p7_ko = p7_k - p_k
    p8_ko = p8_k - p_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1.T, p1_ko.reshape(len(p1_ko), 1))
    p2_kp1o = np.matmul(R_k_kp1.T, p2_ko.reshape(len(p2_ko), 1))
    p3_kp1o = np.matmul(R_k_kp1.T, p3_ko.reshape(len(p3_ko), 1))
    p4_kp1o = np.matmul(R_k_kp1.T, p4_ko.reshape(len(p4_ko), 1))
    p5_kp1o = np.matmul(R_k_kp1.T, p5_ko.reshape(len(p5_ko), 1))
    p6_kp1o = np.matmul(R_k_kp1.T, p6_ko.reshape(len(p6_ko), 1))
    p7_kp1o = np.matmul(R_k_kp1.T, p7_ko.reshape(len(p7_ko), 1))
    p8_kp1o = np.matmul(R_k_kp1.T, p8_ko.reshape(len(p8_ko), 1))

    # Translate them back
    p1_kp1 = (p1_kp1o.T + p_k + v_k*dt).ravel()
    p2_kp1 = (p2_kp1o.T + p_k + v_k*dt).ravel()
    p3_kp1 = (p3_kp1o.T + p_k + v_k*dt).ravel()
    p4_kp1 = (p4_kp1o.T + p_k + v_k*dt).ravel()
    p5_kp1 = (p5_kp1o.T + p_k + v_k*dt).ravel()
    p6_kp1 = (p6_kp1o.T + p_k + v_k*dt).ravel()
    p7_kp1 = (p7_kp1o.T + p_k + v_k*dt).ravel()
    p8_kp1 = (p8_kp1o.T + p_k + v_k*dt).ravel()

    # Final box
    drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'g')
    ##########

    # Compute Jacobian
    F_kp1 = F_matrix(dt, R_k_kp1, p_k, p1_k, p2_k, p3_k, p4_k, p5_k, p6_k, p7_k, p8_k)

    # Update Covariance
    P_kp1 = np.matmul(F_kp1, np.matmul(P_k, F_kp1.T)) + Q

    # Make updated State vector
    x_kp1 = np.array([p_kp1, v_kp1, omega_kp1, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1]).ravel()

    # Calculate the Kalman gain
    K_kp1 = np.matmul(P_kp1, np.linalg.inv(P_kp1 + R))

    # Get Measurement Vector
    z_kp1 = np.array([poses[:,i], vels[:,i], omegas[:,i], p1s[:,i], p2s[:,i], p3s[:,i], p4s[:,i], p5s[:,i], p6s[:,i],
                      p7s[:,i], p8s[:,i]]).ravel()

    # Calculate Residual
    #res_kp1 = z_kp1 - x_kp1
    res_kp1 = x_kp1 - x_kp1

    # Update State
    x_kp1 = x_kp1 + np.matmul(K_kp1, res_kp1)

    # Update Covariance
    P_kp1 = np.matmul(np.eye(len(K_kp1)) - K_kp1, P_kp1)

    # Transfer states and covariance from kp1 to k
    P_k = P_kp1.copy()
    x_k = x_kp1.copy()



# plt.show()








