import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan2
import boundingbox
from stl import mesh
from estimateOmega import estimate
from associationdata import nearest_search

from mpl_toolkits import mplot3d
from matplotlib import pyplot

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
# Dynamics initializations
r0 = [0, -0.0040, 0]  # initial starting position of chaser (km)
rdot0 = [-0.0001, 0.0, 0.0001]  # initial velocity of debris relative to chaser(km/s)
R = 500 + 6378  # Altitude of orbit (km)
mu = 398600.5  # Gravitational constant
omeg = math.sqrt(mu / R ** 3)  # n in the derivations
Rot_0 = np.identity(3) # initial starting rotation matrix/orientation
omega_L = np.array([1.,1.,1.]) # inertial, unchanging angular velocity of debris
omega_L_axis = omega_L/np.linalg.norm(omega_L)

# specify time frame and time step
nframes = 3
dt = 0.01

# simulate debris velocity (linear and angular) in {L} frame from dynamics
x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, omeg)
debris_pos = np.vstack([x,y,z]).T
debris_vel = np.vstack([vx,vy,vz]).T

# specify Lidar resolution and range
# LiDAR point cloud generation initializations
h_resolution = 40  # Number of rays horizontally
v_resolution = 40  # Number of rays vertically
h_range = 120  # Vertical lidar angle range in degrees
v_range = 60  # Horizontal lidar angle range in degrees


# generate debris mesh (we only want to do this once to improve compute efficiency)
debris_file = 'kompsat-1-v9.stl'
debris = mesh.Mesh.from_file(debris_file)  # Grab satellite mesh

# Simulation Loop
XBs = []
YBs = []
ZBs = []
PBs = []
VBs = []
Rot_L_to_B = []
for i in range(nframes):
    # solve for rotation matrix  B^R_L (R*L => B)
    Rot_L_to_B.append(getR(x[i],y[i],z[i]))
    # express the position and velocity in B
    debris_pos_B = Rot_L_to_B[i]@debris_pos[i]
    debris_vel_B = Rot_L_to_B[i]@debris_vel[i]

    # orient debris mesh in {B}
    if i==0: # move the debris to its initial location
        Rot_to_B = Rot_L_to_B[i]@Rot_0
        debris.rotate_using_matrix(Rot_to_B.T) # transpose because numpy stl rotation matrix is done backwards
    else:
        # rotation stacks upon previous rotations
        debris.rotate(Rot_L_to_B[i]@omega_L_axis,-np.linalg.norm(omega_L*dt))
    trans_to_B = debris_pos_B
    debris.translate(trans_to_B)
    # do lidarScan in {B}
    
    omega_B = Rot_L_to_B[i]@omega_L
    X, Y, Z, V_los = lidarScan2.point_cloud(O_B, h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)  # Generate distances
    # obtain points and LOS velocities in {B}
    XBs.append(X)
    YBs.append(Y)
    ZBs.append(Z)
    PBs.append(np.vstack([X,Y,Z]).T)
    VBs.append(V_los)
    # remember, LOS velocities do not care about rotation speed
    
    # undo the translation since we have debris positions in {L}
    if i+1 < nframes:
        debris.translate(-trans_to_B)
    pass

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
p_0 = np.array(debris_pos[0])  # Guess of initial position of debris - *need to formulate guess*
v_0 = np.array(debris_vel[0])  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = np.array(omega_L)  # Initial guess of angular velocities - *need to formulate guess*
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

# Current states
x_k = x_0.copy()  # State vector
P_k = P_0.copy()  # covariance matrix

# Get Final measurement vectors
z_p_s = [list(p_0)]
z_s = []
x_s = [x_0]
P_s = [P_0]

# convert points and LOS velocities to {L}
for i in range(nframes):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

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
    c_k = (p1_k + p2_k + p3_k + p4_k + p5_k + p6_k + p7_k + p8_k) / 8.

    # Position update
    p_kp1 = v_k * dt + (p_k + c_k) / 2

    # Velocity update
    v_kp1 = v_k.copy()

    # Angular velocity update
    omega_kp1 = omega_k.copy()

    # Vertice updates
    p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, R_k_kp1 = verticeupdate(dt, x_k)

    # Final box
    #drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'g')

    # Compute Jacobian
    F_kp1 = F_matrix(dt, R_k_kp1, p_k, p1_k, p2_k, p3_k, p4_k, p5_k, p6_k, p7_k, p8_k)

    # Update Covariance
    P_kp1 = np.matmul(F_kp1, np.matmul(P_k, F_kp1.T)) + Q

    # Make updated State vector
    x_kp1 = np.array([p_kp1, v_kp1, omega_kp1, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1]).ravel()

    # Calculate the Kalman gain
    K_kp1 = np.matmul(P_kp1, np.linalg.inv(P_kp1 + R))

    #######################
    # Make measurements
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
    z_pi_k, z_p_k = boundingbox.bbox3d(X_i, Y_i, Z_i, ax)

    ax.scatter(debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2], color='g')

    drawrectangle(ax, z_pi_k[:,0], z_pi_k[:,1], z_pi_k[:,2], z_pi_k[:,3], z_pi_k[:,4], z_pi_k[:,5], z_pi_k[:,6], z_pi_k[:,7], 'b')

    # Vertice association
    pi_pk1 = [p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1]
    z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k = nearest_search(pi_pk1, z_pi_k, z_p_k)

    drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'g')
    drawrectangle(ax, z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k, 'r')

    # Estimate linear velocity
    # z_v_k = (np.array(z_p_k) - np.array(z_p_s[i-1]))/dt
    print(VBs[i])
    print(np.mean(VBs[i]))
    z_v_k = debris_vel[i]
    print(z_v_k)

    # find angular velocity from LOS velocities
    c = debris_pos[i]
    v_c = debris_vel[i]
    #z_omega_k = estimate(X_i, Y_i, Z_i, p_k, v_k, VBs[i])
    z_omega_k = estimate(X_i, Y_i, Z_i, c, v_c, VBs[i])

    # Get Measurement Vector
    z_kp1 = np.array(
        [z_p_k, z_v_k, z_omega_k, z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k]).ravel()

    ####################

    # Calculate Residual
    res_kp1 = z_kp1 - x_kp1

    # Update State
    x_kp1 = x_kp1 + np.matmul(K_kp1, res_kp1)

    # Update Covariance
    P_kp1 = np.matmul(np.eye(len(K_kp1)) - K_kp1, P_kp1)

    # Transfer states and covariance from kp1 to k
    P_k = P_kp1.copy()
    x_k = x_kp1.copy()

    ax.scatter(x_k[0], x_k[1], x_k[2], 'r' )

    # Append for analysis
    z_p_s.append(z_p_k)
    P_s.append(P_k)
    x_s.append(x_k)
    z_s.append(z_kp1)




#plt.show()



