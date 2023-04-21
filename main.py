#import lidarScan
import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan
import boundingbox
from estimateOmega import estimate


from mpl_toolkits import mplot3d
from matplotlib import pyplot


# Dynamics initializations
r0 = [0, 0, 0]  # initial starting position of chaser (km)
rdot0 = [-0.0001, 0.0, 0.0001]  # initial velocity of debris relative to chaser(km/s)
R = 500 + 6378  # Altitude of orbit (km)
mu = 398600.5  # Gravitational constant
omeg = math.sqrt(mu / R ** 3)  # n in the derivations

# Target info
sat_mesh_file = 'kompsat-1-v9.stl'  # Grab the satellite mesh
omega = [0.768, 0.5, 0.1]  # Absolute angular velocity of satellite (tumbling rate rad/s)

# LiDAR point cloud generation initializations
horizontal_resolution = 40  # Number of rays horizontally
vertical_resolution = 10  # Number of rays vertically
h_range = 20  # Vertical lidar angle range in degrees
v_range = 10  # Horizontal lidar angle range in degrees

# Time relevant initializations
nframes = 4  # number of samples
dt = 0.1  # time interval (i.e. total time is dt*nframes)

# Generate positions, velocities, distances and speeds over proposed trajectory
#x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, omeg)

# Generate positions - Constant view point
# x = -40*np.ones(nframes)
x = np.zeros(nframes)
y = np.zeros(nframes)
z = np.zeros(nframes)
vx = np.zeros(nframes)
vy = np.zeros(nframes)
vz = np.zeros(nframes)
sat_pos = np.zeros([nframes, 3])
sat_pos[0] = np.array([0, 0, 10])
v_rel = np.vstack([vx, vy, vz]).T
# Generate LiDAR point cloud from positions of chaser
Xs = []
Ys = []
Zs = []
Vs = []
for i in range(nframes):
    t = dt*i
    O_B = np.array([x[i], y[i], z[i]])  # Centroid of LiDAR (chaser) (milimeters!)
    X, Y, Z, V_los = lidarScan.point_cloud(O_B, horizontal_resolution, vertical_resolution, h_range, v_range, sat_mesh_file, sat_pos[i], v_rel[i], omega, t)  # Generate distances
    # Generate velocities
    # print(X)
    # print(Y)
    # print(Z)
    plt.show()
    Xs.append(X)
    Ys.append(Y)
    Zs.append(Z)
    Vs.append(V_los)
    if i+1 < nframes:
        sat_pos[i+1] = sat_pos[i] + v_rel[i]*t

# Algorithm
for i in range(nframes):
    t = dt*i

    # Position and velocity data for this time
    pos_b_i = [x[i], y[i], z[i]]
    vel_b_i = [vx[i], vy[i], vz[i]]
    # print(pos_b_i)
    # print(vel_b_i)

    # Point Cloud measurement for this time
    X_i = Xs[i]
    Y_i = Ys[i]
    Z_i = Zs[i]

    # Estimated 3d bounding box for this point cloud
    boundingbox.bbox3d(X_i, Y_i, Z_i)
    
    # assume we get some center position and velocity estimate
    # calcuate angular velocity based on linear least squares regression
    c = sat_pos[i]
    v_c = v_rel[i]
    est_omega = estimate(X_i, Y_i, Z_i, c, v_c, Vs[i])
    print(f"omega = {est_omega}")
    pass

plt.show()

# print(Xs)

