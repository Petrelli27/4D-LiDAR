import pickle
import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import lidarScan2

with open('sim2000_noise.pickle', 'rb') as sim_data:
    data = pickle.load(sim_data)
# some utility functions
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
    e = e/np.linalg.norm(e)
    phi = np.arccos(np.dot(z_B, z_L)) # z_B and z_L are already unit vectors
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + tilde(e)*np.sin(phi)
    return R.T

XBs = []
YBs = []
ZBs = []
PBs = []
VBs = []

# Rot_L_to_B = data[7]
# Rot_B_to_L = [np.linalg.inv(r) for r in Rot_L_to_B]
omega_L = data[8]
dt = data[9]
nframes = len(data[0])
Rot_L_to_B = []

# initialize debris position, velocity and orientation
O_B = np.array([0,0,0])
O_L = np.array([0,0,0])
# Dynamics initializations
r0 = [0, -0.004, 0]  # initial starting position of chaser (km)
rdot0 = [-0.0001, 0.0, 0.0001]  # initial velocity of debris relative to chaser(km/s)
R = 500 + 6378  # Altitude of orbit (km)
mu = 398600.5  # Gravitational constant
omeg = math.sqrt(mu / R ** 3)  # n in the derivations
Rot_0 = np.identity(3) # initial starting rotation matrix/orientation
omega_L = np.array([1.,1.,1.]) # inertial, unchanging angular velocity of debris
omega_L_axis = omega_L/np.linalg.norm(omega_L)

# specify Lidar resolution and range
# LiDAR point cloud generation initializations
h_resolution = 40  # Number of rays horizontally
v_resolution = 40  # Number of rays vertically
h_range = 120  # Vertical lidar angle range in degrees
v_range = 60  # Horizontal lidar angle range in degrees

# simulate debris velocity (linear and angular) in {L} frame from dynamics
x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, omeg)
debris_pos = np.vstack([x,y,z]).T
debris_vel = np.vstack([vx,vy,vz]).T
for i in range(nframes):
    print(i)
    
    Rot_L_to_B.append(getR(x[i],y[i],z[i]))
    # express the position and velocity in B
    debris_pos_B = Rot_L_to_B[i]@debris_pos[i]
    debris_vel_B = Rot_L_to_B[i]@debris_vel[i]
    omega_B = Rot_L_to_B[i]@omega_L
    X, Y, Z, V_los = lidarScan2.point_cloud(O_B, h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)  # Generate distances
    # obtain points and LOS velocities in {B}
    XBs.append(X)
    YBs.append(Y)
    ZBs.append(Z)
    PBs.append(np.vstack([X,Y,Z]).T)
    VBs.append(V_los)

data2 = []
data2.append(XBs)
data2.append(YBs)
data2.append(ZBs)
data2.append(PBs)
data2.append(VBs)
data2.append(debris_pos)
data2.append(debris_vel)
data2.append(Rot_L_to_B)
data2.append(omega_L)
data2.append(dt)
with open('sim2000_fixed.pickle', 'wb') as sim_data:
    pickle.dump(data2, sim_data)