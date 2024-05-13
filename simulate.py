import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan2
from stl import mesh
import pickle

from mpl_toolkits import mplot3d
from matplotlib import pyplot

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



# initialize debris position, velocity and orientation
O_B = np.array([0,0,0])
O_L = np.array([0,0,0])
# Dynamics initializations
# r0 = [0, -0.004, 0]  # initial starting position of chaser (km)
# rdot0 = [-0.0001, 0.0, 0.0001]  # initial velocity of debris relative to chaser(km/s)
r0 = [-0.17, -0.35, -0.02]  # initial starting position of chaser (km) - New initial conditions!!
rdot0 = [0.000, 0.00045, 0.0001]  # initial velocity of debris relative to chaser(km/s) - New initial conditions!!
R = 670 + 6378  # Altitude of orbit (km)
mu = 398600.5  # Gravitational constant
omeg = math.sqrt(mu / R ** 3)  # n in the derivations
Rot_0 = np.identity(3) # initial starting rotation matrix/orientation
omega_L = np.array([1, 1, 1]) # inertial, unchanging angular velocity of debris
omega_L_axis = omega_L/np.linalg.norm(omega_L)

# specify time frame and time step
nframes = 2000
dt = 0.05

# simulate debris velocity (linear and angular) in {L} frame from dynamics
x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, omeg)
debris_pos = np.vstack([x,y,z]).T
debris_vel = np.vstack([vx,vy,vz]).T
# specify Lidar resolution and range
# LiDAR point cloud generation initializations
ang_res = 0.025  # angular resolution of Aeries 2 LiDAR
# h_resolution = 40  # Number of rays horizontally
# v_resolution = 40  # Number of rays vertically
res_box = 7
# h_range = 120  # Vertical lidar angle range in degrees
# v_range = 60  # Horizontal lidar angle range in degrees


# generate debris mesh (we only want to do this once to improve compute efficiency)
debris_file = 'kompsat-1-v9.stl'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Simulation Loop
XBs = []
YBs = []
ZBs = []
PBs = []
VBs = []
Rot_L_to_B = []
for i in range(nframes):
    print("Current iteration: " + str(i))

    fov = np.rad2deg(2*np.arctan2(res_box / 2, d[i]))
    h_resolution = min(int(fov / ang_res),60)
    v_resolution = min(int(fov / ang_res),60)
    h_range = fov  # Vertical lidar angle range in degrees
    v_range = fov  # Horizontal lidar angle range in degrees


    print("FOV is (degrees): " + str(fov))
    print("Resolution is (rays): " + str(h_resolution))
    debris = mesh.Mesh.from_file(debris_file)  # Grab satellite mesh
    # solve for rotation matrix  B^R_L (R*L => B)
    Rot_L_to_B.append(getR(x[i],y[i],z[i]))
    # express the position and velocity in B
    debris_pos_B = Rot_L_to_B[i]@debris_pos[i]
    debris_vel_B = Rot_L_to_B[i]@debris_vel[i]

    # what we notices:
        # rotation of mesh does not match the true rotation

    # Confirmed:
        # getR is okay (we saw small rotation, the axes were consistent)
        # numpy stl rotations need to be inverted for both matrix and angle axis
        # evec and not evec.T (for coordinates frames but not bounding boxes)

    # our goal right now:
        # get the mesh/point cloud to always align with green
        # get blue to slightly match green
        # yellow and blue are well associated

    # where we are at:
        # we think the issue is with simulate
        # may be missing a rotation

    # orient debris mesh in {B}
    if i==0: # move the debris to its initial location
        Rot_to_B = Rot_L_to_B[i]@Rot_0
        debris.rotate_using_matrix(Rot_to_B.T) # transpose because numpy stl rotation matrix is done backwards
    else:
        debris.rotate_using_matrix(Rot_L_to_B[i].T)
        debris.rotate(Rot_L_to_B[i]@omega_L_axis,-np.linalg.norm(omega_L*dt*i))

    trans_to_B = debris_pos_B
    debris.translate(trans_to_B)
    # do lidarScan in {B}
    
    omega_B = Rot_L_to_B[i]@omega_L
    X, Y, Z, V_los = lidarScan2.point_cloud(O_B, h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)  # Generate distances
    # obtain points and LOS velocities in {B}
    print("Number of rays hit: " + str(len(X)))
    XBs.append(X)
    YBs.append(Y)
    ZBs.append(Z)
    PBs.append(np.vstack([X,Y,Z]).T)
    VBs.append(V_los)
    # remember, LOS velocities do not care about rotation speed


    # X_L, Y_L, Z_L = Rot_L_to_B[i].T @ np.array([X, Y, Z])
    # ax.scatter(X_L, Y_L, Z_L, marker='o', s=10, label=str(i))
    #
    # debris_pos_i = debris_pos[i]
    #
    # if i >3:
    #
    #     ax.plot([debris_pos_i[0], debris_pos_i[0] + 1], [debris_pos_i[1], debris_pos_i[1] + 0],
    #         [debris_pos_i[2], debris_pos_i[2] + 0],
    #         color='black', linewidth=4)
    #     ax.legend()
    #     plt.show()

    # undo the translation since we have debris positions in {L}
    # if i+1 < nframes:
    #     debris.translate(-trans_to_B)
    # pass

    #plt.show()

data = []
data.append(XBs)
data.append(YBs)
data.append(ZBs)
data.append(PBs)
data.append(VBs)
data.append(debris_pos)
data.append(debris_vel)
data.append(Rot_L_to_B)
data.append(omega_L)
data.append(dt)
with open('sim_kompsat_neg_om_long.pickle', 'wb') as sim_data:
    pickle.dump(data, sim_data)

