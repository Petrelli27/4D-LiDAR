import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan2
from stl import mesh
import pickle
import multiprocessing as mp

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

def process_frame(i, debris_file, debris_pos, debris_vel, omega_L, dt, r0, rdot0, omeg, res_box, ang_res):
    print(f"Processing frame: {i}")
    
    x, y, z = debris_pos[i]
    vx, vy, vz = debris_vel[i]
    d = np.linalg.norm(debris_pos[i])

    fov = np.rad2deg(2*np.arctan2(res_box / 2, d))
    h_resolution = min(int(fov / ang_res), 60)
    v_resolution = min(int(fov / ang_res), 60)
    h_range = fov
    v_range = fov

    debris = mesh.Mesh.from_file(debris_file)
    Rot_L_to_B = getR(x, y, z)
    debris_pos_B = Rot_L_to_B @ debris_pos[i]
    debris_vel_B = Rot_L_to_B @ debris_vel[i]

    if i == 0:
        Rot_to_B = Rot_L_to_B @ np.identity(3)
        debris.rotate_using_matrix(Rot_to_B.T)
    else:
        debris.rotate_using_matrix(Rot_L_to_B.T)
        debris.rotate(Rot_L_to_B @ (omega_L / np.linalg.norm(omega_L)), -np.linalg.norm(omega_L * dt * i))

    debris.translate(debris_pos_B)
    
    omega_B = Rot_L_to_B @ omega_L
    X, Y, Z, V_los = lidarScan2.point_cloud(np.array([0,0,0]), h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)
    P = np.vstack([X, Y, Z]).T

    return X, Y, Z, P, V_los, Rot_L_to_B

import multiprocessing as mp

if __name__ == '__main__':
    # Your existing initialization code here...
    # initialize debris position, velocity and orientation
    O_B = np.array([0,0,0])
    O_L = np.array([0,0,0])
    # Dynamics initializations
    # r0 = [0, -0.004, 0]  # initial starting position of chaser (km)
    # rdot0 = [-0.0001, 0.0, 0.0001]  # initial velocity of debris relative to chaser(km/s)
    r0 = [-0.17, -0.35, 0.03]  # initial starting position of chaser (km) - New initial conditions!!
    rdot0 = [0.000, 0.00045, -0.0002]  # initial velocity of debris relative to chaser(km/s) - New initial conditions!!
    R = 670 + 6378  # Altitude of orbit (km)
    mu = 398600.5  # Gravitational constant
    omeg = math.sqrt(mu / R ** 3)  # n in the derivations
    Rot_0 = np.identity(3) # initial starting rotation matrix/orientation
    omega_L = np.array([-0.5, 0.3, 1]) # inertial, unchanging angular velocity of debris
    omega_L_axis = omega_L/np.linalg.norm(omega_L)

    # specify time frame and time step
    nframes = 400
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

    # Create a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Prepare arguments for each frame
    args = [(i, debris_file, debris_pos, debris_vel, omega_L, dt, r0, rdot0, omeg, res_box, ang_res) for i in range(nframes)]

    # Run the processing in parallel
    results = pool.starmap(process_frame, args)

    # Close the pool
    pool.close()
    pool.join()

    # Unpack the results
    XBs, YBs, ZBs, PBs, VBs, Rot_L_to_B = zip(*results)

    # Convert lists to numpy arrays if needed
    XBs = np.array(XBs)
    YBs = np.array(YBs)
    ZBs = np.array(ZBs)
    PBs = np.array(PBs)
    VBs = np.array(VBs)
    Rot_L_to_B = np.array(Rot_L_to_B)

    # Your existing code to save the data...

    data = []
    data.append(XBs)
    data.append(YBs)
    data.append(ZBs)
    # data.append(PBs)
    data.append(VBs)
    data.append(debris_pos)
    data.append(debris_vel)
    data.append(Rot_L_to_B)
    data.append(omega_L)
    data.append(dt)
    with open('sim_kompsat_parallel_test.pickle', 'wb') as sim_data:
        pickle.dump(data, sim_data)