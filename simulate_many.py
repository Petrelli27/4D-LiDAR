import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
import math
import lidarScan3
import trimesh
import pickle
import multiprocessing as mp
import itertools

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

def process_frame(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, omeg, res_box, ang_res):
    x, y, z = debris_pos[i]
    vx, vy, vz = debris_vel[i]
    d = np.linalg.norm(debris_pos[i])

    fov = np.rad2deg(2*np.arctan2(res_box / 2, d))
    h_resolution = min(int(fov / ang_res), 60)
    v_resolution = min(int(fov / ang_res), 60)
    h_range = fov
    v_range = fov

    debris = trimesh.load(debris_file)
    Rot_L_to_B = getR(x, y, z)
    debris_pos_B = Rot_L_to_B @ debris_pos[i]
    debris_vel_B = Rot_L_to_B @ debris_vel[i]
                
    # trimesh rotation
    Rot_4by4 = np.eye(4)
    Rot_4by4[:3,:3] = Rot_L_to_B
    debris.apply_transform(Rot_4by4)
    axis = Rot_L_to_B @ (omega_L / np.linalg.norm(omega_L))
    angle = np.linalg.norm(omega_L * dt * i)
    debris.apply_transform(trimesh.transformations.rotation_matrix(angle_0 + angle, axis))

    debris.apply_transform(trimesh.transformations.translation_matrix(debris_pos_B))
    
    omega_B = Rot_L_to_B @ omega_L
    X, Y, Z, V_los = lidarScan3.point_cloud(np.array([0,0,0]), h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)
    P = np.vstack([X, Y, Z]).T
    # visualize_trimesh(debris, np.column_stack((X,Y,Z)))
    print(f"Processing frame: {i}")

    return X, Y, Z, P, V_los, Rot_L_to_B

def get_initial_conditions(conditions_count=0):
    px = 0.001*np.array([-350., -150., 10.])
    py = 0.001*np.array([-140., 40.])
    pz = 0.001*np.array([-20., 5.])
    vx = 0.001*np.array([0.1, 0.5])
    vy = 0.001*np.array([-0.2, -0.8])
    vz = 0.001*np.array([0.3])
    altitudes = np.array([670., 35786.])
    r = altitudes + 6378.
    mu = 398600.5  # Gravitational constant
    mean_motions = np.sqrt(mu / r ** 3)  # n in the derivations
    omx = [1, 0.5, 0.1]
    omy = [0.8, 0.3]
    omz = [0.6, -0.2]
    angle_0 = [0, 45, 90]
    
    starts = itertools.product(px, py, pz, vx, vy, vz, angle_0, omx, omy, omz, mean_motions)
    # count = sum(1 for _ in starts) # how many different starts there are
    # print(count)
    if conditions_count == 0:
        return starts
    else:
        limited_starts = itertools.islice(starts, conditions_count)
        return limited_starts


def run_single_simulation(sim_parameters):
    px, py, pz, vx, vy, vz, angle_0, omx, omy, omz, mean_motion = sim_parameters
    r0 = np.array([px, py, pz])
    rdot0 = np.array([vx, vy, vz])
    omega_L = np.array([omx, omy, omz])

    # Your existing initialization code here...
    O_B = np.array([0,0,0])
    O_L = np.array([0,0,0])
    
    nframes = 2000
    dt = 0.05

    # simulate debris velocity (linear and angular) in {L} frame from dynamics
    x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, mean_motion)
    debris_pos = np.vstack([x,y,z]).T
    debris_vel = np.vstack([vx,vy,vz]).T
    if max(d) > 500:
        # too far, avoid simulation
        return

    # LiDAR point cloud generation initializations
    ang_res = 0.025  # angular resolution of Aeries 2 LiDAR
    res_box = 7

    # load debris mesh
    debris_file = 'kompsat-1-v9.stl'

    XBs, YBs, ZBs, PBs, VBs, Rot_L_to_Bs = [], [], [], [], [], []

    for i in range(nframes):
        X, Y, Z, P, V_los, Rot_L_to_B = process_frame(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, mean_motion, res_box, ang_res)
        XBs.append(X)
        YBs.append(Y)
        ZBs.append(Z)
        PBs.append(P)
        VBs.append(V_los)
        Rot_L_to_Bs.append(Rot_L_to_B)

    # Package the results
    simulation_data = {
        'XBs': XBs, 'YBs': YBs, 'ZBs': ZBs, 'PBs': PBs, 'VBs': VBs,
        'debris_pos': debris_pos, 'debris_vel': debris_vel,
        'Rot_L_to_B': Rot_L_to_Bs, 'omega_L': omega_L, 'dt': dt, 'angle_0': angle_0
    }

    return simulation_data

if __name__ == '__main__':
    initial_conditions_list = get_initial_conditions(8)

    # Create a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Run the simulations in parallel
    results = pool.map(run_single_simulation, initial_conditions_list)

    # Close the pool
    pool.close()
    pool.join()

    # Save the results
    for i, simulation_data in enumerate(results):
        with open(f'sim_kompsat_trimesh_test_{i}.pickle', 'wb') as sim_data:
            pickle.dump(simulation_data, sim_data)