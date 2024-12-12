import dynamics
import matplotlib.pyplot as plt
import numpy as np
import lidarScan3
import trimesh
import pickle
import os
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI
import pandas as pd

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

def process_frame(rank, i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, omeg, res_box, ang_res):
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
    x_prev, y_prev, z_prev = debris_pos[i-1] if i>0 else debris_pos[i]
    Rot_L_to_B_prev = getR(x_prev, y_prev, z_prev)
    debris_pos_B = Rot_L_to_B @ debris_pos[i]
    debris_vel_B = Rot_L_to_B @ debris_vel[i]
                
    # trimesh rotation
    Rot_4by4 = np.eye(4)
    Rot_4by4[:3,:3] = Rot_L_to_B
    debris.apply_transform(Rot_4by4)
    axis = Rot_L_to_B @ (omega_L / np.linalg.norm(omega_L))
    angle_0_rad = np.deg2rad(angle_0)
    angle = np.linalg.norm(omega_L * dt * i)
    debris.apply_transform(trimesh.transformations.rotation_matrix(angle_0_rad + angle, axis))

    debris.apply_transform(trimesh.transformations.translation_matrix(debris_pos_B))
    
    omega_B = Rot_L_to_B @ omega_L
    X, Y, Z, V_los = lidarScan3.point_cloud(np.array([0,0,0]), h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B, Rot_L_to_B, Rot_L_to_B_prev, dt)
    P = np.vstack([X, Y, Z]).T
    # visualize_trimesh(debris, np.column_stack((X,Y,Z)))
    print(f"Process {rank} processing frame: {i}")

    return X, Y, Z, P, V_los, Rot_L_to_B

def get_initial_conditions(conditions_count=100):
    starts_dict = []
    mu = 398600.5
    i = 0
    dt = 0.05
    while i<conditions_count:
        # Position (in km)
        px = np.random.uniform(-0.35, 0.35)
        py = np.random.uniform(-0.35, 0.35)
        pz = np.random.uniform(-0.35, 0.35)
        
        # Velocity (in km/s)
        vx = np.random.uniform(-0.001, 0.001)
        vy = np.random.uniform(-0.001, 0.001)
        vz = np.random.uniform(-0.001, 0.001)
        
        # Initial angle (in degrees)
        angle_0 = np.random.uniform(0, 360)
        
        # Angular velocity (in rad/s)
        omx = np.random.uniform(-1.0, 1.0)
        omy = np.random.uniform(-1.0, 1.0)
        omz = np.random.uniform(-1.0, 1.0)
        
        # Altitude (in km)
        altitude = np.random.uniform(670, 35786)
        r = altitude + 6378.  # Earth radius added
        mean_motion = np.sqrt(mu / r**3)
        r0 = np.array([px, py, pz])
        rdot0 = np.array([vx, vy, vz])
    
        if i==0:
            nframes = 4000
        elif i==1:
            nframes = 4000
        else:
            nframes = 4000

        _, _, _, _, _, _, d, _ = dynamics.propagate(dt, nframes, r0, rdot0, mean_motion)
        if max(d) > 500:
            # too far, avoid appending this result
            continue

        starts_dict.append({
            'px': px, 'py': py, 'pz': pz,
            'vx': vx, 'vy': vy, 'vz': vz,
            'angle_0': angle_0,
            'omx': omx, 'omy': omy, 'omz': omz,
            'mean_motion': mean_motion,
            'nframes': nframes  # Randomly choose one of these values
        })
        i += 1
    return starts_dict

def run_single_simulation(rank, sim_parameters, sim_index):
    r0 = np.array([sim_parameters['px'], sim_parameters['py'], sim_parameters['pz']])
    rdot0 = np.array([sim_parameters['vx'], sim_parameters['vy'], sim_parameters['vz']])
    omega_L = np.array([sim_parameters['omx'], sim_parameters['omy'], sim_parameters['omz']])
    angle_0 = sim_parameters['angle_0']
    mean_motion = sim_parameters['mean_motion']
    nframes = sim_parameters['nframes']

    # Your existing initialization code here...
    O_B = np.array([0,0,0])
    O_L = np.array([0,0,0])
    
    dt = 0.05

    # simulate debris velocity (linear and angular) in {L} frame from dynamics
    x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, mean_motion)
    debris_pos = np.vstack([x,y,z]).T
    debris_vel = np.vstack([vx,vy,vz]).T

    # LiDAR point cloud generation initializations
    ang_res = 0.025  # angular resolution of Aeries 2 LiDAR
    res_box = 7

    # load debris mesh
    debris_file = 'kompsat-1-v9.stl'

    XBs, YBs, ZBs, PBs, VBs, Rot_L_to_Bs = [], [], [], [], [], []

    for i in range(nframes):
        X, Y, Z, P, V_los, Rot_L_to_B = process_frame(rank, i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, mean_motion, res_box, ang_res)
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

    # Save the simulation data immediately
    os.makedirs('results', exist_ok=True)
    with open(f'results/sim_kompsat_trimesh_test_{sim_index}.pickle', 'wb') as sim_data:
        pickle.dump(simulation_data, sim_data)

    print(f"Process {rank} completed and saved simulation {sim_index}")

    return sim_index  # Return just the index instead of the full data
 

if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_tests = 4
    try:
        initial_conditions_list = list(get_initial_conditions(num_tests))
        total_conditions = len(initial_conditions_list)

        # Distribute work among processes
        local_results = []
        for i in range(rank, total_conditions, size):
            conditions = initial_conditions_list[i]
            sim_index = run_single_simulation(rank, conditions, i)
            if sim_index is not None:
                local_results.append(sim_index)

        # Gather all results to process 0
        all_completed = comm.gather(local_results, root=0)

        # Process 0 saves all results
        if rank == 0:
            ini_cond_df = pd.DataFrame(initial_conditions_list)
            flat_completed = [item for sublist in all_completed for item in sublist]
            ini_cond_df['file index'] = flat_completed
            ini_cond_df.to_csv('initial_conditions.csv', sep=',', header=True, index=False)
            print(f"Total completed simulations: {len(flat_completed)}")
            print("Completed simulation indices:", flat_completed)

    except Exception as e:
        print(f"Error on process {rank}: {str(e)}")
        comm.Abort(1)  # Abort all MPI processes if an error occurs
