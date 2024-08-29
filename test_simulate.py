import matplotlib.pyplot as plt
import numpy as np
import lidarScan3
from mpl_toolkits import mplot3d
import trimesh
import pickle
import os
import pandas as pd
import lidarScan2
from stl import mesh
import dynamics
import random
from mytools import *

def tilde(v):
    if v.ndim ==1:
        v = v[:,np.newaxis]
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

def rodrigues(omega, dt):
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = tilde(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R

def rodrigues_axis_angle(axis, angle):
    axis = axis/np.linalg.norm(axis)  # Unit vector along omega
    phi = angle
    ee_t = np.matmul(axis.reshape(len(axis), 1), axis.reshape(1, len(axis)))
    e_tilde = tilde(axis)
    R = ee_t + (np.eye(len(axis)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R

def visualize_trimesh(mesh, intersections, ax):
    # Plot the mesh
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.2))

    # Plot the intersection points
    # ax.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], c='r', s=20)

    # Set axis limits
    ax.set_xlim(mesh.bounds[:, 0])
    ax.set_ylim(mesh.bounds[:, 1])
    ax.set_zlim(mesh.bounds[:, 2])

    # plt.show()

def process_frame1(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, omeg, res_box, ang_res):
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
    angle_0_rad = np.deg2rad(angle_0)
    angle = np.linalg.norm(omega_L)* dt * i
    debris.apply_transform(trimesh.transformations.rotation_matrix(angle_0_rad + angle, axis))

    debris.apply_transform(trimesh.transformations.translation_matrix(debris_pos_B))
    
    omega_B = Rot_L_to_B @ omega_L
    X, Y, Z, V_los = lidarScan3.point_cloud(np.array([0,0,0]), h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)
    P = np.vstack([X, Y, Z]).T

    # visualize_trimesh(debris, np.column_stack((X,Y,Z)), ax)

    return X, Y, Z, P, V_los, Rot_L_to_B

def process_frame2(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, omeg, res_box, ang_res):
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

    omega_B = Rot_L_to_B.T @ omega_L
    R_0 = rodrigues_axis_angle(omega_B, np.deg2rad(angle_0))
    # Rot_to_B = Rot_L_to_B @ R_0
    debris.rotate_using_matrix(Rot_L_to_B.T)
    debris.rotate(Rot_L_to_B @ (omega_L / np.linalg.norm(omega_L)), -np.linalg.norm(omega_L) * dt * i - np.deg2rad(angle_0))

    debris.translate(debris_pos_B)
    
    omega_B = Rot_L_to_B @ omega_L
    X, Y, Z, V_los = lidarScan2.point_cloud(np.array([0,0,0]), h_resolution, v_resolution, h_range, v_range, debris, debris_pos_B, debris_vel_B, omega_B)
    P = np.vstack([X, Y, Z]).T

    # visualize in B
    # ax.add_collection3d(mplot3d.art3d.Poly3DCollection(debris.vectors, facecolors = ("orange"), alpha=0.2))

    return X, Y, Z, P, V_los, Rot_L_to_B

def get_initial_conditions(conditions_count=40):
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

        px = 0.0818974
        py = -22.23418/1000
        pz = 198.992/1000
        vx = 0.2436358/1000
        vy = -0.57754/1000
        vz = 0.227488/1000
        angle_0 = 169.34
        omx = -0.3365
        omy = 0.1999
        omz = 0.64057
        
        # Altitude (in km)
        altitude = np.random.uniform(670, 35786)
        r = altitude + 6378.  # Earth radius added
        mean_motion = np.sqrt(mu / r**3)
        r0 = np.array([px, py, pz])
        rdot0 = np.array([vx, vy, vz])

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

def get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini):

    Rot_0 = quat2rotm(q_ini)
    # Rot_0 = np.eye(3)
    # print(Rot_0)
    q_s = []
    for i in range(len(debris_pos)):

        # get rotation matrix for that timestep
        Rot_i = rodrigues(omega_true, dt * i)
        q_i = rotm2quat(Rot_i @ Rot_0)
        if i == 0:
            q_s.append(q_i)
        else:
            q_i_alt = -q_i
            q_prev = q_s[i-1]
            if np.linalg.norm(q_i - q_prev) < np.linalg.norm(q_i_alt - q_prev):
                q_s.append(q_i)
            else:
                q_s.append(q_i_alt)
    return q_s


def run_single_simulation(sim_parameters):
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
    q_true = np.array(get_true_orientation(False, omega_L, debris_pos, dt, rotm2quat(rodrigues_axis_angle(omega_L, np.deg2rad(angle_0)))))
    for i in range(nframes):
        visualize_flag = i%50==0
        
        X1, Y1, Z1, P1, V_los1, Rot_L_to_B1 = process_frame1(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, mean_motion, res_box, ang_res)
        # X2, Y2, Z2, P2, V_los2, Rot_L_to_B2 = process_frame2(i, debris_file, debris_pos, debris_vel, angle_0, omega_L, dt, r0, rdot0, mean_motion, res_box, ang_res)
        print(f'frame {i}')
        # Check values here
        tspan = np.arange(0, dt*nframes, dt)
        # Transform points from frame B to frame L
        P1_L = Rot_L_to_B1.T @ P1.T
        X1_L, Y1_L, Z1_L = P1_L[0], P1_L[1], P1_L[2]

        # P2_L = Rot_L_to_B2.T @ P2.T
        # X2_L, Y2_L, Z2_L = P2_L[0], P2_L[1], P2_L[2]

        # Plot the transformed points in frame L
        if visualize_flag:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(X1_L, Y1_L, Z1_L, label='new', marker='.', s=1)
            # ax.scatter(X2_L, Y2_L, Z2_L, label='old', marker='.', s=1)
            ax.legend()
            z_p_k = debris_pos[i]
            R_true = quat2rotm(q_true[i])
            # plot true
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 0]], [z_p_k[1], z_p_k[1] + R_true[1, 0]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 0]],
                    color='green', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 1]], [z_p_k[1], z_p_k[1] + R_true[1, 1]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 1]],
                    color='green', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 2]], [z_p_k[1], z_p_k[1] + R_true[1, 2]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 2]],
                    color='green', linewidth=4, label='True')
            ax.axis('equal')
            plt.show()
        
    # Package the results
    simulation_data = {
        'XBs': XBs, 'YBs': YBs, 'ZBs': ZBs, 'PBs': PBs, 'VBs': VBs,
        'debris_pos': debris_pos, 'debris_vel': debris_vel,
        'Rot_L_to_B': Rot_L_to_Bs, 'omega_L': omega_L, 'dt': dt, 'angle_0': angle_0
    }


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    initial_conditions = get_initial_conditions(1)
    for ic in initial_conditions:
        run_single_simulation(ic)
    print("done")

