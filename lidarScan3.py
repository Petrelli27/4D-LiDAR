import matplotlib.pyplot as plt
import numpy as np
import math
import lidarNoise
from mpl_toolkits import mplot3d
from matplotlib import pyplot

import numpy as np
import trimesh
import lidarNoise  # Assuming you still want to use your custom noise function

def point_cloud(O_B, horizontal_resolution, vertical_resolution, h_range, v_range, sat_mesh, sat_pos, v_rel, omega):
    # Generate rays
    theta_r = np.deg2rad(np.linspace(-h_range/2, h_range/2, horizontal_resolution))
    phi_r = np.deg2rad(np.linspace(-v_range/2, v_range/2, vertical_resolution))
    
    Theta, Phi = np.meshgrid(theta_r, phi_r)
    
    rays = np.stack([
        np.cos(Phi) * np.sin(Theta),
        np.sin(Phi),
        np.cos(Theta) * np.cos(Phi)
    ], axis=-1).reshape(-1, 3)

    # Perform ray casting
    locations, index_ray, index_tri = sat_mesh.ray.intersects_location(
        ray_origins=np.tile(O_B, (len(rays), 1)),
        ray_directions=rays, multiple_hits=False
    )

    # If no intersections, return empty arrays
    if len(locations) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Calculate relative positions
    useful_rel_locations = locations - O_B

    # Calculate velocities
    r = useful_rel_locations - sat_pos
    u_los = -useful_rel_locations / np.linalg.norm(useful_rel_locations, axis=1)[:, np.newaxis]
    v_los_s = np.sum(np.cross(omega, r) * u_los, axis=1) + np.dot(v_rel, u_los.T)
    v_los_v = u_los * v_los_s[:, np.newaxis]

    # Add noise to lidar scan results
    Xs, Ys, Zs = useful_rel_locations.T
    Xn, Yn, Zn, Vn = lidarNoise.add_noise(Xs, Ys, Zs, v_los_s)

    return Xn, Yn, Zn, Vn

