import matplotlib.pyplot as plt
import numpy as np
import math
import lidarNoise
from mpl_toolkits import mplot3d
from matplotlib import pyplot

import numpy as np
import trimesh
import lidarNoise  # Assuming you still want to use your custom noise function

def point_cloud(O_B, horizontal_resolution, vertical_resolution, h_range, v_range, sat_mesh, sat_pos, v_rel, omega, Rot_L_to_B, Rot_L_to_B_prev, dt):
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

    # calculate relative rotation of L w.r.t. B
    Rlb = Rot_L_to_B_prev.T @ Rot_L_to_B  # shorthand
    angle_B_to_B = 2 * np.arctan2(np.linalg.norm(Rlb - Rlb.T)/2, 1)
    if angle_B_to_B < 1e-3:
        omega_L_to_B = np.array([0,0,0])
    else:
        axis_B_to_B = 1./(2*np.sin(angle_B_to_B))*np.array([Rlb[2,1]-Rlb[1,2],Rlb[0,2]-Rlb[2,0],Rlb[1,0]-Rlb[0,1]])
        axis_B_to_B = np.transpose(Rot_L_to_B) @ axis_B_to_B / np.linalg.norm(axis_B_to_B)
        omega_L_to_B = (angle_B_to_B * axis_B_to_B)/dt

    v_rel_B = v_rel + np.cross(-omega_L_to_B, Rot_L_to_B @ sat_pos)
    v_los_s = np.sum(np.cross(omega, r) * u_los, axis=1) + np.dot(v_rel_B, u_los.T)
    v_los_v = u_los * v_los_s[:, np.newaxis]

    # Add noise to lidar scan results
    Xs, Ys, Zs = useful_rel_locations.T
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

    # --- Generate both noisy datasets (you already have these two lines) ---
    Xno, Yno, Zno, Vno = lidarNoise.add_noise(Xs, Ys, Zs, v_los_s)
    Xn, Yn, Zn, Vn = lidarNoise.add_noise_with_global_pointing(Xs, Ys, Zs, v_los_s)

    # --- Helper: set equal scale in 3D so spheres look like spheres ---
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]
        max_range = max([x_range, y_range, z_range]) / 2.0

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
        ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
        ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

    # --- Figure 1: 3D positions ---
    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Compute combined limits for fair comparison
    X_all = np.concatenate([Xno, Xn])
    Y_all = np.concatenate([Yno, Yn])
    Z_all = np.concatenate([Zno, Zn])

    pad = 0.02 * max(
        np.ptp(X_all) if np.ptp(X_all) > 0 else 1.0,
        np.ptp(Y_all) if np.ptp(Y_all) > 0 else 1.0,
        np.ptp(Z_all) if np.ptp(Z_all) > 0 else 1.0,
    )

    ax1.scatter(Xno, Yno, Zno, s=6, alpha=0.6, label='Range-only noise')
    ax1.scatter(Xn, Yn, Zn, s=6, alpha=0.6, label='Range + pointing')

    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('3D position point clouds: range vs. range+pointing')

    # Set symmetric-ish limits with a little padding
    ax1.set_xlim(np.min(X_all) - pad, np.max(X_all) + pad)
    ax1.set_ylim(np.min(Y_all) - pad, np.max(Y_all) + pad)
    ax1.set_zlim(np.min(Z_all) - pad, np.max(Z_all) + pad)
    set_axes_equal(ax1)
    ax1.legend(loc='upper right')

    # --- Figure 2: velocities (overlayed histograms) ---
    fig2 = plt.figure(figsize=(7, 5))
    ax2 = fig2.add_subplot(111)

    bins = 50
    ax2.hist(Vno, bins=bins, density=True, alpha=0.5, label='Range-only noise')
    ax2.hist(Vn, bins=bins, density=True, alpha=0.5, label='Range + pointing')

    ax2.set_xlabel('LOS velocity [m/s]')
    ax2.set_ylabel('Density')
    ax2.set_title('Velocity distributions: range vs. range+pointing')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # --- Optional: quick quantitative summaries ---
    print("Δpos RMS (meters):",
          np.sqrt(np.mean((Xn - Xno) ** 2 + (Yn - Yno) ** 2 + (Zn - Zno) ** 2)))
    print("Δvel mean abs diff (m/s):", np.mean(np.abs(Vn - Vno)))

    return Xn, Yn, Zn, Vn

