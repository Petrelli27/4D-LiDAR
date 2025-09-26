import numpy as np

def add_noise(X, Y, Z, V):
    # for Aeries II, the specs are:
    # minimum range precision = 2cm
    # minimum velocity precision = 3cm/s
    # this is based on the perspective of the LiDAR, therefore use {B} frame
    p = np.vstack([X,Y,Z]).T # position vector
    distances = np.linalg.norm(p,axis=1)
    distances = distances[:,np.newaxis]
    distance_noise = np.random.normal(0,0.02,len(distances))
    distance_noise = distance_noise[:,np.newaxis] # change to column vector
    velocity_noise = np.random.normal(0,0.03,len(V))
    u_los = p/distances # this is not negative just to make things simpler
    p_noisy = u_los*(distances + distance_noise)
    V_n = V + velocity_noise

    X_n = p_noisy[:,0]
    Y_n = p_noisy[:,1]
    Z_n = p_noisy[:,2]
    return X_n, Y_n, Z_n, V_n


def add_noise_with_global_pointing(
    X, Y, Z, V,
    range_sigma=0.02,      # m
    vel_sigma=0.03,        # m/s
    point_sigma_deg=0.002, # 1-sigma pointing (deg)
    angular_mode='gaussian',  # 'gaussian' or 'uniform'
    omega_xyz_deg=None       # optional (rx, ry, rz) in deg; if given, use exactly
):
    """
    Same angular mispoint for all rays. Pointing is modeled as a small rotation
    of the LOS in the sensor frame, using u_tilted ≈ normalize(u + ω×u).
    If omega_xyz_deg is provided (tuple of 3 angles in deg), that exact rotation is used.
    """
    p = np.vstack([X, Y, Z]).T
    d = np.linalg.norm(p, axis=1)
    d = np.where(d == 0, 1e-12, d)
    u = (p.T / d).T  # unit LOS per point

    # --- choose one global small rotation vector omega (radians) ---
    if omega_xyz_deg is not None:
        rx, ry, rz = np.deg2rad(omega_xyz_deg)
    else:
        sig = np.deg2rad(point_sigma_deg)
        if angular_mode == 'gaussian':
            rx, ry, rz = np.random.normal(0.0, sig, size=3)
        elif angular_mode == 'uniform':
            rx = np.random.uniform(-sig, sig)
            ry = np.random.uniform(-sig, sig)
            rz = np.random.uniform(-sig, sig)
        else:
            raise ValueError("angular_mode must be 'gaussian' or 'uniform'")
    omega = np.array([rx, ry, rz])  # same for all points

    # --- apply small rotation: u_tilted = normalize(u + omega x u) ---
    omega_tile = np.tile(omega, (len(u), 1))
    u_tilted = u + np.cross(omega_tile, u)
    u_tilted /= np.linalg.norm(u_tilted, axis=1, keepdims=True)

    # --- add range & velocity noise as before ---
    d_noisy = d + np.random.normal(0.0, range_sigma, size=len(d))
    v_noise = np.random.normal(0.0, vel_sigma, size=len(V))

    p_noisy = (u_tilted.T * d_noisy).T
    V_n = V + v_noise

    X_n = p_noisy[:, 0]
    Y_n = p_noisy[:, 1]
    Z_n = p_noisy[:, 2]

    # Return also the actual rotation used (deg) so you "know" your pointing
    return X_n, Y_n, Z_n, V_n
