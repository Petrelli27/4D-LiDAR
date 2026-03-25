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


def add_noise_with_global_pointing_and_range_dependent_error(
    X, Y, Z, V,
    range_sigma=0.02,         # baseline range sigma [m]
    vel_sigma=0.03,           # baseline velocity sigma [m/s]
    point_sigma_deg=0.002,    # 1-sigma pointing (deg)
    angular_mode='gaussian',  # 'gaussian' or 'uniform'
    omega_xyz_deg=None        # optional (rx, ry, rz) in deg; if given, use exactly
):
    """
    Same angular mispoint for all rays. Pointing is modeled as a small rotation
    of the LOS in the sensor frame, using u_tilted ≈ normalize(u + ω×u).
    If omega_xyz_deg is provided (tuple of 3 angles in deg), that exact rotation is used.

    Range and velocity noise are both made pointwise range-dependent with a baseline:
        sigma(r) = baseline_sigma,                         for r <= 10 m
                 = baseline_sigma + k * (r - 10),         for r > 10 m

    where:
        k = 3e-2 * 6.666666e-2 ≈ 0.002
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    V = np.asarray(V, dtype=float)

    p = np.column_stack((X, Y, Z))              # (N,3)
    d = np.linalg.norm(p, axis=1)               # (N,)
    d_safe = np.where(d == 0.0, 1e-12, d)
    u = p / d_safe[:, None]                     # unit LOS per point

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

    omega = np.array([rx, ry, rz], dtype=float)

    # --- apply same small rotation to all LOS rays ---
    u_tilted = u + np.cross(omega[None, :], u)
    u_tilted /= np.linalg.norm(u_tilted, axis=1, keepdims=True)

    # --- piecewise per-point sigmas ---
    k = 3e-2 * 6.666666e-2   # ≈ 0.002
    excess_range = np.maximum(d - 10.0, 0.0)

    sigma_range_i = range_sigma + k * excess_range
    sigma_vel_i   = vel_sigma   + k * excess_range

    # --- draw independent per-point noise ---
    d_noisy = d + np.random.normal(0.0, sigma_range_i, size=d.shape)
    v_noise = np.random.normal(0.0, sigma_vel_i, size=V.shape)

    # Optional: clamp range to nonnegative if desired
    d_noisy = np.maximum(d_noisy, 0.0)

    # --- reconstruct noisy points from tilted LOS + noisy range ---
    p_noisy = u_tilted * d_noisy[:, None]
    V_n = V + v_noise

    X_n = p_noisy[:, 0]
    Y_n = p_noisy[:, 1]
    Z_n = p_noisy[:, 2]

    return X_n, Y_n, Z_n, V_n


def add_noise_with_global_pointing_and_range_dependent_error_and_dropout(
    X, Y, Z, V,
    range_sigma=0.02,         # baseline range sigma [m]
    vel_sigma=0.03,           # baseline velocity sigma [m/s]
    point_sigma_deg=0.002,    # 1-sigma pointing (deg)
    angular_mode='gaussian',  # 'gaussian' or 'uniform'
    omega_xyz_deg=None,       # optional (rx, ry, rz) in deg; if given, use exactly
    dropout_alpha=None,       # Beta(alpha, beta) for frame-level dropout
    dropout_beta=None,
    return_dropout=False,
    do_dropout=False
):
    """
    Same angular mispoint for all rays. Pointing is modeled as a small rotation
    of the LOS in the sensor frame, using u_tilted ≈ normalize(u + ω×u).
    If omega_xyz_deg is provided (tuple of 3 angles in deg), that exact rotation is used.

    Range and velocity noise are both made pointwise range-dependent with a baseline:
        sigma(r) = baseline_sigma,                         for r <= 10 m
                 = baseline_sigma + k * (r - 10),         for r > 10 m

    where:
        k = 3e-2 * 6.666666e-2 ≈ 0.002

    Frame-level dropout:
        p_drop_frame ~ Beta(dropout_alpha, dropout_beta)

    Then each point is independently dropped with probability p_drop_frame.

    If dropout_alpha or dropout_beta is None, no dropout is applied.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    V = np.asarray(V, dtype=float)

    if not (X.shape == Y.shape == Z.shape == V.shape):
        raise ValueError("X, Y, Z, and V must have the same shape")

    p = np.column_stack((X, Y, Z))
    d = np.linalg.norm(p, axis=1)
    d_safe = np.where(d == 0.0, 1e-12, d)
    u = p / d_safe[:, None]

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

    omega = np.array([rx, ry, rz], dtype=float)

    # --- apply same small rotation to all LOS rays ---
    u_tilted = u + np.cross(omega[None, :], u)
    u_tilted /= np.linalg.norm(u_tilted, axis=1, keepdims=True)

    # --- piecewise per-point sigmas ---
    k = 3e-2 * 6.666666e-2   # ≈ 0.002
    excess_range = np.maximum(d - 10.0, 0.0)

    sigma_range_i = range_sigma + k * excess_range
    sigma_vel_i   = vel_sigma   + k * excess_range

    # --- draw independent per-point noise ---
    d_noisy = d + np.random.normal(0.0, sigma_range_i, size=d.shape)
    v_noise = np.random.normal(0.0, sigma_vel_i, size=V.shape)

    d_noisy = np.maximum(d_noisy, 0.0)

    # --- reconstruct noisy points from tilted LOS + noisy range ---
    p_noisy = u_tilted * d_noisy[:, None]
    V_n = V + v_noise


    X_n = p_noisy[:, 0]
    Y_n = p_noisy[:, 1]
    Z_n = p_noisy[:, 2]

    if do_dropout == False:
        return X_n, Y_n, Z_n, V_n

    # --- frame-level Beta dropout ---
    p_drop_frame = 0.0
    if dropout_alpha is not None or dropout_beta is not None:
        if dropout_alpha is None or dropout_beta is None:
            raise ValueError("Provide both dropout_alpha and dropout_beta, or neither")
        if dropout_alpha <= 0 or dropout_beta <= 0:
            raise ValueError("dropout_alpha and dropout_beta must be > 0")

        p_drop_frame = np.random.beta(dropout_alpha, dropout_beta)

        keep_mask = np.random.random(size=X_n.shape[0]) > p_drop_frame
        X_n = X_n[keep_mask]
        Y_n = Y_n[keep_mask]
        Z_n = Z_n[keep_mask]
        V_n = V_n[keep_mask]

    if return_dropout:
        return X_n, Y_n, Z_n, V_n, p_drop_frame

    return X_n, Y_n, Z_n, V_n