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

def add_noise_with_pointing(X, Y, Z, V,
                            range_sigma=0.02,          # m (2 cm)
                            vel_sigma=0.03,            # m/s (3 cm/s)
                            point_sigma_deg=0.002,     # pointing 1-sigma in degrees
                            angular_mode='gaussian'     # or 'uniform' in ±sigma
                           ):
    """
    Adds LiDAR range & velocity noise and pointing uncertainty.
    Pointing uncertainty is modeled as a small 2D angular perturbation
    in the tangent plane orthogonal to the LOS, with known realized angles.
    """
    p = np.vstack([X, Y, Z]).T                            # (N,3)
    d = np.linalg.norm(p, axis=1)                         # (N,)
    # Guard against zero distance
    d = np.where(d == 0, 1e-12, d)
    u = (p.T / d).T                                       # (N,3) unit LOS

    # --- Build an orthonormal basis (e1, e2) in the plane ⟂ to u ---
    # Choose a reference not parallel to u for cross product stability
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(u), 1))
    nearly_parallel = np.abs(u[:, 2]) > 0.9               # if too parallel to z-hat, use x-hat
    ref[nearly_parallel] = np.array([1.0, 0.0, 0.0])

    e1 = np.cross(u, ref)
    e1_norm = np.linalg.norm(e1, axis=1, keepdims=True)
    e1 = e1 / np.maximum(e1_norm, 1e-12)
    e2 = np.cross(u, e1)                                   # already orthonormal if u,e1 are

    # --- Sample small angular errors (alpha, beta) in radians ---
    sig = np.deg2rad(point_sigma_deg)
    if angular_mode == 'gaussian':
        alpha = np.random.normal(0.0, sig, size=len(u))
        beta  = np.random.normal(0.0, sig, size=len(u))
    elif angular_mode == 'uniform':
        alpha = np.random.uniform(-sig, sig, size=len(u))
        beta  = np.random.uniform(-sig, sig, size=len(u))
    else:
        raise ValueError("angular_mode must be 'gaussian' or 'uniform'")

    # --- Apply pointing error: small-angle perturbation in tangent plane ---
    # u_tilted ≈ normalize(u + alpha*e1 + beta*e2)
    u_tilted = u + (alpha[:, None] * e1) + (beta[:, None] * e2)
    u_tilted /= np.linalg.norm(u_tilted, axis=1, keepdims=True)

    # --- Range & velocity noise (your original specs) ---
    distance_noise = np.random.normal(0.0, range_sigma, size=len(d))
    v_noise = np.random.normal(0.0, vel_sigma, size=len(V))

    # Noisy position lies along the *mispointed* LOS with noisy range
    d_noisy = d + distance_noise
    p_noisy = (u_tilted.T * d_noisy).T

    V_n = V + v_noise

    X_n = p_noisy[:, 0]
    Y_n = p_noisy[:, 1]
    Z_n = p_noisy[:, 2]

    # Return also the realized pointing info that is "known" to you
    return X_n, Y_n, Z_n, V_n