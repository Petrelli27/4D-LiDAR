# !/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def CW2(r0, rdot0, omeg, t):
    """
    Wiltshire equations
    :param r0: the position of the chaser relative to the target,
                x is radial to earth, y is along movement of target and z is along angular momemtum
    :param rdot0: the relative velocity of the chaser
    :param omeg: is n is the wiltshire equations
    :param t: time
    :return: returns the position and velocity at the specified time
    """

    x0 = r0[0]
    y0 = r0[1]
    z0 = r0[2]
    xdot0 = rdot0[0]
    ydot0 = rdot0[1]
    zdot0 = rdot0[2]

    xt = (4 * x0 + (2 * ydot0) / omeg) + (xdot0 / omeg) * math.sin(omeg * t) - (3 * x0 + (2 * ydot0) / omeg) * math.cos(
        omeg * t)
    yt = (y0 - (2 * xdot0) / omeg) + ((2 * xdot0) / omeg) * math.cos(omeg * t) + (
                6 * x0 + (4 * ydot0) / omeg) * math.sin(omeg * t) - (6 * omeg * x0 + 3 * ydot0) * t
    zt = z0 * math.cos(omeg * t) + (zdot0 / omeg) * math.sin(omeg * t)

    xdott = (3 * omeg * x0 + 2 * ydot0) * math.sin(omeg * t) + xdot0 * math.cos(omeg * t)
    ydott = (6 * omeg * x0 + 4 * ydot0) * math.cos(omeg * t) - 2 * xdot0 * math.sin(omeg * t) - (
                6 * omeg * x0 + 3 * ydot0)
    zdott = zdot0 * math.cos(omeg * t) - z0 * omeg * math.sin(omeg * t)

    return [xt, yt, zt], [xdott, ydott, zdott]

def two_body_ode(t, state, mu):
    """
    Equations of motion for a single spacecraft under two-body gravity.
    state = [x, y, z, vx, vy, vz]  (ECI, km and km/s)
    """
    r = state[:3]
    v = state[3:]
    r_norm3 = np.dot(r, r) ** 1.5
    accel = -mu / r_norm3 * r
    return np.concatenate([v, accel])
 
 
def lvlh_axes(r_c, v_c):
    """
    Build the LVLH frame unit vectors from chaser ECI state.
      x_hat : radial (outward)
      y_hat : along-track (in direction of velocity for circular orbit)
      z_hat : angular momentum direction
    Returns (x_hat, y_hat, z_hat) as 3-element arrays.
    """
    x_hat = r_c / np.linalg.norm(r_c)
    h = np.cross(r_c, v_c)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat
 
 
def eci_to_lvlh(r_c, v_c, r_d_eci, v_d_eci):
    """
    Convert debris ECI position/velocity to LVLH relative state.
    Returns relative position and velocity in LVLH frame (km, km/s).
    """
    x_hat, y_hat, z_hat = lvlh_axes(r_c, v_c)
    R_lvlh = np.array([x_hat, y_hat, z_hat])   # 3×3 rotation matrix
 
    # Relative ECI vectors
    dr_eci = r_d_eci - r_c
    dv_eci = v_d_eci - v_c
 
    # Angular velocity of LVLH frame: omega_vec = h / |r|^2 (along z_hat)
    h_vec = np.cross(r_c, v_c)
    omega_vec = h_vec / np.dot(r_c, r_c)       # rad/s, in ECI
 
    # Relative position in LVLH
    dr_lvlh = R_lvlh @ dr_eci
 
    # Relative velocity in LVLH (subtract frame rotation: v_rel = R*(dv_eci - omega×dr_eci))
    dv_lvlh = R_lvlh @ (dv_eci - np.cross(omega_vec, dr_eci))
 
    return dr_lvlh, dv_lvlh
 
 
def lvlh_to_eci_state(R_km, omeg, mu, dr_lvlh, dv_lvlh):
    """
    Given a circular chaser orbit radius and an initial LVLH relative state (of debris),
    return the debris' ECI state vector.
    The chaser is placed at [R, 0, 0] with velocity [0, sqrt(mu/R), 0] in ECI
    (a canonical in-plane reference orientation).
    """
    # Chaser ECI state (canonical orientation)
    r_c = np.array([R_km, 0.0, 0.0])
    v_circ = math.sqrt(mu / R_km)
    v_c = np.array([0.0, v_circ, 0.0])
 
    x_hat, y_hat, z_hat = lvlh_axes(r_c, v_c)
 
    # Debris ECI position
    r_d = r_c + x_hat * dr_lvlh[0] + y_hat * dr_lvlh[1] + z_hat * dr_lvlh[2]
 
    # Angular velocity of LVLH frame
    h_vec = np.cross(r_c, v_c)
    omega_vec = h_vec / np.dot(r_c, r_c)
 
    # Debris ECI velocity (invert the frame-rotation correction)
    R_lvlh = np.array([x_hat, y_hat, z_hat])
    dv_eci = R_lvlh.T @ dv_lvlh + np.cross(omega_vec, r_d - r_c)
    v_d = v_c + dv_eci
 
    return r_c, v_c, r_d, v_d

def propagate(dt, nframes, r0, rdot0, omeg, R=None, mu=398600.5): # uses nonlinear dynamics
    """
    Propagate relative orbital motion using the full nonlinear two-body equations
    integrated with SciPy's RK45 solver.
 
    dt:      time step in seconds
    nframes: number of output frames (total time = dt * nframes)
    r0:      initial relative position of chaser in LVLH frame (km)
    rdot0:   initial relative velocity of chaser in LVLH frame (km/s)
    omeg:    mean motion n = sqrt(mu/R^3) (rad/s) — used to derive R if
                    R is not supplied explicitly
    R:       target orbit radius (km); derived from omeg if None
    mu:      gravitational parameter (km^3/s^2), default Earth
    return:        xs, ys, zs  relative position   (m)
                    vxs, vys, vzs relative velocity   (m/s)
                    ds            range to debris     (m)
                    vs            relative speed       (m/s)
    """
    # Derive orbit radius from mean motion if not provided
    if R is None:
        R = (mu / omeg ** 2) ** (1.0 / 3.0)
 
    r0 = np.asarray(r0, dtype=float)
    rdot0 = np.asarray(rdot0, dtype=float)
 
    # Convert initial LVLH relative state → ECI absolute states
    r_c0_eci, v_c0_eci, r_d0_eci, v_d0_eci = lvlh_to_eci_state(R, omeg, mu, r0, rdot0)
    state0 = np.concatenate([r_c0_eci, v_c0_eci, r_d0_eci, v_d0_eci])
 
    def ode(t, state):
        ds_chaser = two_body_ode(t, state[0:6], mu)
        ds_debris = two_body_ode(t, state[6:12], mu)
        return np.concatenate([ds_chaser, ds_debris])
 
    # Time grid for output
    t_span = (0.0, dt * (nframes - 1))
    t_eval = np.linspace(0.0, dt * (nframes - 1), nframes)
 
    sol = solve_ivp(
        ode,
        t_span,
        state0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        dense_output=False,
    )
 
    if not sol.success:
        raise RuntimeError(f"RK45 integration failed: {sol.message}")
 
    # Unpack and convert to LVLH at each timestep
    m2km = 1000.0   # factor to convert km → m  (and km/s → m/s)
 
    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []
    ds, vs = [], []
 
    for i in range(nframes):
        r_chaser = sol.y[0:3, i]
        v_chaser = sol.y[3:6, i]
        r_debris = sol.y[6:9, i]
        v_debris = sol.y[9:12, i]
 
        dr_lvlh, dv_lvlh = eci_to_lvlh(r_chaser, v_chaser, r_debris, v_debris)
 
        d = np.linalg.norm(dr_lvlh)      # km
        v = np.linalg.norm(dv_lvlh)      # km/s
 
        xs.append(dr_lvlh[0] * m2km)
        ys.append(dr_lvlh[1] * m2km)
        zs.append(dr_lvlh[2] * m2km)
        vxs.append(dv_lvlh[0] * m2km)
        vys.append(dv_lvlh[1] * m2km)
        vzs.append(dv_lvlh[2] * m2km)
        ds.append(d * m2km)
        vs.append(v * m2km)
 
    return xs, ys, zs, vxs, vys, vzs, ds, vs

# def propagate(dt, nframes, r0, rdot0, omeg): # uses CW dynamics
#     """
#     Propagate the initial conditions over time given time interval and number of frames desired
#     :param dt: time interval in seconds
#     :param nframes: total number of frames to be run (total time = dt*nframes)
#     :param the rest: are initial conditions for the CW2 function
#     :return: lists of x,y,z,vx,vy,vz,v,d position (x,y,z), velocities (vx,vy,vz), speed(v), distances(d)
#             in meters relative to the target (at the origin)
#     """

#     # Final positions and velocities
#     xs = []
#     ys = []
#     zs = []
#     vxs = []
#     vys = []
#     vzs = []
#     ds = []
#     vs = []

#     # Propagate over time
#     for i in range(nframes):

#         # Current time
#         t = dt * i

#         # Propogate dynamics - rvec is position of chaser rel target, rdot_vec is velocity
#         r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)

#         # Distance to target (km)
#         d = math.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)

#         # Speed relative to target (km/s)
#         v = math.sqrt(rdot_vec[0] ** 2 + rdot_vec[1] ** 2 + rdot_vec[2] ** 2)

#         # Append relevant info (in meters!)
#         m2km = 1000
#         vs.append(v * m2km)
#         ds.append(d * m2km)
#         xs.append(r_vec[0] * m2km)
#         ys.append(r_vec[1] * m2km)
#         zs.append(r_vec[2] * m2km)
#         vxs.append(rdot_vec[0] * m2km)
#         vys.append(rdot_vec[1] * m2km)
#         vzs.append(rdot_vec[2] * m2km)

#     # fig = plt.figure
#     # ax = plt.axes(projection='3d')
#     # ax.set_xlabel('x (m)')
#     # ax.set_ylabel('y (m)')
#     # ax.set_zlabel('z (m)')
#     # ax.plot3D(xs, ys, zs, 'gray')
#     # print(ds)
#     # plt.show()


#     return xs, ys, zs, vxs, vys, vzs, ds, vs

if __name__ == '__main__':

    # Dynamics initializations
    r0 = [-0.17, -0.35, -0.02]  # initial starting position of chaser (km)
    rdot0 = [0.000, 0.00045, 0.0001]  # initial velocity of debris relative to chaser(km/s)
    R = 670 + 6378  # Altitude of orbit (km)
    mu = 398600.5  # Gravitational constant
    omeg = math.sqrt(mu / R ** 3)  # n in the derivations

    # specify time frame and time step
    nframes = 20000
    dt = 0.05

    # simulate debris velocity (linear and angular) in {L} frame from dynamics
    xs, ys, zs, vxs, vys, vzs, ds, v = propagate(dt, nframes, r0, rdot0, omeg)
    fig = plt.figure
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.plot3D(xs, ys, zs, 'gray')
    # print(ds)
    plt.show()