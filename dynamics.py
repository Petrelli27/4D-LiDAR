# !/usr/bin/env python3

import math
import matplotlib.pyplot as plt

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


def propagate(dt, nframes, r0, rdot0, omeg):
    """
    Propagate the initial conditions over time given time interval and number of frames desired
    :param dt: time interval in seconds
    :param nframes: total number of frames to be run (total time = dt*nframes)
    :param the rest: are initial conditions for the CW2 function
    :return: lists of x,y,z,vx,vy,vz,v,d position (x,y,z), velocities (vx,vy,vz), speed(v), distances(d)
            in meters relative to the target (at the origin)
    """

    # Final positions and velocities
    xs = []
    ys = []
    zs = []
    vxs = []
    vys = []
    vzs = []
    ds = []
    vs = []

    # Propagate over time
    for i in range(nframes):

        # Current time
        t = dt * i

        # Propogate dynamics - rvec is position of chaser rel target, rdot_vec is velocity
        r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)

        # Distance to target (km)
        d = math.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)

        # Speed relative to target (km/s)
        v = math.sqrt(rdot_vec[0] ** 2 + rdot_vec[1] ** 2 + rdot_vec[2] ** 2)

        # Append relevant info (in meters!)
        m2km = 1000
        vs.append(v * m2km)
        ds.append(d * m2km)
        xs.append(r_vec[0] * m2km)
        ys.append(r_vec[1] * m2km)
        zs.append(r_vec[2] * m2km)
        vxs.append(rdot_vec[0] * m2km)
        vys.append(rdot_vec[1] * m2km)
        vzs.append(rdot_vec[2] * m2km)

    fig = plt.figure
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.plot3D(xs, ys, zs, 'gray')
    # plt.show()

    return xs, ys, zs, vxs, vys, vzs, d, v