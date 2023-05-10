import matplotlib.pyplot as plt
import numpy as np
from stl import mesh
import math
import rayCast
import lidarNoise
from mpl_toolkits import mplot3d
from matplotlib import pyplot

def point_cloud(O_B, horizontal_resolution, vertical_resolution, h_range, v_range, sat_mesh, sat_pos, v_rel, omega):
    """

    :param O_B:
    :param h_res:
    :param v_res:
    :param h_range:
    :param v_range:
    :return:
    """
    theta_r = np.deg2rad(np.linspace(-h_range/2, h_range/2, horizontal_resolution)) # horizontal FoV
    phi_r = np.deg2rad(np.linspace(-v_range/2,v_range/2, vertical_resolution)) # vertical FoV

    Theta, Phi = np.meshgrid(theta_r, phi_r)

    # to have rays coming out from the z-axis
    # theta rotates wrt x axis, with theta = 0 on the z-axis
    # phi rotates wrt the y' axis (after theta rotation), with phi = 0 on the y-z plane
    # xr = np.cos(Phi)*np.sin(Theta)
    # yr = -np.sin(Phi)
    # zr = np.cos(Theta)*np.cos(Phi)

    rays_polar = np.array([Theta.ravel(), Phi.ravel()]).T
    theta = rays_polar[:,[0]]
    phi = rays_polar[:,[1]]
    rays = np.hstack([(np.cos(phi)*np.sin(theta)), np.sin(phi), np.cos(theta)*np.cos(phi)])

    locations = rayCast.cast(O_B, sat_mesh, rays)  # Returns an array of points (X, Y, Z), irrelevant ones are labeled Nan

    #figure = pyplot.figure()
    #axes = figure.add_subplot(projection='3d')

    #axes.add_collection3d(mplot3d.art3d.Poly3DCollection(sat_mesh.vectors, alpha=0.3))

    # Locations in a target centered frame
    xp = np.array(locations[:,0])
    yp = np.array(locations[:,1])
    zp = np.array(locations[:,2])

    # Filter out NaN values
    Xp = []
    Yp = []
    Zp = []
    for idx, x in enumerate(xp):
        Xp.append(xp[idx]) if ~np.isnan(xp[idx]) else Xp
        Yp.append(yp[idx]) if ~np.isnan(yp[idx]) else Yp
        Zp.append(zp[idx]) if ~np.isnan(zp[idx]) else Zp
    Xp = np.array(Xp)
    Yp = np.array(Yp)
    Zp = np.array(Zp)
    # Transform to Xs, Ys, Zs distances from LiDAR frame
    Xs = Xp - O_B[0]
    Ys = Yp - O_B[1]
    Zs = Zp - O_B[2]
    useful_rel_locations = np.vstack([Xs,Ys,Zs]).T

    v_los_v = np.zeros(np.shape(useful_rel_locations)) # initialize line of sight velocity (a vector)
    v_los_s = np.zeros(len(useful_rel_locations)) # line of sight speed (array of scalars)
    for i, p in enumerate(useful_rel_locations):
        r = p - sat_pos # debris relative position
        u_los = -(p)/np.linalg.norm(p)
        v_los_s[i] = np.dot(np.cross(omega, r), u_los) + np.dot(v_rel, u_los)
        v_los_v[i] = u_los * v_los_s[i]

    # add noise to lidar scan results
    Xn, Yn, Zn, Vn = lidarNoise.add_noise(Xs,Ys,Zs,v_los_s)
    v_los_vn = v_los_v/v_los_s[:,np.newaxis]*Vn[:,np.newaxis]
    # Transform to target coordinate frame (z is pointing in the direction of target,

    #axes.scatter3D(O_B[0],O_B[1],O_B[2], marker = '+', color = 'black', zorder=20)
    #axes.scatter3D(Xs,Ys,Zs, color = 'red', zorder=20)
    #axes.quiver(Xn, Yn, Zn, v_los_vn[:,0], v_los_vn[:,1], v_los_vn[:,2], length = 0.4, normalize=False)
    #scale = sat_mesh.points.flatten()
    #axes.set_xlim3d(left=-5, right=5)
    #axes.set_ylim3d(bottom=-5, top=5)
    #axes.set_zlim3d(bottom=-2, top=8)
    # axes.auto_scale_xyz(scale, scale, scale)
    #axes.set_xlabel('x')
    #axes.set_ylabel('y')
    #axes.set_zlabel('z')
    # pyplot.show()

    return Xn, Yn, Zn, Vn
