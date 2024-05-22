from mytools import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D



def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2):
    # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
    ax.plot([x1, x2], [y1, y1], [z1, z1], color='b')  # | (up)
    ax.plot([x2, x2], [y1, y2], [z1, z1], color='b')  # -->
    ax.plot([x2, x1], [y2, y2], [z1, z1], color='b')  # | (down)
    ax.plot([x1, x1], [y2, y1], [z1, z1], color='b')  # <--

    ax.plot([x1, x2], [y1, y1], [z2, z2], color='b')  # | (up)
    ax.plot([x2, x2], [y1, y2], [z2, z2], color='b')  # -->
    ax.plot([x2, x1], [y2, y2], [z2, z2], color='b')  # | (down)
    ax.plot([x1, x1], [y2, y1], [z2, z2], color='b')  # <--

    ax.plot([x1, x1], [y1, y1], [z1, z2], color='b')  # | (up)
    ax.plot([x2, x2], [y2, y2], [z1, z2], color='b')  # -->
    ax.plot([x1, x1], [y2, y2], [z1, z2], color='b')  # | (down)
    ax.plot([x2, x2], [y1, y1], [z1, z2], color='b')  # <--


def bbox3d(x, y, z, return_evec=False):
    """

    :param x: x point cloud data
    :param y: y point cloud data
    :param z: z point cloud data
    :return:  bounding box centroid, l,w,h, vertices
    """
    data = np.vstack([x, y, z])

    means = np.mean(data, axis=1)
    cov = np.cov(data)
    #print(cov)
    # evals, evec = LA.eig(cov)
    evec, s, vh = LA.svd(cov)
    centered_data = data - means[:, np.newaxis]

    # Bounding box based on simple max min (not rotation compensated)
    xmin_mm, xmax_mm, ymin_mm, ymax_mm, zmin_mm, zmax_mm = np.min(centered_data[0, :]), np.max(
        centered_data[0, :]), np.min(centered_data[1, :]), np.max(centered_data[1, :]), np.min(
        centered_data[2, :]), np.max(centered_data[2, :])

    aligned_coords = np.matmul(evec.T, centered_data)

    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(
        aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                          [y1, y2, y2, y1, y1, y2, y2, y1],
                                                          [z1, z1, z1, z1, z2, z2, z2, z2]])

    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]

    nrc = rectCoords(xmin_mm, ymin_mm, zmin_mm, xmax, ymax_mm, zmax_mm)  # nrc = non rotated rectangle
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))  # rrc = rotated rectangle coordinates

    # Translate back to original location
    nrc += means[:, np.newaxis]
    rrc += means[:, np.newaxis]

    # ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b')  # W
    # ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b')  # L
    # ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b')  # H

    #print(np.linalg.norm(rrc[:, 3] - rrc[:, 7]))
    #print(np.linalg.norm(rrc[:, 0] - rrc[:, 1]))
    #print(np.linalg.norm(rrc[:, 1] - rrc[:, 2]))
    #print(np.linalg.norm(nrc[:, 3] - nrc[:, 7]))
    #print(np.linalg.norm(nrc[:, 0] - nrc[:, 1]))
    #print(np.linalg.norm(nrc[:, 1] - nrc[:, 2]))

    #ax.scatter(realigned_coords[0, :], realigned_coords[1, :], realigned_coords[2, :])

    # z1 plane boundary
    #ax.plot(nrc[0, 0:2], nrc[1, 0:2], nrc[2, 0:2], color='b')
    #ax.plot(nrc[0, 1:3], nrc[1, 1:3], nrc[2, 1:3], color='b')
    #ax.plot(nrc[0, 2:4], nrc[1, 2:4], nrc[2, 2:4], color='b')
    #ax.plot(nrc[0, [3, 0]], nrc[1, [3, 0]], nrc[2, [3, 0]], color='b')

    # z2 plane boundary
    #ax.plot(nrc[0, 4:6], nrc[1, 4:6], nrc[2, 4:6], color='b')
    #ax.plot(nrc[0, 5:7], nrc[1, 5:7], nrc[2, 5:7], color='b')
    #ax.plot(nrc[0, 6:], nrc[1, 6:], nrc[2, 6:], color='b')
    #ax.plot(nrc[0, [7, 4]], nrc[1, [7, 4]], nrc[2, [7, 4]], color='b')

    # z1 and z2 connecting boundaries
    #ax.plot(nrc[0, [0, 4]], nrc[1, [0, 4]], nrc[2, [0, 4]], color='b')
    #ax.plot(nrc[0, [1, 5]], nrc[1, [1, 5]], nrc[2, [1, 5]], color='b')
    #ax.plot(nrc[0, [2, 6]], nrc[1, [2, 6]], nrc[2, [2, 6]], color='b')
    #ax.plot(nrc[0, [3, 7]], nrc[1, [3, 7]], nrc[2, [3, 7]], color='b')

    # z1 plane boundary
    #ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b')  # W
    #ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b')  # L
    #ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='b')
    #ax.plot(rrc[0, [3, 0]], rrc[1, [3, 0]], rrc[2, [3, 0]], color='b')

    # z2 plane boundary
    #ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='b')
    #ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='b')
    #ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='b')
    #ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='b')

    # z1 and z2 connecting boundaries
    #ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='b')
    #ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='b')
    #ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='b')
    #ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b')  # H
    # eigen basis
    # ax.plot([means[0], means[0] + evec[0, 0]], [means[1], means[1] + evec[1, 0]], [means[2], means[2] + evec[2, 0]],
            #color='r', linewidth=4)
    # ax.plot([means[0], means[0] + evec[0, 1]], [means[1], means[1] + evec[1, 1]], [means[2], means[2] + evec[2, 1]],
            #color='g', linewidth=4)
    # ax.plot([means[0], means[0] + evec[0, 2]], [means[1], means[1] + evec[1, 2]], [means[2], means[2] + evec[2, 2]],
            #color='k', linewidth=4)

    c_x = sum(rrc[0,:])/len(rrc[0,:])
    c_y = sum(rrc[1,:])/len(rrc[1,:])
    c_z = sum(rrc[2,:])/len(rrc[2,:])

    #ax.scatter(c_x, c_y, c_z, color='b', linewidth=4)
    # evec is aligned with the point cloud and bbox
    if return_evec:
        return rrc, [c_x,c_y,c_z], evec
    else:
        return rrc, [c_x,c_y,c_z]
    
def associated(z_q_k, z_pi_k, z_p_k, R_1):

    # rotate z_pi_k by R (after bbox association)
    # find xmin ymin zmin
    # this is z_p1_k
    R = quat2rotm(z_q_k)
    centered_coords = z_pi_k - np.array(z_p_k).reshape((3,1))
    aligned_coords = R.T @ centered_coords

    ref_points = np.array([[-1,-1,-1],
                          [-1,1,-1],
                          [1,1,-1],
                          [1,-1,-1],
                          [-1,-1,1],
                          [-1,1,1],
                          [1,1,1],
                          [1,-1,1]])
    match_indices = []
    for i, ref_point in enumerate(ref_points):
        match_indices.append(np.argmin(np.linalg.norm(aligned_coords.T - ref_point, axis=1)))
    z_p1_k = aligned_coords.T[match_indices[0]]
    z_p2_k = aligned_coords.T[match_indices[1]]
    z_p3_k = aligned_coords.T[match_indices[2]]
    z_p4_k = aligned_coords.T[match_indices[3]]
    z_p5_k = aligned_coords.T[match_indices[4]]
    z_p6_k = aligned_coords.T[match_indices[5]]
    z_p7_k = aligned_coords.T[match_indices[6]]
    z_p8_k = aligned_coords.T[match_indices[7]] 

    # for i, point in enumerate(aligned_coords.T):


    #     x, y, z = point
    #     if x < 0 and y < 0 and z < 0:
    #         z_p1_k = point
    #     elif x < 0 and y > 0 and z < 0:
    #         z_p2_k = point
    #     elif x > 0 and y > 0  and z < 0:
    #         z_p3_k = point
    #     elif x > 0 and y < 0 and z < 0:
    #         z_p4_k = point
    #     elif x < 0 and y < 0 and z > 0:
    #         z_p5_k = point
    #     elif x < 0 and y > 0 and z > 0:
    #         z_p6_k = point
    #     elif x > 0 and y > 0 and z > 0:
    #         z_p7_k = point
    #     else:
    #         z_p8_k = point


    aligned_coords_final = np.array([z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k]).T

    xmax = max(aligned_coords_final[0, :])
    xmin = min(aligned_coords_final[0, :])
    ymax = max(aligned_coords_final[1, :])
    ymin = min(aligned_coords_final[1, :])
    zmax = max(aligned_coords_final[2, :])
    zmin = min(aligned_coords_final[2, :])
    L = xmax - xmin
    W = ymax - ymin
    H = zmax - zmin

    associatedBbox = R@aligned_coords_final + np.array(z_p_k).reshape((3, 1))

    return associatedBbox, L, W, H

def from_params(p, q, length, width, height):
    R = quat2rotm(q)
    vertices = np.array([
        [-length / 2, -width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2],
        [-length / 2, -width / 2, height / 2],
        [-length / 2, width / 2, height / 2],
        [length / 2, width / 2, height / 2],
        [length / 2, -width / 2, height / 2]])
    bbox = (R @ vertices.T) + p.reshape(3,1) # rotate and move
    return bbox
