from mytools import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
# from sklearn.decomposition import PCA


def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color, linewidth):
    # z1 plane boundary
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], color=color, linewidth=linewidth)

    # z1 plane boundary
    ax.plot([p5[0], p6[0]], [p5[1], p6[1]], [p5[2], p6[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p6[0], p7[0]], [p6[1], p7[1]], [p6[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p7[0], p8[0]], [p7[1], p8[1]], [p7[2], p8[2]], color=color, linewidth=linewidth)
    ax.plot([p8[0], p5[0]], [p8[1], p5[1]], [p8[2], p5[2]], color=color, linewidth=linewidth)

    # Connecting
    ax.plot([p1[0], p5[0]], [p1[1], p5[1]], [p1[2], p5[2]], color=color, linewidth=linewidth)  # W
    ax.plot([p2[0], p6[0]], [p2[1], p6[1]], [p2[2], p6[2]], color=color, linewidth=linewidth)
    ax.plot([p3[0], p7[0]], [p3[1], p7[1]], [p3[2], p7[2]], color=color, linewidth=linewidth)
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color, linewidth=linewidth)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='#9b42f5')


def custom_arccos(x):
    """
    Compute the arccosine of x and ensure the angle is between 0 and 90 degrees.

    Parameters:
    x (float or array-like): The input value(s) for which to compute the arccosine.

    Returns:
    float or ndarray: The arccosine of x, constrained to be between 0 and 90 degrees.
    """
    # Compute the arccosine in radians
    angle_rad = np.arccos(x)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    # Adjust the angle to be within 0 to 90 degrees
    angle_deg = np.where(angle_deg > 90, 180 - angle_deg, angle_deg)

    return angle_deg


def gram_schmidt(vectors):
    """Perform Gram-Schmidt process to create an orthonormal basis."""
    basis = []
    for v in vectors:
        # Orthogonalize
        for b in basis:
            v = v - np.dot(b, v) * b
        # Normalize
        if np.linalg.norm(v) > 1e-10:  # To avoid division by zero
            v = v / np.linalg.norm(v)
            basis.append(v)

    # print(len(basis))
    if len(basis) == 2:
        basis.append(np.cross(basis[0], basis[1]))
    if len(basis) < 2:
        print('Error: not enough basis vectors')
    return np.array(basis)

def boundingbox3D_RANSAC(x, y, z, return_evec=False, visualize=False):

    # Apply Ransac to remove bad points that are alone
    points = np.vstack((x, y, z)).T  # orginal point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Apply RANSAC to segment a plane
    # Parameters
    distance_threshold = 0.04  # Adjust based on your data
    ransac_n = 3
    num_iterations = 1000
    min_inliers = 3  # Minimum number of inliers to consider a plane valid
    # Container for all planes
    all_planes = []
    all_points = np.array([0, 0, 0])  # final point cloud
    remaining_points = pcd
    normal_vecs = []
    # o3d.visualization.draw_geometries([remaining_points])
    while len(remaining_points.points) > min_inliers:
        # Apply RANSAC to segment a plane
        plane_model, inliers = remaining_points.segment_plane(distance_threshold=distance_threshold,
                                                              ransac_n=ransac_n,
                                                              num_iterations=num_iterations)

        normal_vecs.append(plane_model[:3])
        # Extract inlier points
        inlier_cloud = remaining_points.select_by_index(inliers)
        all_planes.append(inlier_cloud)
        all_points = np.vstack((all_points, np.array(inlier_cloud.points)))

        # Extract outlier points
        remaining_points = remaining_points.select_by_index(inliers, invert=True)

        # Optional: visualize each detected plane
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red plane
        # remaining_points.paint_uniform_color([0.0, 1, 0])  # Green remaining points
        # o3d.visualization.draw_geometries([inlier_cloud, remaining_points])

    # Visualize all detected planes
    all_planes_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]  # Different colors for planes
    for idx, plane in enumerate(all_planes):
        color = all_planes_colors[idx % len(all_planes_colors)]
        plane.paint_uniform_color(color)

    # Rankings
    # projections = all_points[1:, :].dot(principal_components.T)
    ranking = np.zeros((len(normal_vecs), len(normal_vecs)))
    for ix, normal_veci in enumerate(normal_vecs):
        for jx, normal_vecj in enumerate(normal_vecs):
            if ix == jx:
                ranking[ix, jx] = 0
            else:
                veci = normal_veci / np.linalg.norm(normal_veci)
                vecj = normal_vecj / np.linalg.norm(normal_vecj)
                ranking[ix, jx] = custom_arccos(np.dot(veci, vecj))

    # print(ranking)
    rankings = np.sum(ranking, axis=0)
    # print(rankings)
    normal_vecs = np.array(normal_vecs)

    # order of planes is red green blue magenta
    if len(remaining_points.points) > 0:
        remaining_points.paint_uniform_color([0, 0, 0])
        all_planes.append(remaining_points)

    if visualize:
        o3d.visualization.draw_geometries(all_planes)
        

    # Pair the vectors with their corresponding values
    paired_list = list(zip(rankings, normal_vecs))

    # Sort the pairs by values in decreasing order
    sorted_pairs = sorted(paired_list, key=lambda x: x[0], reverse=True)

    # Unzip the sorted pairs
    sorted_values, sorted_vectors = zip(*sorted_pairs)
    # sorted_vectors = np.array(sorted_vectors)
    sorted_vectors = normal_vecs

    # update points
    points = all_points[1:, :]  # one to get rid of zero zero zero from beginning
    projections = gram_schmidt(sorted_vectors)
    projections = projections

    means = np.mean(points, axis=0)
    centered_data = points - means
    centered_data = centered_data.T

    aligned_coords = np.matmul(projections, centered_data)

    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(
        aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                          [y1, y2, y2, y1, y1, y2, y2, y1],
                                                          [z1, z1, z1, z1, z2, z2, z2, z2]])

    rrc = np.matmul(projections.T, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))  # rrc = rotated rectangle coordinates

    # Translate back to original location
    rrc = rrc.T + means
    rrc = rrc.T

    c_x = sum(rrc[0, :]) / len(rrc[0, :])
    c_y = sum(rrc[1, :]) / len(rrc[1, :])
    c_z = sum(rrc[2, :]) / len(rrc[2, :])

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot([0, projections[0, 0]], [0, projections[0, 1]], [0, projections[0, 2]], color='red')
        ax.plot([0, projections[1, 0]], [0, projections[1, 1]], [0, projections[1, 2]], color='red')
        ax.plot([0, projections[2, 0]], [0, projections[2, 1]], [0, projections[2, 2]], color='red')
        ax.plot([0, sorted_vectors[0, 0]], [0, sorted_vectors[0, 1]], [0, sorted_vectors[0, 2]], color='blue')
        ax.plot([0, sorted_vectors[1, 0]], [0, sorted_vectors[1, 1]], [0, sorted_vectors[1, 2]], color='blue')
        if len(sorted_vectors) > 2:
            ax.plot([0, sorted_vectors[2, 0]], [0, sorted_vectors[2, 1]], [0, sorted_vectors[2, 2]], color='blue')
        ax.scatter(centered_data[0, :], centered_data[1, :], centered_data[2, :], color='black', s=1)
        associatedBbox_2 = rrc.T - means
        associatedBbox_2 = associatedBbox_2.T
        drawrectangle(ax, associatedBbox_2[:, 0], associatedBbox_2[:, 1], associatedBbox_2[:, 2], associatedBbox_2[:, 3],
                    associatedBbox_2[:, 4], associatedBbox_2[:, 5], associatedBbox_2[:, 6], associatedBbox_2[:, 7],
                    'orange', 2)

        plt.show()

    # ax.scatter(c_x, c_y, c_z, color='b', linewidth=4)
    # evec is aligned with the point cloud and bbox
    if return_evec:
        return rrc, [c_x, c_y, c_z], projections.T, sorted_vectors
    else:
        return rrc, [c_x, c_y, c_z], sorted_vectors



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
