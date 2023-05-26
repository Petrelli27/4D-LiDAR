import numpy as np
from math import trunc

# some utility functions
def tilde(v):
    vx = v[0,0]
    vy = v[1,0]
    vz = v[2,0]
    v_tilde = np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]])
    return v_tilde

def getR(x,y,z):
    # we want to find the rotation matrix that takes [x,y,z] to [0,0,1]
    p = np.array([x,y,z]) # position vector, but also z-axis of b frame
    z_L = np.array([0,0,1]) # z-axis of L frame
    z_B = p/np.linalg.norm(p)
    e = (np.cross(z_L, z_B))[:,np.newaxis] # 3x1 axis of rotation
    phi = np.arccos(np.dot(z_B, z_L)) # z_B and z_L are already unit vectors
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + tilde(e)*np.sin(phi)
    return R.T

def nearest_search(pi_k, z_pi_k, z_c_k):
    """ Identify a set of points from the target set that is closest to the
        source set, based on nearest neighbour search with the L2 norm.

        Keyword Arguments:
        ------------------
        pcd_source: list with 3 columns containing point cloud data from the source (x, y ,z)


        pcd_target : list with 3 columns containing point cloud data from the target (x, y ,z)

        Returns:
        --------
        corr_source, corr_target : corr_source[i] is the (x,y,z) point which has the closest
                                    Euclidean distance to point corr_target[i], for each point
                                    in pcd_source

        ec_dist_mean : data containing the mean euclidean distance between the points of corr_source
                        and corr_target for a single iteration of ICP (see ICP funciton later)
        """

    z_pi_k = z_pi_k.T - z_c_k
    ec_dist_i = []
    ec_dists = np.zeros([8,8])
    z_pis = np.zeros([8,3])

    for i, pi_k_i in enumerate(pi_k):

        # Compute the L2 norm between these two points
        ec_dist = np.sqrt(sum((i - j) ** 2 for i, j in zip(z_pi_k.T, pi_k_i)))
        ec_dists[i,:] = ec_dist

    for i, ec_dist in enumerate(ec_dists):
        #print(ec_dists)
        all_idx = np.argmin(ec_dists)
        pi_idx = trunc(all_idx/8)
        z_idx = all_idx % 8
        #print(z_idx)
        z_pis[pi_idx, :] = z_pi_k[z_idx, :]
        ec_dists[pi_idx, :] = 1000
        ec_dists[:, z_idx] = 1000

    z_p1_k = z_pis[0, :] + np.array(z_c_k)
    z_p2_k = z_pis[1, :] + np.array(z_c_k)
    z_p3_k = z_pis[2, :] + np.array(z_c_k)
    z_p4_k = z_pis[3, :] + np.array(z_c_k)
    z_p5_k = z_pis[4, :] + np.array(z_c_k)
    z_p6_k = z_pis[5, :] + np.array(z_c_k)
    z_p7_k = z_pis[6, :] + np.array(z_c_k)
    z_p8_k = z_pis[7, :] + np.array(z_c_k)

    return z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k

def mahalobis_association(pi_k, z_pi_k, z_c_k, cov):


    z_pi_k = z_pi_k.T - z_c_k
    mb_dists = np.zeros([8,8])
    z_pis = np.zeros([8,3])

    for i, pi_k_i in enumerate(pi_k):

        cov_i = cov[3*i:3*i+3, 3*i:3*i+3]
        for j, z_pi_k_j in enumerate(z_pi_k):
            # Compute the Mahalobis distance between two points
            res = z_pi_k_j - pi_k_i
            mb_dist = np.matmul(res, np.matmul(np.linalg.inv(cov_i), res))
            mb_dists[i, j] = mb_dist

    for i, mb_dist in enumerate(mb_dists):
        #print(ec_dists)
        all_idx = np.argmin(mb_dists)
        pi_idx = trunc(all_idx/8)
        z_idx = all_idx % 8
        #print(z_idx)
        z_pis[pi_idx,:] = z_pi_k[z_idx,:]
        mb_dists[pi_idx,:] = 1000
        mb_dists[:, z_idx] = 1000

    z_p1_k = z_pis[0, :] + np.array(z_c_k)
    z_p2_k = z_pis[1, :] + np.array(z_c_k)
    z_p3_k = z_pis[2, :] + np.array(z_c_k)
    z_p4_k = z_pis[3, :] + np.array(z_c_k)
    z_p5_k = z_pis[4, :] + np.array(z_c_k)
    z_p6_k = z_pis[5, :] + np.array(z_c_k)
    z_p7_k = z_pis[6, :] + np.array(z_c_k)
    z_p8_k = z_pis[7, :] + np.array(z_c_k)

    return z_p1_k, z_p2_k, z_p3_k, z_p4_k, z_p5_k, z_p6_k, z_p7_k, z_p8_k