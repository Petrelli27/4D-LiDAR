import numpy as np


def estimate_LLS(x, y, z, c, v_c, v):
    """
    x, y, z are 1xn numpy arrays
    c is the estimated debris center of mass
    v_c is the estimated linear velocity of the debris
    v is 1xn numpy array giving the respective LOS speed
    """
    p = np.vstack([x,y,z]).T
    r = p-c
    u_los = -(p)/(np.linalg.norm(p, axis=1)[:,np.newaxis])
    # b = v - np.dot(v_c, u_los) # this is what we want, but not how np.dot() dehaves
    b = v - u_los@v_c # dot product each row of v_c and u_los
    A = np.zeros(np.shape(p))
    for i, Arow in enumerate(A):
        A[i] = np.cross(r[i], u_los[i])
    
    est_omega = np.linalg.lstsq(A,b, rcond=None)[0]
    # numpy has the linear least squares solver built in
    return est_omega

def estimate_kabsch(prev_box_B, cur_box_B, dt):
    prev_box_B_centroid = np.mean(prev_box_B, axis=0)
    cur_box_B_centroid = np.mean(cur_box_B, axis=0)
    prev_box_B -= prev_box_B_centroid # line up centroids
    cur_box_B -= cur_box_B_centroid
    prev_rec_B = prev_box_B[:,0:2] # remove z since it doesn't matter
    cur_rec_B = cur_box_B[:,0:2]
    
    cov_rec_B = prev_rec_B.T @ cur_rec_B # covariance
    prev_box_B = cur_box_B # for next iteration

    d = 1 if np.linalg.det(cov_rec_B) >= 0 else -1
    U, S, Vh = np.linalg.svd(cov_rec_B)
    Rot_los = Vh.T @ np.array([[1,0],[0,d]]) @ U.T
    # Rot_los = Rot_los.T # why did we want a transpose here? should remove this line
    angle_rot_los = np.arctan2(Rot_los[1,0],Rot_los[0,0])
    omega_los_B =  np.array([0,0,angle_rot_los / dt])

    return omega_los_B

def estimate_rotation_B(Rot_L_to_B, i, dt):
    if i == 0:
        omega_L_to_B = np.array([0,0,0])
    else:
        Rlb = Rot_L_to_B[i-1].T @ Rot_L_to_B[i]  # shorthand
        angle_B_to_B = np.arccos((np.trace(Rlb) - 1)/2)
        axis_B_to_B = 1./(2*np.sin(angle_B_to_B))*np.array([Rlb[2,1]-Rlb[1,2],Rlb[0,2]-Rlb[2,0],Rlb[1,0]-Rlb[0,1]])
        axis_B_to_B = np.transpose(Rot_L_to_B[i]) @ axis_B_to_B / np.linalg.norm(axis_B_to_B)
        omega_L_to_B = (angle_B_to_B * axis_B_to_B)/dt
    return omega_L_to_B
