import numpy as np


def estimate(x, y, z, c, v_c, v):
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
