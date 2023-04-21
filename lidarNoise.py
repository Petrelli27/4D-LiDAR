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