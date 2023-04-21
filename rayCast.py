import numpy as np
from stl import mesh
import math

def cast(origin, model, rays):
    # assert that origin and rays are np arrays
    assert(isinstance(origin, np.ndarray))
    assert(isinstance(rays, np.ndarray))
    locations = np.zeros([len(rays), 3])
    # assume each ray is a unit vector
    # Moller Trumbore Algorithm (but uses numpy's linalg.solve() instead of cramer's method)
    for j, ray in enumerate(rays):
        D = ray[:,np.newaxis] # change to column vector
        t = np.array([0.]*len(model))
        # O + tD = (1-u-v)V0 + uV1 + vV2, where V0, V1, V2 are triangle vertices
        triangle_mask = np.array([0]*len(model))
        for i, ni in enumerate(model.normals): # loop over each triangle
            A = np.hstack((-D, (model.v1[i]-model.v0[i])[:,np.newaxis], (model.v2[i]-model.v0[i])[:,np.newaxis]))
            if np.dot(ni, ray)>0 or np.linalg.det(A) < 1e-6: 
                # eliminate backwards facing triangle or near singular matrix
                triangle_mask[i] = 1
                continue
            tuv = np.linalg.solve(A, origin - model.v0[i])
            t[i] = tuv[0]
            u = tuv[1]
            v = tuv[2]
            if t[i]<=0 or u<0 or v<0 or u+v>1:
                triangle_mask[i] = 1 # invalid collision
                continue

        masked_t = np.ma.masked_array(t, mask = triangle_mask)
        true_t = masked_t.min()
        locations[j] = origin + true_t*ray if true_t > 0 else np.array([np.nan, np.nan, np.nan])
    return locations