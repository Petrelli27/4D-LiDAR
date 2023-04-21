import numpy as np
from stl import mesh
import math
import time

from mpl_toolkits import mplot3d
from matplotlib import pyplot

figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
sat_mesh = mesh.Mesh.from_file('kompsat-1-v6.stl')

sat_mesh.rotate([1,0,0], math.pi/4) # keep in mind this rotation is always done in the base frame
sat_mesh.translate([0,2040, 1000])

# uses the Moller Trumbore Algorithm instead
# let O_l be the origin
O_l = np.array([0,0,0]) # lidar location
D = np.array([0.5,1,0.01]) # some random ray direction
D = D/np.linalg.norm(D) # normalized unit vector
start_time = time.time()
D = D[:,np.newaxis] # change to column vector
t = np.array([0.]*len(sat_mesh))
# O + tD = (1-u-v)V0 + uV1 + vV2, where V0, V1, V2 are triangle vertices
triangle_mask = np.array([0]*len(sat_mesh))
for i, ni in enumerate(sat_mesh.normals):
    A = np.hstack((-D, (sat_mesh.v1[i]-sat_mesh.v0[i])[:,np.newaxis], (sat_mesh.v2[i]-sat_mesh.v0[i])[:,np.newaxis]))
    if np.linalg.det(A) < 1e-6: # near singular matrix
        triangle_mask[i] = 1
        continue
    tuv = np.linalg.solve(A, O_l-sat_mesh.v0[i])
    t[i] = tuv[0]
    u = tuv[1]
    v = tuv[2]
    if u<0 or v<0 or u+v>1:
        triangle_mask[i] = 1 # invalid collision

masked_t = np.ma.masked_array(t, mask = triangle_mask)
true_t = masked_t.min() # find the smallest t
inf_t = 10000
true_p = O_l + np.squeeze(D)*true_t # location of lidar reflection
ray_p = O_l + np.squeeze(D)*inf_t
end_time = time.time()
# sat_v = sat_mesh.vectors
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(sat_mesh.vectors, alpha=0.3))
x = np.array([O_l[0], ray_p[0]])
y = np.array([O_l[1], ray_p[1]])
z = np.array([O_l[2], ray_p[2]])
xp = np.array([true_p[0]])
yp = np.array([true_p[1]])
zp = np.array([true_p[2]])
axes.plot3D(x,y,z, 'orange')
axes.scatter3D(xp,yp,zp, marker = 'o', color = 'red', zorder=20)

scale = sat_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
pyplot.show()
print(f'calculation in {end_time - start_time} ')