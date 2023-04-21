import numpy as np
from stl import mesh
import math

from mpl_toolkits import mplot3d
from matplotlib import pyplot

figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')
sat_mesh = mesh.Mesh.from_file('kompsat-1-v6.stl')

#sat_mesh.rotate([1,0,0], math.pi/4) # keep in mind this rotation is always done in the base frame
sat_mesh.translate([0,2000, 100])

# let l_0 be the origin
l_0 = np.array([0,0,0]) # lidar location
v = [0,1,0] # some random ray direction
v = v/np.linalg.norm(v) # normalized unit vector
t = np.einsum('ij,ij->i', (sat_mesh.centroids - l_0), sat_mesh.normals)/(np.einsum('ij,j->i', sat_mesh.normals, v))
# the einsum is numpy magic for doing dot products along a specific dimension of our 2d arrays

# now we need to eliminate all t that doesn't work
# ensure that the point is inside the triangle
# p = l_0 + t[:,np.newaxis]@v[:,np.newaxis].T # p = l_0 + [v*ti for ti in t]
p = p = l_0 + [v*ti for ti in t]
triangle_mask = np.array([0]*len(sat_mesh)) # at first there is no mask
for i, ni in enumerate(sat_mesh.normals):
    if t[i] <= 1 or t[i] == math.inf:
        # ray meets plane with negative length or near zero length (1mm), eliminated
        # ray at infinite length, eliminated
        triangle_mask[i] = 1
        continue
    # elif np.dot(ni, v)>0: 
    #     # if normal and ray vector don't cancel out, the direction is wrong (other side)
    #     triangle_mask[i] = 1
    #     continue
    v1 = sat_mesh.v0[i] - l_0
    v2 = sat_mesh.v1[i] - l_0
    v3 = sat_mesh.v2[i] - l_0
    n1 = np.cross(v2,v1)
    n2 = np.cross(v3,v2)
    n3 = np.cross(v1,v3)
    ray = p[i]-l_0 # ray vector
    if np.dot(ray, n1) <0 or np.dot(ray,n2)<0 or np.dot(ray,n3)<0:
        triangle_mask[i] = 1
        continue
masked_t = np.ma.masked_array(t, mask = triangle_mask)
true_t = masked_t.min() # find the smallest t
inf_t = 50000
true_p = l_0 + v*true_t # location of lidar reflection
ray_p = l_0 + v*inf_t

# sat_v = sat_mesh.vectors
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(sat_mesh.vectors, edgecolor='k'))
x = np.array([l_0[0], ray_p[0]])
y = np.array([l_0[1], ray_p[1]])
z = np.array([l_0[2], ray_p[2]])
xp = np.array([true_p[0]])
yp = np.array([true_p[1]])
zp = np.array([true_p[2]])
axes.plot3D(x,y,z, 'black')
axes.scatter3D(xp,yp,zp, marker = '+', color = 'red')

scale = sat_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
pyplot.show()
print('finished')