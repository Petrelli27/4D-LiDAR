import matplotlib.pyplot as plt
import numpy as np
from stl import mesh
import math
import time

from mpl_toolkits import mplot3d
from matplotlib import pyplot

sat_mesh_file = 'irregular sample.stl'
sat_mesh = mesh.Mesh.from_file(sat_mesh_file)

sat_mesh.rotate([1,0,0], np.deg2rad(-10)) # rotate 1 radian about x axis
figure = pyplot.figure()
axes = figure.add_subplot(projection='3d')

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(sat_mesh.vectors, alpha=0.3))

scale = sat_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')
pyplot.show()