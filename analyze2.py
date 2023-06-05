import dynamics
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import math
import lidarScan2
import boundingbox
from stl import mesh
from estimateOmega import estimate
from associationdata import nearest_search
import pickle

from mpl_toolkits import mplot3d
from matplotlib import pyplot

def skew(vector):
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def getR(x,y,z):
    # we want to find the rotation matrix that takes [x,y,z] to [0,0,1]
    p = np.array([x,y,z]) # position vector, but also z-axis of b frame
    z_L = np.array([0,0,1]) # z-axis of L frame
    z_B = p/np.linalg.norm(p)
    e = (np.cross(z_L, z_B))[:,np.newaxis] # 3x1 axis of rotation
    phi = np.arccos(np.dot(z_B, z_L)) # z_B and z_L are already unit vectors
    R = e@(e.T) + (np.identity(3)-(e@e.T))*np.cos(phi) + tilde(e)*np.sin(phi)
    return R.T

def rodrigues(omega, dt):
    
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R

# initialize debris position, velocity and orientation
O_B = np.array([0,0,0])
O_L = np.array([0,0,0])

with open('sim_kompsat670.pickle', 'rb') as sim_data:
    data = pickle.load(sim_data)
XBs = data[0]
YBs = data[1]
ZBs = data[2]
PBs = data[3]
VBs = data[4]
debris_pos = data[5]
debris_vel = data[6]
Rot_L_to_B = data[7]
Rot_B_to_L = [np.transpose(r) for r in Rot_L_to_B]
omega_L = data[8]
dt = data[9]

# Estimation Loop
XLs = []
YLs = []
ZLs = []
PLs = []
VLs = VBs
nframes = len(VBs)
# Running the simulation

# Initializations in L Frame
L_0 = 2  # Initial Length of box - x
D_0 = 2  # Initial Width of box - y
H_0 = 2  # Initial Height of box - z
p_0 = [-200., -200., -50.]#np.array(debris_pos[0])  # Guess of initial position of debris - *need to formulate guess*
v_0 = [1., 1., 1.]#np.array(debris_vel[0])  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
omega_0 = [5.,-5.,4.] #np.array(omega_L)  # Initial guess of angular velocities - *need to formulate guess*

for i in range(nframes):
    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]

    ##############
    # Update state
    ##############

    # Centroid from vertices
    c_k = 0 

    # Position update
    p_kp1 = v_k * dt + p_k

    # Velocity update
    v_kp1 = v_k.copy()

    # Angular velocity update
    omega_kp1 = omega_k.copy()

    # Compute Jacobian

    # Update Covariance

    # Make updated State vector

    #######################
    # Make measurements
    #######################

    # find bounding box from points

    # Return bounding box and centroid estimate of bounding box

    # Orientation association

    # find angular velocity from LOS velocities
    # 1. Linear Least Squares
    # 2. Rotation of B Frame
    # 3. Kabsch

    # Get Measurement Vector

    # Calculate Kalman gain

    # Calculate Residual

    # Update State

    # Update Covariance

    # Transfer states and covariance from kp1 to k

    # Append for analysis

    # Plot relevant figures