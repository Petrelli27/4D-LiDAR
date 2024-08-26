import copy
import os.path

import matplotlib.pyplot as plt
import pandas as pd

from mytools import *
import numpy as np
import boundingbox
from estimateOmega import estimate_LLS, estimate_kabsch, estimate_rotation_B
from associationdata import rotation_association
import pickle
import scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import random

random.seed(42)
np.random.seed(42)

def get_dimensions(p1, p, q):
    p1_to_p = p - p1
    R = quat2rotm(q)
    p1_debris_frame = R.T @ p1_to_p
    L = 2*p1_debris_frame[0]
    W = 2*p1_debris_frame[1]
    D = 2*p1_debris_frame[2]
    return L, W, D

def sum_of_sinusoids(t_fit, *params_fit):
    y_fit = np.zeros_like(t_fit)
    num_sinusoids = (len(params_fit) - 0) // 4
    for i in range(num_sinusoids):
        A = params_fit[4 * i]
        omega = params_fit[4 * i + 2]
        phi = params_fit[4 * i + 1]
        C = params_fit[4 * i + 3]
        y_fit += A * np.sin(omega * t_fit + phi) + C
    return y_fit


def remove_bias(start_t, dt, y, estimated, num_sinusoids, freq_threshold, freq_skip, true, params_ini):

    nframes = len(y)
    time_interval = (nframes - 1) * dt
    y_orig = y.copy()
    t = np.linspace(start=start_t, stop=start_t + time_interval, num=nframes)
    y = y - estimated

    # Compute the FFT
    y_fft = np.fft.fft(y)
    freq = np.fft.fftfreq(nframes, d=t[1] - t[0])
    y_fft = y_fft[freq > freq_threshold]
    freq = freq[freq > freq_threshold]

    # Compute the magnitudes of the FFT
    magnitudes = np.abs(y_fft)

    # Only consider the positive frequencies (first half of the FFT result)
    positive_frequencies = freq
    positive_magnitudes = magnitudes

    # Find the peaks in the FFT magnitude spectrum
    peaks, _ = find_peaks(positive_magnitudes)

    # Extract peak magnitudes and their corresponding frequencies
    peak_magnitudes = positive_magnitudes[peaks]
    peak_frequencies = positive_frequencies[peaks]

    indices = np.arange(-1, -num_sinusoids * freq_skip - freq_skip, -freq_skip)
    top_peak_indices = np.argsort(peak_magnitudes)[indices[::-1]][::-1]

    # Extract the top three peak frequencies and their magnitudes
    top_frequencies = peak_frequencies[top_peak_indices]
    top_magnitudes = peak_magnitudes[top_peak_indices]

    initial_amplitude = max(y) - min(y)
    initial_phase = 0
    initial_constant = np.mean(y)
    initial_frequencies = 2 * np.pi * top_frequencies

    initial_guess = []
    if len(params_ini) == 0:
        for index in range(0, num_sinusoids):
            initial_guess.append(initial_amplitude)
            initial_guess.append(initial_phase)
            initial_guess.append(initial_frequencies[index])
            initial_guess.append(initial_constant)
    else:
        initial_guess = params_ini

    # Perform the curve fitting
    params, params_covariance = curve_fit(sum_of_sinusoids, t, y, p0=initial_guess)
    constant = max(sum_of_sinusoids(t, *params))
    # print(constant)

    return params, constant


def correct_bias(z_p_k_meas, curr_i, dt_here, parameters, constants, R_i_L_to_B, R_i_B_to_L):
    # correct bias
    bias_z = sum_of_sinusoids(curr_i * dt_here, *parameters[2])
    bias_y = sum_of_sinusoids(curr_i * dt_here, *parameters[1])
    bias_x = sum_of_sinusoids(curr_i * dt_here, *parameters[0])
    z_p_k_B = R_i_L_to_B @ z_p_k_meas
    z_p_k_B[2] = z_p_k_B[2] + constants[2]
    # z_p_k_B[0] = z_p_k_B[0] - bias_x
    # z_p_k_B[1] = z_p_k_B[1] - bias_y
    z_p_k_L = R_i_B_to_L @ z_p_k_B

    return z_p_k_L


def drawrectangle(ax, p1, p2, p3, p4, p5, p6, p7, p8, color, linewidth, label):
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
    ax.plot([p4[0], p8[0]], [p4[1], p8[1]], [p4[2], p8[2]], color=color, linewidth=linewidth, label=label)

    ax.scatter(p1[0], p1[1], p1[2], color='b')
    ax.scatter(p2[0], p2[1], p2[2], color='g')
    ax.scatter(p3[0], p3[1], p3[2], color='r')
    ax.scatter(p4[0], p4[1], p4[2], color='c')
    ax.scatter(p5[0], p5[1], p5[2], color='m')
    ax.scatter(p6[0], p6[1], p6[2], color='y')
    ax.scatter(p7[0], p7[1], p7[2], color='k')
    ax.scatter(p8[0], p8[1], p8[2], color='#9b42f5')


def skew(vector):
    vector = list(vector)
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def rodrigues(omega, dt):
    e_omega = omega / np.linalg.norm(omega)  # Unit vector along omega
    phi = np.linalg.norm(omega) * dt
    ee_t = np.matmul(e_omega.reshape(len(e_omega), 1), e_omega.reshape(1, len(e_omega)))
    e_tilde = skew(e_omega)
    R = ee_t + (np.eye(len(e_omega)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R


def rodrigues_axis_angle(axis, angle):
    axis = axis/np.linalg.norm(axis)  # Unit vector along omega
    phi = angle
    ee_t = np.matmul(axis.reshape(len(axis), 1), axis.reshape(1, len(axis)))
    e_tilde = skew(axis)
    R = ee_t + (np.eye(len(axis)) - ee_t) * np.cos(phi) + e_tilde * np.sin(phi)
    return R


def verticeupdate(dt, x_k):
    # Decompose the state vector
    p_k = x_k[:3]
    v_k = x_k[3:6]
    omega_k = x_k[6:9]
    p1_k = x_k[9:12]

    # Rotation matrix - rodrigues formula
    R_k_kp1 = rodrigues(omega_k, dt)
    R_k_kp1 = R_k_kp1

    # Translate vertex to origin
    p1_ko = p1_k - p_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1, p1_ko.reshape(len(p1_ko), 1))

    # Translate vertex back to new expected origin
    p1_kp1 = (p1_kp1o.T + p_k + v_k * dt).ravel()

    return p1_kp1, R_k_kp1


def orientationupdate(dt, x_k):
    # Decompose the state vector
    omega_k = x_k[6:9]
    q_k = x_k[12:16]
    qw = q_k[0];
    qx = q_k[1];
    qy = q_k[2];
    qz = q_k[3]

    # hamilton = np.array([-omega_k[0] * q_k[1] - omega_k[1] * q_k[2] - omega_k[2] * q_k[3],
    #             omega_k[0] * q_k[0] + omega_k[2] * q_k[2] - omega_k[1] * q_k[3],
    #             omega_k[1] * q_k[0] - omega_k[2] * q_k[1] + omega_k[0] * q_k[3],
    #             omega_k[2] * q_k[0] + omega_k[1] * q_k[1] - omega_k[0] * q_k[2]])

    dqkdt = 0.5 * np.array([[-qx, -qy, -qz],
                            [qw, qz, -qy],
                            [-qz, qw, qx],
                            [qy, -qx, qw]]) @ omega_k

    q_kp1 = normalize_quat(dqkdt * dt + q_k)
    q_kp1_pos = q_kp1  # if q_kp1[0] >=0 else -q_kp1
    return q_kp1_pos


def get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, q_ini):

    Rot_0 = quat2rotm(q_ini)
    # Rot_0 = np.eye(3)
    # print(Rot_0)
    q_s = []
    for i in range(len(debris_pos)):

        # get rotation matrix for that timestep
        Rot_i = rodrigues(omega_true, dt * i)
        q_i = rotm2quat(Rot_i @ Rot_0)
        if i == 0:
            q_s.append(q_i)
        else:
            q_i_alt = -q_i
            q_prev = q_s[i-1]
            if np.linalg.norm(q_i - q_prev) < np.linalg.norm(q_i_alt - q_prev):
                q_s.append(q_i)
            else:
                q_s.append(q_i_alt)
    return q_s

def rotate_to_within_45_q_true(q_true, q_ini):
    R_true = quat2rotm(q_true)
    R_ini = quat2rotm(q_ini)
    R_rel = R_true.T @ R_ini
    # R_rel = R_ini.T @ R_true # R_true.T @ R_ini
    axes_candidates = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]
    angles = []
    Rs = []
    for x in axes_candidates:
        for y in axes_candidates:
            if np.dot(x, y) == 0:
                z = np.cross(x, y)
                R_candidate = np.vstack([x,y,z])
                R_net = R_rel @ R_candidate
                theta = np.arccos(0.5*(np.trace(R_net)-1))
                angles.append(theta)
                Rs.append(R_candidate)
            else:
                continue
    best_index = np.argmin(angles)
    q_ini_adjusted = rotm2quat(Rs[best_index])
    return q_ini_adjusted

def recalibrate_true_orientation(q_true, q_measurement, recalibrate_frame):
    q_recalibrate = q_true[recalibrate_frame]
    R_recalibrate = quat2rotm(q_recalibrate)
    R_measurement = quat2rotm(q_measurement)
    R_rel = R_measurement.T @ R_recalibrate
    axes_candidates = [[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]]
    angles = []
    Rs = []
    # find the best 90 degree rotation to match q_recalibrate and q_measurement
    for x in axes_candidates:
        for y in axes_candidates:
            if np.dot(x, y) == 0:
                z = np.cross(x, y)
                R_candidate = np.vstack([x,y,z])
                R_net = R_rel @ R_candidate
                theta = np.arccos(0.5*(np.trace(R_net)-1))
                angles.append(theta)
                Rs.append(R_candidate)
            else:
                continue
    best_index = np.argmin(angles)
    R_offset = Rs[best_index]
    q_true_recalibrated = q_true.copy()
    for i, q in enumerate(q_true):
        R_true_old = quat2rotm(q)
        R_true_new = R_true_old @ R_offset
        q_new = rotm2quat(R_true_new)
        q_true_recalibrated[i] = q_new
    return q_true_recalibrated

def run(pickle_file, configs, logger):

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # initialize debris position, velocity and orientation
    O_B = np.array([0, 0, 0])
    O_L = np.array([0, 0, 0])

    with open(os.path.join(configs['pickle_directory_name'], pickle_file), 'rb') as sim_data:
        # with open('sim_kompsat_neg_om_longer.pickle', 'rb') as sim_data:
        # with open('sim_new_conditions.pickle', 'rb') as sim_data:
        data = pickle.load(sim_data)

    XBs = data['XBs']
    YBs = data['YBs']
    ZBs = data['ZBs']
    PBs = data['PBs']
    VBs = data['VBs']

    debris_pos = data['debris_pos']
    debris_vel = data['debris_vel']
    Rot_L_to_B = data['Rot_L_to_B']
    Rot_B_to_L = [np.transpose(r) for r in Rot_L_to_B]
    omega_L = data['omega_L']
    dt = data['dt']
    initial_angle_rotation = data['angle_0']
    RC_flag = False

    # Estimation Loop
    XLs = []  # store point cloud x in L
    YLs = []
    ZLs = []
    PLs = []  # store x, y, z point cloud in L
    VLs = VBs  # store velocity point cloud
    x_s = []  # store states over time
    z_s = []  # store measurements over time
    P_s = []  # store covariances in time
    errors = [0]
    nframes = len(VBs)

    # Running the simulation - Initializations

    # Initializations in L Frame
    vT_0 = configs['ini_vel_guess']  # Initial guess of relative velocity of debris, can be based on how fast plan to approach during rendezvous
    omega_0 = configs['ini_ang_vel_guess']  # rad/s
    omega_true = omega_L
    q_ini = configs['ini_orientation']
    q_true = np.array(get_true_orientation(Rot_L_to_B, omega_true, debris_pos, dt, rotm2quat(rodrigues_axis_angle(omega_L, np.deg2rad(initial_angle_rotation)))))
    # q_ini = rotate_to_within_45_q_true(q_true[0,:], q_ini)
    q_true_ini = q_true.copy() # keep track of q_true for debug purposes
    # q_ini = q_true[0,:] # start with q_true for debug purposes only
    p_0 = np.array([0., 0., 0.])   # these should be arbitrary, first position and vertex is according to first measurement, just to get num_states
    p1_0 = p_0 + np.array([0., 0., 0.])
    x_0 = np.hstack([p_0, vT_0, omega_0, p1_0, q_ini])
    num_states = len(x_0)

    # Initial covariance
    P_0 = np.diag(configs['ini_covariance_guess'])  # Initial Covariance matrix

    # Process noise covariance matrix
    qp = configs['ini_process_noise_cov'][0]
    qv = configs['ini_process_noise_cov'][1]
    qom = configs['ini_process_noise_cov'][2]
    qp1 = configs['ini_process_noise_cov'][3]
    qq = configs['ini_process_noise_cov'][4]
    Q = np.diag([qp, qp, qp, qv, qv, qv, qom, qom, qom, qp1, qp1, qp1, qq, qq, qq, qq])

    # Measurement noise covariance matrix
    p = configs['ini_meas_noise_cov'][0]
    om = configs['ini_meas_noise_cov'][1]
    p1 = configs['ini_meas_noise_cov'][2]
    q = configs['ini_meas_noise_cov'][3]
    R1 = np.diag([p, p, p, om, om, om, p1, p1, p1, q, q, q, q])
    R2 = np.diag([p, p, p, om, om, om, p1, p1, p1])

    z_q_k_1_previous = np.zeros((4,))
    z_q_k_2_previous = np.zeros((4,))
    # q_km1 = np.zeros((4,))

    # Measurement matrix
    H1 = np.zeros([len(P_0)-3, len(P_0)])  # no measuring of velocity
    H1[0:3,0:3] = np.eye(3)
    H1[3:,6:] = np.eye(10)
    bad_attitude_measurement_flag = False
    adapt = False

    H2 = np.zeros([9,16])
    H2[0:3,0:3] = np.eye(3)
    H2[3:6,6:9] = np.eye(3)
    H2[6:, 9:12] = np.eye(3)

    # Kabsch estimation parameters
    n_moving_average = configs['moving_average_len']
    settling_time = configs['settling_time']
    # Record keeping for angular velocity estimate
    omegas_kabsch_b = np.zeros((nframes, 3))
    omegas_lls_b = np.zeros((nframes, 3))
    omega_kabsch_b_box = np.zeros((n_moving_average, 3))

    # Get Final measurement vectors
    q_kp1s = []
    z_p_s = [p_0]
    zv_mags = []
    z_omegas = []
    z_v_s = []
    z_s = []
    z_rans = []
    z_pcas = []
    x_s = [x_0]
    P_s = []
    original_pos_meas = []
    estimated_pos = [p_0]
    rotation_errors = [0]

    # bias config
    interval_time = 0  # for bias part
    done = 0
    params_x = []
    params_y = []
    params_z = []
    centroids_inB = []
    true_pos_inB = []
    q_kp1s =[]
    metrics = []
    z_s_all = []
    without_correction = []
    bbox1_dimensions =[]
    bbox2_dimensions = []
    bbox3_dimensions = [2*(p_0 - p1_0)]

    # ukf weight values
    alpha = configs['alpha']
    beta = configs['beta']
    kappa = configs['kappa']
    # cholesky decomposition to calc sigmapoints
    epsilon = configs['epsilon']  # constant to ensure positive defineteness
    dimL = len(x_0)
    lambd = alpha ** 2 * (dimL + kappa) - dimL
    w_0_m = lambd / (lambd + dimL)  # first weight for computing the mean
    w_j_m = 0.5 / (lambd + dimL)  # consequent weights for computing the mean
    w_0_c = w_0_m + (1 - alpha ** 2 + beta)  # first weight for computing covariance
    w_j_c = w_j_m
    tolerance = configs['tolerance']  # threshold to which the ISPKF iterates, i.e., iterate until difference between states is below threshold


    # data gathering
    master_file = pd.DataFrame(columns=configs['master_file_columns'])
    # various columns
    file_names = []
    ransac_errors = []
    pca_errors = []
    prediction_errors = []
    perfect_metric_choices = []
    short_metric_choices = []
    number_of_pointss = []
    points_diffs = []
    z_spreads = []
    x_spreads = []
    y_spreads = []
    pca_prev_diffs = []
    ransac_prev_diffs = []
    x_spread_diffs = []
    y_spread_diffs = []
    z_spread_diffs = []
    ransac_pca_diffs = []
    pca_pred_diffs = []
    ransac_pred_diffs = []

    metric_boxes = {"pca 0": [0, 0, 0, 0, 0],
               "ransac 0": [0, 0, 0, 0, 0],
               "ransac 1": [0, 0, 0, 0, 0],
               "ransac 2": [0, 0, 0, 0, 0],
               "ransac 3.1": [0, 0, 0, 0, 0],
               "pca 3.1": [0, 0, 0, 0, 0],
               "pca 3.2": [0, 0, 0, 0, 0],
               "pca 3.3": [0, 0, 0, 0, 0],
               "ransac 4.1": [0, 0, 0, 0, 0],
               "ransac 4.2": [0, 0, 0, 0, 0],
               "pca 4.1": [0, 0, 0, 0, 0],
               "pred 4.1": [0, 0, 0, 0, 0],
               "ransac 5": [0, 0, 0, 0, 0],
               "ransac 6": [0, 0, 0, 0, 0],
               "ransac 7": [0, 0, 0, 0, 0],
               "ransac 8": [0, 0, 0, 0, 0]}

    for i in range(nframes):
        PLs.append((Rot_L_to_B[i].T @ (PBs[i]).T).T)
        # find bounding box from points
        XLs.append(PLs[i][:, 0])
        YLs.append(PLs[i][:, 1])
        ZLs.append(PLs[i][:, 2])
        X_i = XLs[i]
        Y_i = YLs[i]
        Z_i = ZLs[i]
        z_pi_k_1, z_p_k_1, R_1, evals = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
        z_q_k_1 = rotm2quat(R_1)
        z_pi_k_2, z_p_k_2, R_1_2, normal_vecs, ranking, num_planes = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, z_q_k_1, True, False)
        if R_1_2.size == 0:
            ransac_error = True
        else:
            ransac_error = False

        if not ransac_error:
            z_q_k_2, _, _ = rotation_association(z_q_k_1, R_1_2)
            if np.rad2deg(quat_angle_diff(z_q_k_1, z_q_k_2)) < configs['ransac_pca_threshold']:
                starting_frame = i
                q_key_measurement = z_q_k_2
                q_true = recalibrate_true_orientation(q_true, q_key_measurement, starting_frame)
                #logger.info(f"pickle {pickle_file} recalibrated q_true based on frame {starting_frame}")
                break
    # q_ini = q_true[0,:]
    q_ini = rotate_to_within_45_q_true(q_true[0,:], q_ini)

    for i in range(nframes):

        if rank == 0:
            logger.info(f"Iteration {i} of {nframes}")
        else:
            pass

        # Use first measurements for initializations of states - not implemented currently, just chose initial states up top
        if i > 0:
            # state vector as mean for sigmapoint transform
            mu_sp = x_k.copy()

            # covariance matrix for sigmapoint transform
            sigma_zz = P_k.copy()

            # cholesky decomposition for lower triangular matrix
            try:
                L = scipy.linalg.cholesky(sigma_zz, lower=True)
            except np.linalg.LinAlgError:  # happens when the diagonal is zero but numerically speaking has negative elements
                np.fill_diagonal(sigma_zz, sigma_zz.diagonal() + epsilon)
                L = scipy.linalg.cholesky(sigma_zz, lower=True)

            # initial sigmapoint
            sp_0 = mu_sp

            # other sigmapoints
            sp_s = [sp_0]
            sqrt_term = np.sqrt(dimL + lambd)
            for idx in range(0, dimL):
                col_i_L = L[:, idx]
                sp_i = mu_sp + sqrt_term * col_i_L
                sp_s.append(sp_i)
            for idx in range(0, dimL):
                col_i_L = L[:, idx]
                sp_i_L = mu_sp - sqrt_term * col_i_L
                sp_s.append(sp_i_L)

            # pass each point through prediction model
            x_kp1 = np.zeros((num_states,))
            sp_kp1s = []
            for jdx, sp in enumerate(sp_s):

                # Decompose the state vector
                p_k = sp[:3]
                v_k = sp[3:6]
                omega_k = sp[6:9]
                p1_k = sp[9:12]
                q_k = sp[12:]

                ##############
                # Prediction
                ##############

                # Position update
                p_kp1 = v_k * dt + p_k

                # Velocity update
                v_kp1 = v_k.copy()

                # Angular velocity update
                omega_kp1 = omega_k.copy()

                # Vertex update
                p1_kp1, R_k_kp1 = verticeupdate(dt, sp)

                # Orientation Update
                if i == 1:
                    q_kp1s.append(q_ini)

                q_kp1 = orientationupdate(dt, sp)
                q_kp1s.append(q_kp1)

                sp_kp1_jdx = np.hstack([p_kp1, v_kp1, omega_kp1, p1_kp1, q_kp1]).ravel()
                sp_kp1s.append(sp_kp1_jdx)

                # weighted sum of sigma points to get updated state
                if jdx == 0:
                    x_kp1 += w_0_m * sp_kp1_jdx
                else:
                    x_kp1 += w_j_m * sp_kp1_jdx

            z_v_s.append(x_k[3:6])

            # update covariance
            P_kp1 = np.zeros((num_states, num_states))
            for kdx, sp in enumerate(sp_kp1s):
                diff = sp - x_kp1
                if kdx == 0:
                    P_kp1 += w_0_c * np.outer(diff, diff.T)
                else:
                    P_kp1 += w_j_c * np.outer(diff, diff.T)

            # add proces noise
            P_kp1 += Q

            # try and smooth out covariance off diagonals to ensure symmetry
            P_kp1 = 0.5 * P_kp1 + 0.5 * P_kp1.T

        #######################
        # Measurements
        #######################

        PLs.append((Rot_L_to_B[i].T @ (PBs[i]).T).T)
        # find bounding box from points
        XLs.append(PLs[i][:, 0])
        YLs.append(PLs[i][:, 1])
        ZLs.append(PLs[i][:, 2])
        X_i = XLs[i]
        Y_i = YLs[i]
        Z_i = ZLs[i]
        
        num_points = len(Z_i)
        #logger.info(f"Number of points in point cloud for rank {rank}: {num_points}")

        # Return bounding box and centroid estimate of bounding box
        z_pi_k_1, z_p_k_1, R_1, evals = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
        if i == 0:
            q_kp1 = q_ini
        z_pi_k_2, z_p_k_2, R_1_2, normal_vecs, ranking, num_planes = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, q_kp1, True, False)

        if R_1_2.size == 0:
            ransac_error = True
        else:
            ransac_error = False

        ############
        # bias removal
        ############

        original_pos_meas.append(z_p_k_1)
        centroids_inB.append(Rot_L_to_B[i] @ z_p_k_1)
        true_pos_inB.append(Rot_L_to_B[i] @ debris_pos[i, :])

        curr_t = i * dt
        t_start = configs['start_time']  # when the first bias calculation should be initiated
        t_interval = configs['interval']  # how many seconds of data should be collected each time

        # grab data every interval
        if curr_t >= (t_start + t_interval):
            if (curr_t + t_start) % t_interval == 0 and done == 0:  # grab new data
                interval_time = curr_t - t_interval
                z_in_b = [Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(original_pos_meas)]
                z = np.array(z_in_b)
                z = z[int(interval_time / dt):, :]
                estimated_inB = np.array([Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(estimated_pos)])
                estimated = np.array(estimated_inB)
                estimated = estimated[int(interval_time / dt):, :]
                true_inB = np.array([Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(debris_pos)])
                true = np.array(true_inB)
                true = true[int(interval_time / dt):int((interval_time + t_interval) / dt) + 1, :]

                thresh = configs['threshold']  # initial threshold to remove frequencies obtained from crosstalk with baseband frequency
                num_sin = configs['number_of_sinusoids']  # number of sinusoids to use to fit the data
                skip = configs['number_of_skips']  # when choosing frequencies from frequency according to decreasing magnitude, skips this many frequencies
                params_z, constant_z = remove_bias(interval_time, dt, z[:, 2], estimated[:, 2], num_sin, thresh, skip, true[:, 2], params_z)
                parameters = [params_x, params_y, params_z]

                constants = [0, 0, constant_z]
                done = 1

        #####################

        # Orientation association
        # R_1 is obtained from bounding box
        if i == 0:
            z_q_k_1 = rotm2quat(R_1)  # this rotation is to set initial orientation to match with true
            if not ransac_error:
                z_q_k_2, _, _ = rotation_association(z_q_k_1, R_1_2)
            # z_q_k_1 = rotm2quat(R_1 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0.,
            #                                                                   1.]]))  # this rotation is to set initial orientation to match with true
            # if not ransac_error:
            #     z_q_k_2 = rotm2quat(R_1_2 @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0.,
            #                                                                     1.]]))  # this rotation is to set initial orientation to match with true
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
            perfect_metric = False
        else:
            z_q_k_1, _, error = rotation_association(q_kp1, R_1)
            if not ransac_error:
                z_q_k_2, bad_attitude_measurement_flag_2, error_2 = rotation_association(q_kp1, R_1_2)
            if quat_angle_diff(z_q_k_1, q_true[i, :]) > np.deg2rad(35):
                perfect_metric = True
            else:
                perfect_metric = False

        if i > 0:
            LWD = 2 * quat2rotm(q_kp1).T @ (p_kp1 - p1_kp1)
            L = LWD[0];
            W = LWD[1];
            D = LWD[2]
            predictedBbox = boundingbox.from_params(p_kp1, q_kp1, L, W, D)  # just use the predicted box instead

            # first use q from R_1 to get L,W,D
            # then use z_q_k (not perfectly aligned) to get
        associatedBbox_1, Lm, Wm, Dm = boundingbox.associated(z_q_k_1, z_pi_k_1, z_p_k_1,
                                                              R_1)  # L: along x-axis, W: along y-axis D: along z-axis
        z_p1_k_1 = associatedBbox_1[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)
        if not ransac_error:
            associatedBbox_2, Lm_2, Wm_2, Dm_2 = boundingbox.associated(z_q_k_2, z_pi_k_2, z_p_k_2, R_1_2)
            z_p1_k_2 = associatedBbox_2[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)

        if i == 0:
            associatedBbox = associatedBbox_1.copy()
            z_p1_k = associatedBbox_1[:, 0]
            z_q_k_1_previous = z_q_k_1.copy()
            if not ransac_error:
                z_q_k_2_previous = z_q_k_2.copy()

        if i > 0:
            ###########################################################################3
            if not ransac_error:
                ransac_pred_diff = np.rad2deg(quat_angle_diff(q_kp1, z_q_k_2))
                ransac_pca_diff = np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_1))
                ransac_prev_diff = np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_2_previous))
                ransac_true_diff = np.rad2deg(quat_angle_diff(z_q_k_2, q_true[i, :]))
            else:
                nonsense_value = 1000
                ransac_pred_diff = nonsense_value
                ransac_pca_diff = nonsense_value
                ransac_prev_diff = nonsense_value
                ransac_true_diff = nonsense_value


            pca_pred_diff = np.rad2deg(quat_angle_diff(q_kp1, z_q_k_1))
            pca_prev_diff = np.rad2deg(quat_angle_diff(z_q_k_1, z_q_k_1_previous))
            pca_true_diff = np.rad2deg(quat_angle_diff(z_q_k_1, q_true[i, :]))
            pred_true_diff = np.rad2deg(quat_angle_diff(q_kp1, q_true[i, :]))
            # pred_prev_diff = np.rad2deg(quat_angle_diff(q_kp1, q_km1))
            short_metric_thresh = configs['short_metric_thresh']

            if i > configs['start']:
                pca_prev_thresh = configs['previous_threshold_multiplier'] * dt * np.rad2deg(np.linalg.norm(omega_kp1))
                ran_prev_thresh = configs['previous_threshold_multiplier'] * dt * np.rad2deg(np.linalg.norm(omega_kp1))


        if i == 0:
            use_measurement = 2  # ransac by default
            short_metric_choice = "ransac 0"
        elif i > configs['start']:
            # short metric
            RP = ransac_pred_diff < short_metric_thresh
            CP = pca_pred_diff < short_metric_thresh
            RC = ransac_pca_diff < short_metric_thresh
            RR = ransac_prev_diff < ran_prev_thresh
            CC = pca_prev_diff < pca_prev_thresh
            if RP and CP and (not RC):
                use_measurement = 2  # ransac
                short_metric_choice = "ransac 1"
            elif RP and (not CP) and (not RC):
                use_measurement = 2  # ransac
                short_metric_choice = "ransac 2"
            elif (not RP) and CP and (not RC):
                if RR and CC:
                    use_measurement = 1
                    short_metric_choice = "pca 3.1"
                elif RR and (not CC):
                    use_measurement = 2
                    short_metric_choice = "ransac 3.1"
                elif CC and (not RR):
                    use_measurement = 1
                    short_metric_choice = "pca 3.2"
                else:
                    use_measurement = 1
                    short_metric_choice = "pca 3.3"
            elif (not RP) and (not CP) and (not RC):
                if RR and CC:
                    use_measurement = 1
                    short_metric_choice = "ransac 4.1"
                elif RR and (not CC):
                    use_measurement = 2
                    short_metric_choice = "ransac 4.2"
                elif CC and (not RR):
                    use_measurement = 1
                    short_metric_choice = "pca 4.1"
                else:
                    use_measurement = 3
                    short_metric_choice = "pred 4.1"
            elif (not RP) and CP and RC:
                use_measurement = 2  # ransac
                short_metric_choice = "ransac 5"
            elif RP and (not CP) and RC:
                use_measurement = 2
                short_metric_choice = "ransac 6"
            elif (not RP) and (not CP) and (RC):
                use_measurement = 2
                short_metric_choice = "ransac 7"
            elif (RP and CP and RC):
                use_measurement = 2
                short_metric_choice = "ransac 8"
        else:  # at the start, don't use prediction
            if ransac_pred_diff > pca_pred_diff:
                use_measurement = 1
                short_metric_choice = "pca 0"
            else:
                use_measurement = 2
                short_metric_choice = "ransac 0"
        metric_boxes[short_metric_choice][0] += 1
        short_metric_choices.append(short_metric_choice)

        if i == 0:
            ideal_measurement = 2
            perfect_metric_choice = 'first'
        else:
            if ransac_true_diff < configs['true_orientation_difference']:
                ideal_measurement = 2
                perfect_metric_choice = "ransac"
                metric_boxes[short_metric_choice][2] += 1
            else:
                if pca_true_diff < configs['true_orientation_difference']:
                    ideal_measurement = 1
                    perfect_metric_choice = "pca"
                    metric_boxes[short_metric_choice][1] += 1
                else:
                    if pred_true_diff < configs['true_orientation_difference']:
                        ideal_measurement= 3
                        perfect_metric_choice = "pred"
                        metric_boxes[short_metric_choice][3] += 1
                    else:
                        values = [pca_true_diff, ransac_true_diff, pred_true_diff]
                        min_index, min_value = min(enumerate(values), key=lambda x: x[1])
                        ideal_measurement = min_index + 1  # we want from 1 to 3
                        if ideal_measurement == 1:
                            perfect_metric_choice = "pca"
                            metric_boxes[short_metric_choice][2] += 1
                            metric_boxes[short_metric_choice][4] += 1
                        elif ideal_measurement == 2:
                            perfect_metric_choice = "ransac"
                            metric_boxes[short_metric_choice][1] += 1
                            metric_boxes[short_metric_choice][4] += 1
                        elif ideal_measurement == 3:
                            perfect_metric_choice = "prediction"
                            metric_boxes[short_metric_choice][3] += 1
                            metric_boxes[short_metric_choice][4] += 1
                        else:
                            perfect_metric_choice = "error"
        perfect_metric_choices.append(perfect_metric_choice)

        if configs['use_perfect_metric']:
            use_measurement = ideal_measurement
        if use_measurement == 2:
            # use ransac
            z_q_k = z_q_k_2.copy()
            z_pi_k = z_pi_k_2.copy()
            z_p_k = z_p_k_2.copy()
            z_p1_k = associatedBbox_2[:, 0]
            associatedBbox = associatedBbox_2.copy()
            adapt = False
            choice = 'ransac'
        elif use_measurement == 1:
            # use pca
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
            z_p1_k = associatedBbox_1[:, 0]
            associatedBbox = associatedBbox_1.copy()
            adapt = False
            choice = 'pca'
        else:
            # use prediction
            associatedBbox = predictedBbox.copy()
            z_p_k = z_p_k_1.copy()
            z_p1_k = associatedBbox[:, 0]
            adapt = True
            choice = 'prediction'

                ######################################

        without_correction.append(z_p_k)
        bbox1_dimensions.append([Lm, Wm, Dm])
        if not ransac_error:
            bbox2_dimensions.append([Lm_2, Wm_2, Dm_2])
        else:
            bbox2_dimensions.append([0, 0, 0])
        if curr_t >= (t_start + t_interval):
            z_p_k_z = correct_bias(z_p_k, i, dt, parameters, constants, Rot_L_to_B[i], Rot_B_to_L[i])
            z_p_k = z_p_k_z


        # find angular velocity from LOS velocities
        if i > 0:
            # 1. Linear Least Squares
            omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i] @ z_p_k, Rot_L_to_B[i] @ v_k, VBs[i])
            omega_LLS = Rot_B_to_L[i] @ omega_LLS_B

        # 2. Rotation of B Frame
        omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)

        # 3. Kabsch
        ################ to use Kabsch you need i > 0, to wait for state initializations?
        if i == 0:
            omega_los_L = np.array([0, 0, 0])
            prev_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
            prev_box_B = (Rot_L_to_B[i] @ prev_box_L.T).T
        else:
            cur_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
            cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
            # rotate previous box with everything else
            # prev_box_B = (rodrigues((omega_LLS + omega_L_to_B), dt) @ prev_box_B.T).T
            omega_los_B = estimate_kabsch(prev_box_B, cur_box_B, dt)
            prev_box_B = cur_box_B.copy()  # for next iteration

            # using moving average to smooth out omega_los_B
            omega_kabsch_b_box[i % n_moving_average] = omega_los_B
            if i < n_moving_average:
                omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i + 1], axis=0)
            else:
                omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
            omega_los_L = Rot_B_to_L[i] @ omega_los_B_averaged

        # Combine angular velocity estimates
        if i == 0:
            z_omega_k = omega_0
        elif i <= settling_time:
            z_omega_k = omega_LLS + omega_L_to_B  # ignores kabsch
        else:
            z_omega_k = omega_LLS + omega_L_to_B + omega_los_L

        #################################

        ##############
        # Update - Combine Measurement and Estimates
        ##############

        # Compute Measurement Vector
        # if False:
        if adapt:
            z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k])
            H = H2
            R = R2
        else:
            z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k])
            H = H1
            R = R1

        # Set initial states to measurements
        if i == 0:
            x_k = np.hstack([z_p_k, vT_0, z_omega_k, z_p1_k, q_ini])  # state
            P_k = P_0.copy()  # covariance matrix

        num_meas = len(z_kp1)

        if i > 0:
            #################
            # iterated measurement update
            #################
            x_op = x_kp1.copy()
            P_op = P_kp1.copy()

            current_difference = 1  # initialize to a high value so that it can enter the loop, this is the current difference between states of consecutive iterations

            # iterate to desired threshold
            while current_difference > tolerance:
                ####################
                # measurement update
                ###################

                # state vector as mean for sigmapoint transform
                mu_sp_m = x_op.copy()

                # stack covariance matrix with process noise
                sigma_zz_m = P_op.copy()

                # cholesky, ensure positive definiteness
                try:
                    L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)
                except np.linalg.LinAlgError:
                    np.fill_diagonal(sigma_zz_m, sigma_zz_m.diagonal() + epsilon)
                    L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)

                # initial sigmapoint
                sp_0_m = mu_sp_m

                # other sigmapoints
                sp_s_m = [sp_0_m]
                sqrt_term_m = np.sqrt(dimL + lambd)
                for idx in range(0, dimL):
                    col_i_L_m = L_m[:, idx]
                    sp_i_m = mu_sp_m + sqrt_term_m * col_i_L_m
                    sp_s_m.append(sp_i_m)
                for idx in range(0, dimL):
                    col_i_L_m = L_m[:, idx]
                    sp_i_L_m = mu_sp_m - sqrt_term_m * col_i_L_m
                    sp_s_m.append(sp_i_L_m)

                # pass each point through measurement model
                y_kp1_s_m = []

                mu_y_kp1_m = np.zeros((num_meas,))
                for jdx, sp_m in enumerate(sp_s_m):
                    y_kp1_m = H @ sp_m
                    if jdx == 0:
                        mu_y_kp1_m += w_0_m * y_kp1_m
                    else:
                        mu_y_kp1_m += w_j_m * y_kp1_m
                    y_kp1_s_m.append(y_kp1_m)

                # various aposteriori covariances
                sigma_yy = np.zeros((num_meas, num_meas))
                sigma_xy = np.zeros((num_states, num_meas))
                for kdx, sp in enumerate(sp_s_m):
                    diff_x_m = sp - x_kp1
                    diff_y_m = y_kp1_s_m[kdx] - mu_y_kp1_m
                    if kdx == 0:
                        sigma_yy += w_0_c * np.outer(diff_y_m, diff_y_m.T)
                        sigma_xy += w_0_c * np.outer(diff_x_m, diff_y_m.T)
                    else:
                        sigma_yy += w_j_c * np.outer(diff_y_m, diff_y_m.T)
                        sigma_xy += w_j_c * np.outer(diff_x_m, diff_y_m.T)

                # Kalman gain
                sigma_yy += R
                K_kp1 = np.matmul(sigma_xy, np.linalg.inv(sigma_yy))

                # Calculate Residual
                res_kp1 = z_kp1 - mu_y_kp1_m

                x_op_prev = x_op.copy()

                # Update State
                x_op = x_op + K_kp1 @ res_kp1

                # Update Covariance
                P_op = P_op - K_kp1 @ sigma_yy @ K_kp1.T

                current_difference = np.linalg.norm(x_op - x_op_prev)

            # Transfer states and covariance from kp1 to k
            P_k = P_op.copy()
            x_op[12:] = normalize_quat(x_op[12:])
            x_k = x_op.copy()

            x_p_k = x_k[0:3]
            x_p1_k = x_k[9:12]
            x_q_k = x_k[12:16]
            rotation_errors.append(quat_angle_diff(x_q_k, q_true[i, :]))
            Le, We, De = get_dimensions(x_p1_k, x_p_k, x_q_k)
            bbox3_dimensions.append([Le, We, De])

            # smooth out covariance off diagonals
            P_k = 0.5 * P_k + 0.5 * P_k.T

            z_q_k_1_previous = z_q_k_1.copy()
            if not ransac_error:
                z_q_k_2_previous = z_q_k_2.copy()

        z_s.append(z_kp1)
        z_pcas.append(np.hstack([z_p_k_1, z_omega_k, associatedBbox_1[:, 0], z_q_k_1]))
        if not ransac_error:
            z_rans.append(np.hstack([z_p_k_2, z_omega_k, associatedBbox_2[:, 0], z_q_k_2]))
        else:
            z_rans.append(np.hstack([np.zeros_like(z_p_k_1), np.zeros_like(z_omega_k), np.zeros_like(associatedBbox_1[:, 0]), np.zeros_like(z_q_k_1)]))

        # Append for analysis
        P_s.append(P_k)
        x_s.append(x_k)
        estimated_pos.append(x_k[:3])

        # Append data for master file
        file_names.append(pickle_file)
        if i == 0:
            ransac_errors.append(0)
            pca_errors.append(0)
            prediction_errors.append(0)
        else:
            ransac_errors.append(ransac_true_diff)
            pca_errors.append(pca_true_diff)
            prediction_errors.append(pred_true_diff)
        number_of_pointss.append(num_points)
        if i == 0:
            points_diffs.append(0)
        else:
            points_diffs.append(num_points - number_of_pointss[i - 1])
        z_spreads.append(max(Z_i) - min(Z_i))
        x_spreads.append(max(X_i) - min(X_i))
        y_spreads.append((max(Y_i) - min(Y_i)))
        if i == 0:
            pca_prev_diffs.append(0)
            ransac_prev_diffs.append(0)
            ransac_pred_diffs.append(0)
            pca_pred_diffs.append(0)
            ransac_pca_diffs.append(0)             
        else:
            pca_prev_diffs.append(pca_prev_diff)
            ransac_prev_diffs.append(ransac_prev_diff)
            ransac_pca_diffs.append(ransac_pca_diff)
            ransac_pred_diffs.append(ransac_pred_diff)
            pca_pred_diffs.append(pca_pred_diff)
        if i == 0:
            x_spread_diffs.append(0)
            y_spread_diffs.append(0)
            z_spread_diffs.append(0)
        else:
            x_spread_diffs.append(x_spreads[i] - x_spreads[i-1])
            y_spread_diffs.append(y_spreads[i] - y_spreads[i - 1])
            z_spread_diffs.append(z_spreads[i] - z_spreads[i - 1])
        if (not RC_flag) and i > configs['start'] and RC:
            RC_flag = True
            q_true = recalibrate_true_orientation(q_true, z_q_k, i)

    # Create final dataframe
    master_file['file_name'] = file_names
    master_file['ransac_error'] = ransac_errors
    master_file['pca_error'] = pca_errors
    master_file['prediction_error'] = prediction_errors
    master_file['perfect_metric_choice'] = perfect_metric_choices
    master_file['short_metric_choice'] = short_metric_choices
    master_file['number_of_points'] = number_of_pointss
    master_file['points_diff'] = points_diffs
    master_file['z_spread'] = z_spreads
    master_file['x_spread'] = x_spreads
    master_file['y_spread'] = y_spreads
    master_file['pca_prev_diff'] = pca_prev_diffs
    master_file['ransac_prev_diff'] = ransac_prev_diffs
    master_file['x_spread_diff'] = x_spread_diffs
    master_file['y_spread_diff'] = y_spread_diffs
    master_file['z_spread_diff'] = z_spread_diffs
    master_file['ransac_pca_diff'] = ransac_pca_diffs
    master_file['ransac_pred_diff'] = ransac_pred_diffs
    master_file['pca_pred_diff'] = pca_pred_diffs

    master_file.to_csv('full_results/results_of_' + pickle_file.split('.')[0] + '.csv', sep=',', header=True, index=False)

    ######
    # box assigment experiment
    #####
    assignment_results = pd.DataFrame(metric_boxes)
    assignment_results.to_csv('assignment_results/' + configs['assignment_results_file_name'] + pickle_file.split('.')[0] + '.csv', sep=',', header=True, index=False)

    ##############
    # Plot relevant figures
    ############


    x_s = np.array(x_s)
    x_s = x_s[1:, :]

    #####################
    # errors over last 'error_start' seconds
    ####################

    # position
    start_time_2 = -int(configs['error_calc_start'] / dt)
    rmse_px = np.sqrt(np.mean((x_s[start_time_2:, 0] - debris_pos[start_time_2:nframes, 0]) ** 2))
    rmse_py = np.sqrt(np.mean((x_s[start_time_2:, 1] - debris_pos[start_time_2:nframes, 1]) ** 2))
    rmse_pz = np.sqrt(np.mean((x_s[start_time_2:, 2] - debris_pos[start_time_2:nframes, 2]) ** 2))
    me_px = np.mean(x_s[start_time_2:, 0] - debris_pos[start_time_2:nframes, 0])
    me_py = np.mean(x_s[start_time_2:, 1] - debris_pos[start_time_2:nframes, 1])
    me_pz = np.mean(x_s[start_time_2:, 2] - debris_pos[start_time_2:nframes, 2])

    # angular velocity
    rmse_omx = np.sqrt(np.mean((x_s[start_time_2:, 6] - omega_true[0]) ** 2))
    rmse_omy = np.sqrt(np.mean((x_s[start_time_2:, 7] - omega_true[1]) ** 2))
    rmse_omz = np.sqrt(np.mean((x_s[start_time_2:, 8] - omega_true[2]) ** 2))
    me_omx = np.mean((x_s[start_time_2:, 6] - omega_true[0]))
    me_omy = np.mean((x_s[start_time_2:, 7] - omega_true[1]))
    me_omz = np.mean((x_s[start_time_2:, 8] - omega_true[2]))

    # linear velocity
    rmse_vdx = np.sqrt(np.mean((x_s[start_time_2:, 3] - debris_vel[start_time_2:nframes, 0]) ** 2))
    rmse_vdy = np.sqrt(np.mean((x_s[start_time_2:, 4] - debris_vel[start_time_2:nframes, 1]) ** 2))
    rmse_vdz = np.sqrt(np.mean((x_s[start_time_2:, 5] - debris_vel[start_time_2:nframes, 2]) ** 2))
    me_vdx = np.mean(x_s[start_time_2:, 3] - debris_vel[start_time_2:nframes, 0])
    me_vdy = np.mean(x_s[start_time_2:, 4] - debris_vel[start_time_2:nframes, 1])
    me_vdz = np.mean(x_s[start_time_2:, 5] - debris_vel[start_time_2:nframes, 2])

    # orientation rmse
    rmse_q = np.sqrt(np.mean(np.rad2deg(rotation_errors[start_time_2:nframes]) ** 2))

    # bias errors
    b_start = int(t_start / dt)
    b_end = int((t_start + t_interval) / dt)
    rmse_x_before = np.sqrt(np.mean((x_s[b_start:b_end, 0] - debris_pos[b_start:b_end, 0]) ** 2))
    rmse_y_before = np.sqrt(np.mean((x_s[b_start:b_end, 1] - debris_pos[b_start:b_end, 1]) ** 2))
    rmse_z_before = np.sqrt(np.mean((x_s[b_start:b_end, 2] - debris_pos[b_start:b_end, 2]) ** 2))
    me_x_before = np.mean(x_s[b_start:b_end, 0] - debris_pos[b_start:b_end, 0])
    me_y_before = np.mean(x_s[b_start:b_end, 1] - debris_pos[b_start:b_end, 1])
    me_z_before = np.mean(x_s[b_start:b_end, 2] - debris_pos[b_start:b_end, 2])

    rmse_x_after = np.sqrt(np.mean((x_s[b_end:, 0] - debris_pos[b_end:nframes, 0]) ** 2))
    rmse_y_after = np.sqrt(np.mean((x_s[b_end:, 1] - debris_pos[b_end:nframes, 1]) ** 2))
    rmse_z_after = np.sqrt(np.mean((x_s[b_end:, 2] - debris_pos[b_end:nframes, 2]) ** 2))
    me_x_after = np.mean(x_s[b_end:, 0] - debris_pos[b_end:nframes, 0])
    me_y_after = np.mean(x_s[b_end:, 1] - debris_pos[b_end:nframes, 1])
    me_z_after = np.mean(x_s[b_end:, 2] - debris_pos[b_end:nframes, 2])

    if 'p' in configs['options']:
        logger.info("Position RMSE: " + str([rmse_px, rmse_py, rmse_pz]))
        logger.info("Position ME: " + str([me_px, me_py, me_pz]))
        logger.info("Ang. Vel. RMSE: " + str([rmse_omx, rmse_omy, rmse_omz]))
        logger.info("Ang. Vel. ME: " + str([me_omx, me_omy, me_omz]))
        logger.info("Lin. Vel. RMSE: " + str([rmse_vdx, rmse_vdy, rmse_vdz]))
        logger.info("Lin. ME: " + str([me_vdx, me_vdy, me_vdz]))
        logger.info("Bias before RMSE: " + str([rmse_x_before, rmse_y_before, rmse_z_before]))
        logger.info("Bias after RMSE: " + str([rmse_x_after, rmse_y_after, rmse_z_after]))
        logger.info("Bias before ME: " + str([me_x_before, me_y_before, me_z_before]))
        logger.info("Bias before ME: " + str([me_x_after, me_y_after, me_z_after]))
        logger.info("Orientation RMSE: " + str(rmse_q))

    results = [rmse_px, rmse_py, rmse_pz, rmse_omx, rmse_omy, rmse_omz, rmse_vdx, rmse_vdy, rmse_vdz, rmse_x_before, rmse_y_before, rmse_z_before,
               rmse_x_after, rmse_y_after, rmse_z_after, rmse_q]

    return results
