import copy
import os.path
from pathlib import Path

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
    params = []
    constant = 0.0
    success = True
    try:
        params, params_covariance = curve_fit(sum_of_sinusoids, t, y, p0=initial_guess)
        constant = max(sum_of_sinusoids(t, *params))
    except RuntimeError:
        success = False
    # print(constant)

    return params, constant, success


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


def add_state_measurement_columns(record, i, x_k, z_p_k, z_omega_k, z_p1_k, z_q_k,
                                  z_q_k_1, z_q_k_2, z_p_k_1, z_p_k_2,
                                  associatedBbox_1, associatedBbox_2,
                                  omega_LLS, omega_L_to_B, omega_los_L,
                                  without_correction):
    """Add state-estimate and measurement columns to a per-frame record dict."""

    def add_vec(prefix, values, names):
        if values is None:
            values = [np.nan] * len(names)
        values = np.asarray(values).reshape(-1)
        for idx, name in enumerate(names):
            record[f"{prefix}_{name}"] = float(values[idx]) if idx < len(values) else np.nan

    def add_quat(prefix, quat):
        if quat is None:
            add_vec(prefix, None, ["w", "x", "y", "z"])
        else:
            add_vec(prefix, normalize_quat(np.asarray(quat).copy()), ["w", "x", "y", "z"])

    # state estimate: [p, v, omega, p1, q]
    add_vec("state_est", x_k[0:3], ["x", "y", "z"])
    add_vec("state_est", x_k[3:6], ["vx", "vy", "vz"])
    add_vec("state_est", x_k[6:9], ["wx", "wy", "wz"])
    add_vec("state_est_p1", x_k[9:12], ["x", "y", "z"])
    add_quat("state_est_q", x_k[12:16])

    # main measurements
    add_vec("meas_p", z_p_k, ["x", "y", "z"])
    add_vec("meas_w", z_omega_k, ["x", "y", "z"])
    add_vec("meas_p1", z_p1_k, ["x", "y", "z"])
    add_quat("meas_q", z_q_k)

    raw_p = without_correction[i] if i < len(without_correction) else [np.nan, np.nan, np.nan]
    add_vec("meas_p_raw", raw_p, ["x", "y", "z"])

    # PCA-specific measurements
    add_vec("meas_pca_p", z_p_k_1, ["x", "y", "z"])
    p1_pca = associatedBbox_1[:, 0] if associatedBbox_1 is not None else None
    add_vec("meas_pca_p1", p1_pca, ["x", "y", "z"])
    add_quat("meas_pca_q", z_q_k_1)

    # RANSAC-specific measurements
    add_vec("meas_ransac_p", z_p_k_2, ["x", "y", "z"])
    p1_ransac = associatedBbox_2[:, 0] if associatedBbox_2 is not None else None
    add_vec("meas_ransac_p1", p1_ransac, ["x", "y", "z"])
    add_quat("meas_ransac_q", z_q_k_2)

    # angular velocity diagnostics
    add_vec("omega_lls", omega_LLS, ["x", "y", "z"])
    add_vec("omega_l_to_b", omega_L_to_B, ["x", "y", "z"])
    add_vec("omega_kabsch", omega_los_L, ["x", "y", "z"])

    return record


def add_truth_columns(record, i, debris_pos, debris_vel, omega_true, q_true):
    """Add truth columns to a per-frame record dict."""

    def add_vec(prefix, values, names):
        if values is None:
            values = [np.nan] * len(names)
        values = np.asarray(values).reshape(-1)
        for idx, name in enumerate(names):
            record[f"{prefix}_{name}"] = float(values[idx]) if idx < len(values) else np.nan

    p_true_i = debris_pos[i] if i < len(debris_pos) else None
    v_true_i = debris_vel[i] if i < len(debris_vel) else None

    if omega_true is None:
        omega_true_i = None
    else:
        omega_true_arr = np.asarray(omega_true)
        omega_true_i = omega_true_arr[i] if omega_true_arr.ndim > 1 and i < len(omega_true_arr) else omega_true_arr.reshape(-1)

    q_true_i = q_true[i] if i < len(q_true) else None
    if q_true_i is not None:
        q_true_i = normalize_quat(np.asarray(q_true_i).copy())

    add_vec("truth_p", p_true_i, ["x", "y", "z"])
    add_vec("truth_v", v_true_i, ["x", "y", "z"])
    add_vec("truth_w", omega_true_i, ["x", "y", "z"])
    add_vec("truth_q", q_true_i, ["w", "x", "y", "z"])

    return record


def add_covariance_and_geometry_columns(record, P_k, debris_pos_i, Le, We, De):
    """Add covariance diagonal, true range, and bbox dimensions to a per-frame record dict."""

    cov_names = [
        "x", "y", "z",
        "vx", "vy", "vz",
        "wx", "wy", "wz",
        "p1_x", "p1_y", "p1_z",
        "qw", "qx", "qy", "qz",
    ]

    if P_k is None:
        diagP = np.full(len(cov_names), np.nan)
    else:
        try:
            diagP = np.diag(np.asarray(P_k)).reshape(-1)
        except Exception:
            diagP = np.full(len(cov_names), np.nan)

    for idx, name in enumerate(cov_names):
        record[f"cov_{name}"] = float(diagP[idx]) if idx < len(diagP) else np.nan

    if debris_pos_i is not None:
        debris_pos_i = np.asarray(debris_pos_i).reshape(-1)
        record["true_range"] = float(np.linalg.norm(debris_pos_i[:3])) if debris_pos_i.size >= 3 else np.nan
    else:
        record["true_range"] = np.nan

    record["Le"] = float(Le) if Le is not None else np.nan
    record["We"] = float(We) if We is not None else np.nan
    record["De"] = float(De) if De is not None else np.nan

    return record

METHOD_PCA = 1
METHOD_RANSAC = 2
METHOD_PREDICTION = 3

METHOD_CODE_TO_NAME = {
    METHOD_PCA: "pca",
    METHOD_RANSAC: "ransac",
    METHOD_PREDICTION: "prediction",
}

METRIC_STAGE_NAMES = {
    0: "inactive_startup",
    1: "agree_ransac",
    2: "ortho_ransac",
    3: "eig_pca",
    4: "boresight_pca",
    5: "pred_fallback",
}

ORACLE_STATUS_NAMES = {
    0: "startup",
    1: "threshold_ransac",
    2: "threshold_pca",
    3: "threshold_pred",
    4: "best_available_fallback",
}

AGREEMENT_NAMES = {
    0: "none",
    1: "pca_ransac",
    2: "pca_pred",
    3: "ransac_pred",
    4: "all_three",
    5: "mixed",
}

TRUE_PASS_PATTERN_NAMES = {
    0: "none",
    1: "pca_only",
    2: "ransac_only",
    3: "pred_only",
    4: "pca_ransac",
    5: "pca_pred",
    6: "ransac_pred",
    7: "all_three",
}


def _classify_pairwise_agreement(*, agree_pca_ransac, agree_pca_pred, agree_ransac_pred):
    if agree_pca_ransac and agree_pca_pred and agree_ransac_pred:
        return 4, AGREEMENT_NAMES[4]
    pair_count = int(bool(agree_pca_ransac)) + int(bool(agree_pca_pred)) + int(bool(agree_ransac_pred))
    if pair_count == 0:
        return 0, AGREEMENT_NAMES[0]
    if pair_count == 1:
        if agree_pca_ransac:
            return 1, AGREEMENT_NAMES[1]
        if agree_pca_pred:
            return 2, AGREEMENT_NAMES[2]
        return 3, AGREEMENT_NAMES[3]
    return 5, AGREEMENT_NAMES[5]


def _classify_true_pass_pattern(*, pca_pass, ransac_pass, pred_pass):
    key = (int(bool(pca_pass)), int(bool(ransac_pass)), int(bool(pred_pass)))
    mapping = {
        (0, 0, 0): 0,
        (1, 0, 0): 1,
        (0, 1, 0): 2,
        (0, 0, 1): 3,
        (1, 1, 0): 4,
        (1, 0, 1): 5,
        (0, 1, 1): 6,
        (1, 1, 1): 7,
    }
    code = mapping[key]
    return code, TRUE_PASS_PATTERN_NAMES[code]


def select_measurement_method(*, i, start_index, ransac_pred_diff, pca_pred_diff, ransac_pca_diff,
                              ransac_vecs_volume, pca_ratio, pca_angle, short_metric_thresh,
                              orthonormal_thresh, eig_thresh, boresight_thresh):
    metric_active = bool(i > start_index)

    agree_ransac_pred = bool(ransac_pred_diff < short_metric_thresh)
    agree_pca_pred = bool(pca_pred_diff < short_metric_thresh)
    agree_pca_ransac = bool(ransac_pca_diff < short_metric_thresh)
    agree_all_three = agree_ransac_pred and agree_pca_pred and agree_pca_ransac
    agreement_code, agreement_name = _classify_pairwise_agreement(
        agree_pca_ransac=agree_pca_ransac,
        agree_pca_pred=agree_pca_pred,
        agree_ransac_pred=agree_ransac_pred,
    )

    flag_ransac_ortho_pass = bool(ransac_vecs_volume > orthonormal_thresh)
    flag_pca_eig_pass = bool(pca_ratio > eig_thresh)
    flag_pca_boresight_pass = bool(pca_angle < boresight_thresh)

    if i == 0:
        choice_code = METHOD_RANSAC
        stage_code = 0
    elif metric_active:
        if agree_pca_ransac:
            choice_code = METHOD_RANSAC
            stage_code = 1
        elif flag_ransac_ortho_pass:
            choice_code = METHOD_RANSAC
            stage_code = 2
        elif flag_pca_eig_pass:
            choice_code = METHOD_PCA
            stage_code = 3
        elif flag_pca_boresight_pass:
            choice_code = METHOD_PCA
            stage_code = 4
        else:
            choice_code = METHOD_PREDICTION
            stage_code = 5
    else:
        if ransac_pred_diff > pca_pred_diff:
            choice_code = METHOD_PCA
        else:
            choice_code = METHOD_RANSAC
        stage_code = 0

    choice_name = METHOD_CODE_TO_NAME[choice_code]
    stage_name = METRIC_STAGE_NAMES[stage_code]

    return {
        "metric_active": metric_active,
        "choice_code": choice_code,
        "choice_name": choice_name,
        "stage_code": stage_code,
        "stage_name": stage_name,
        "short_metric_choice": f"{choice_name} {stage_code}",
        "agree_pca_ransac": agree_pca_ransac,
        "agree_pca_pred": agree_pca_pred,
        "agree_ransac_pred": agree_ransac_pred,
        "agree_all_three": agree_all_three,
        "agreement_code": agreement_code,
        "agreement_name": agreement_name,
        "flag_ransac_ortho_pass": flag_ransac_ortho_pass,
        "flag_pca_eig_pass": flag_pca_eig_pass,
        "flag_pca_boresight_pass": flag_pca_boresight_pass,
    }


def select_oracle_method(*, i, ransac_true_diff, pca_true_diff, pred_true_diff, true_orientation_difference):
    if i == 0:
        choice_code = METHOD_RANSAC
        status_code = 0
        pca_pass = False
        ransac_pass = False
        pred_pass = False
    else:
        ransac_pass = bool(ransac_true_diff < true_orientation_difference)
        pca_pass = bool(pca_true_diff < true_orientation_difference)
        pred_pass = bool(pred_true_diff < true_orientation_difference)

        if ransac_pass:
            choice_code = METHOD_RANSAC
            status_code = 1
        elif pca_pass:
            choice_code = METHOD_PCA
            status_code = 2
        elif pred_pass:
            choice_code = METHOD_PREDICTION
            status_code = 3
        else:
            values = [pca_true_diff, ransac_true_diff, pred_true_diff]
            choice_code = int(np.argmin(values)) + 1
            status_code = 4

    choice_name = METHOD_CODE_TO_NAME[choice_code]
    status_name = ORACLE_STATUS_NAMES[status_code]
    true_pass_pattern_code, true_pass_pattern_name = _classify_true_pass_pattern(
        pca_pass=pca_pass,
        ransac_pass=ransac_pass,
        pred_pass=pred_pass,
    )

    if status_code == 4:
        perfect_metric_choice = "all wrong"
    elif status_code == 0:
        perfect_metric_choice = "first"
    else:
        perfect_metric_choice = choice_name

    return {
        "choice_code": choice_code,
        "choice_name": choice_name,
        "status_code": status_code,
        "status_name": status_name,
        "perfect_metric_choice": perfect_metric_choice,
        "pass_pca": pca_pass,
        "pass_ransac": ransac_pass,
        "pass_pred": pred_pass,
        "pass_all_three": bool(pca_pass and ransac_pass and pred_pass),
        "true_pass_pattern_code": true_pass_pattern_code,
        "true_pass_pattern_name": true_pass_pattern_name,
    }


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

def eigenvalue_metric(evals):
    evals.sort()
    a = evals[0]
    b = evals[1]
    c = evals[2]
    d = c-a
    if abs(d)<1e-8:
        return 0.0
    else:
        return (b-a)*(c-b)/(d*d)

def boresight_metric(Rot_L_to_B, evecs):
    z_axis = Rot_L_to_B.T @ np.array([0, 0, 1])
    
    angles = []
    for evec in evecs:
        cos_angle = np.abs(np.dot(z_axis, evec) / (np.linalg.norm(z_axis) * np.linalg.norm(evec)))
        cos_angle = np.clip(cos_angle, 0, 1)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return np.min(angles)

def _format_float_for_tag(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_combo_name(ransac_pca_threshold, orthonormal_thresh, eig_thresh):
    return (
        f"rpca_{_format_float_for_tag(ransac_pca_threshold)}"
        f"__ortho_{_format_float_for_tag(orthonormal_thresh)}"
        f"__eig_{_format_float_for_tag(eig_thresh)}"
    )


def build_output_stem(task, configs):
    assignment_prefix = configs.get('assignment_results_file_name', 'ass_res_')
    pickle_stem = Path(task['pickle_file']).stem
    return f"{assignment_prefix}{task['geometry_name']}__{task['combo_name']}__run_{task['run_number_for_combo']:03d}__{pickle_stem}"


def run(task, configs, logger):

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # initialize debris position, velocity and orientation
    O_B = np.array([0, 0, 0])
    O_L = np.array([0, 0, 0])

    pickle_path = task['pickle_path']
    pickle_file = task['pickle_file']
    output_stem = build_output_stem(task, configs)

    with open(pickle_path, 'rb') as sim_data:
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
    frame_good = np.asarray(data['use_frame'], dtype=bool)
    frame_good[0] = True  # always start on a non-fully-occluded frame
    partial_occlusion = np.asarray(data.get('partial', np.zeros(len(frame_good), dtype=bool)), dtype=bool)
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
    ransac_orthos = []
    pca_ratios = []
    pca_angles = []

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
    frame_records = []
    short_metric_choices = []
    perfect_metric_choices = []
    assignment_records = []

    for i in range(nframes): 
        PL=((Rot_L_to_B[i].T @ (PBs[i]).T).T)
        # find bounding box from points
        X_i = PL[:,0]
        Y_i = PL[:,1]
        Z_i = PL[:,2]
        z_pi_k_1, z_p_k_1, R_1, evals = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
        z_q_k_1 = rotm2quat(R_1)
        evecs = R_1.copy()
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
    prev_box_B = None
    for i in range(nframes):
        current_frame_good = bool(frame_good[i])
        current_partial = bool(partial_occlusion[i])
        Le = np.nan
        We = np.nan
        De = np.nan

        if rank == 0:
            pass
            #logger.info(f"Iteration {i} of {nframes}")
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

        curr_t = i * dt
        t_start = configs['start_time']  # when the first bias calculation should be initiated
        t_interval = configs['interval']  # how many seconds of data should be collected each time
        bias_removal_success = False
        PLs.append((Rot_L_to_B[i].T @ (PBs[i]).T).T)
        XLs.append(PLs[i][:, 0])
        YLs.append(PLs[i][:, 1])
        ZLs.append(PLs[i][:, 2])
        X_i = XLs[i]
        Y_i = YLs[i]
        Z_i = ZLs[i]
        if current_frame_good:


            num_points = len(Z_i)

            z_pi_k_1, z_p_k_1, R_1, evals = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
            evecs = R_1.copy()
            if i == 0:
                q_kp1 = rotm2quat(R_1)
            z_pi_k_2, z_p_k_2, R_1_2, normal_vecs, ranking, num_planes = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, q_kp1, True, False)

            if R_1_2.size == 0:
                ransac_error = True
            else:
                ransac_error = False

            ransac_vecs_volume = 0.0
            if not ransac_error and normal_vecs.shape[0] >= 3:
                n0 = normal_vecs[0] / np.linalg.norm(normal_vecs[0])
                n1 = normal_vecs[1] / np.linalg.norm(normal_vecs[1])
                n2 = normal_vecs[2] / np.linalg.norm(normal_vecs[2])
                ransac_vecs_volume = abs(np.linalg.det(np.array([n0, n1, n2])))
            ransac_orthos.append(ransac_vecs_volume)

            original_pos_meas.append(z_p_k_1)
            centroids_inB.append(Rot_L_to_B[i] @ z_p_k_1)
            true_pos_inB.append(Rot_L_to_B[i] @ debris_pos[i, :])

            if curr_t >= (t_start + t_interval):
                if (curr_t + t_start) % t_interval == 0 and done == 0:
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

                    thresh = configs['threshold']
                    num_sin = configs['number_of_sinusoids']
                    skip = configs['number_of_skips']
                    params_z, constant_z, bias_removal_success = remove_bias(interval_time, dt, z[:, 2], estimated[:, 2], num_sin, thresh, skip, true[:, 2], params_z)
                    parameters = [params_x, params_y, params_z]
                    constants = [0, 0, constant_z]
                    done = 1

            if i == 0:
                z_q_k_1 = rotm2quat(R_1)
                if not ransac_error:
                    z_q_k_2, _, _ = rotation_association(z_q_k_1, R_1_2)
                z_q_k = z_q_k_1.copy()
                z_pi_k = z_pi_k_1.copy()
                z_p_k = z_p_k_1.copy()
            else:
                z_q_k_1, _, error = rotation_association(q_kp1, R_1)
                if not ransac_error:
                    z_q_k_2, bad_attitude_measurement_flag_2, error_2 = rotation_association(q_kp1, R_1_2)

            if i > 0:
                LWD = 2 * quat2rotm(q_kp1).T @ (p_kp1 - p1_kp1)
                L = LWD[0]
                W = LWD[1]
                D = LWD[2]
                predictedBbox = boundingbox.from_params(p_kp1, q_kp1, L, W, D)

            associatedBbox_1, Lm, Wm, Dm = boundingbox.associated(z_q_k_1, z_pi_k_1, z_p_k_1, R_1)
            z_p1_k_1 = associatedBbox_1[:, 0]
            if not ransac_error:
                associatedBbox_2, Lm_2, Wm_2, Dm_2 = boundingbox.associated(z_q_k_2, z_pi_k_2, z_p_k_2, R_1_2)
                z_p1_k_2 = associatedBbox_2[:, 0]
            else:
                associatedBbox_2 = None
                z_p1_k_2 = None
                Lm_2 = Wm_2 = Dm_2 = np.nan

            if i == 0:
                associatedBbox = associatedBbox_1.copy()
                z_p1_k = associatedBbox_1[:, 0]
                z_q_k_1_previous = z_q_k_1.copy()
                if not ransac_error:
                    z_q_k_2_previous = z_q_k_2.copy()

            ransac_pred_diff = np.nan
            pca_pred_diff = np.nan
            ransac_pca_diff = np.nan
            ransac_prev_diff = np.nan
            pca_prev_diff = np.nan
            ransac_true_diff = np.nan
            pca_true_diff = np.nan
            pred_true_diff = np.nan
            pca_ratio = 0.0
            pca_angle = 0.0

            if i > 0:
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
                pca_ratio = eigenvalue_metric(evals)
                pca_ratios.append(pca_ratio)
                pca_angle = np.rad2deg(boresight_metric(Rot_L_to_B[i], evecs))
                pca_angles.append(pca_angle)
            if i == 0:
                pca_ratios.append(0.0)
                pca_angles.append(0.0)

            metric_result = select_measurement_method(
                i=i,
                start_index=configs['start'],
                ransac_pred_diff=ransac_pred_diff,
                pca_pred_diff=pca_pred_diff,
                ransac_pca_diff=ransac_pca_diff,
                ransac_vecs_volume=ransac_vecs_volume,
                pca_ratio=pca_ratios[-1] if len(pca_ratios) > 0 else 0.0,
                pca_angle=pca_angles[-1] if len(pca_angles) > 0 else 0.0,
                short_metric_thresh=configs['short_metric_thresh'],
                orthonormal_thresh=configs['orthonormal_thresh'],
                eig_thresh=configs['eig_thresh'],
                boresight_thresh=configs['boresight_thresh'],
            )
            use_measurement = metric_result['choice_code']
            short_metric_choice = metric_result['short_metric_choice']
            short_metric_choices.append(short_metric_choice)

            oracle_result = select_oracle_method(
                i=i,
                ransac_true_diff=ransac_true_diff,
                pca_true_diff=pca_true_diff,
                pred_true_diff=pred_true_diff,
                true_orientation_difference=configs['true_orientation_difference'],
            )
            ideal_measurement = oracle_result['choice_code']
            perfect_metric_choice = oracle_result['perfect_metric_choice']
            perfect_metric_choices.append(perfect_metric_choice)

            metric_matches_oracle = bool(metric_result['choice_code'] == oracle_result['choice_code'])
            oracle_override_enabled = bool(configs['use_perfect_metric'])
            metric_overridden_by_oracle = bool(oracle_override_enabled and not metric_matches_oracle)

            final_choice_code = oracle_result['choice_code'] if oracle_override_enabled else metric_result['choice_code']
            final_choice_name = METHOD_CODE_TO_NAME[final_choice_code]
            if configs['use_perfect_metric']:
                use_measurement = ideal_measurement
            if use_measurement == 2:
                try:
                    z_q_k = z_q_k_2.copy()
                    z_pi_k = z_pi_k_2.copy()
                    z_p_k = z_p_k_2.copy()
                    z_p1_k = associatedBbox_2[:, 0]
                    associatedBbox = associatedBbox_2.copy()
                    adapt = False
                    choice = 'ransac'
                except (UnboundLocalError, AttributeError):
                    z_q_k = z_q_k_1.copy()
                    z_pi_k = z_pi_k_1.copy()
                    z_p_k = z_p_k_1.copy()
                    z_p1_k = associatedBbox_1[:, 0]
                    associatedBbox = associatedBbox_1.copy()
                    adapt = False
                    choice = 'pca'
            elif use_measurement == 1:
                z_q_k = z_q_k_1.copy()
                z_pi_k = z_pi_k_1.copy()
                z_p_k = z_p_k_1.copy()
                z_p1_k = associatedBbox_1[:, 0]
                associatedBbox = associatedBbox_1.copy()
                adapt = False
                choice = 'pca'
            else:
                associatedBbox = predictedBbox.copy()
                z_p_k = z_p_k_1.copy()
                z_p1_k = associatedBbox[:, 0]
                adapt = True
                choice = 'prediction'

            without_correction.append(z_p_k)
            bbox1_dimensions.append([Lm, Wm, Dm])
            bbox2_dimensions.append([Lm_2, Wm_2, Dm_2])
            if curr_t >= (t_start + t_interval) and bias_removal_success:
                z_p_k_z = correct_bias(z_p_k, i, dt, parameters, constants, Rot_L_to_B[i], Rot_B_to_L[i])
                z_p_k = z_p_k_z

            omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)
            B_v_BL = np.cross(-Rot_L_to_B[i] @ omega_L_to_B, Rot_L_to_B[i] @ z_p_k)

            omega_LLS = np.zeros(3)
            omega_los_L = np.zeros(3)

            if i > 0:
                omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i] @ z_p_k, Rot_L_to_B[i] @ v_k, VBs[i], B_v_BL)
                omega_LLS = Rot_B_to_L[i] @ omega_LLS_B

            if i == 0 or prev_box_B is None:
                prev_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
                prev_box_B = (Rot_L_to_B[i] @ prev_box_L.T).T
            else:
                cur_box_L = np.transpose(copy.deepcopy(associatedBbox_1))
                cur_box_B = (Rot_L_to_B[i] @ cur_box_L.T).T
                omega_los_B = estimate_kabsch(prev_box_B, cur_box_B, dt)
                prev_box_B = cur_box_B.copy()

                omega_kabsch_b_box[i % n_moving_average] = omega_los_B
                if i < n_moving_average:
                    omega_los_B_averaged = np.mean(omega_kabsch_b_box[0:i + 1], axis=0)
                else:
                    omega_los_B_averaged = np.mean(omega_kabsch_b_box, axis=0)
                omega_los_L = Rot_B_to_L[i] @ omega_los_B_averaged

            if i == 0:
                z_omega_k = omega_0
            elif i <= settling_time:
                z_omega_k = omega_LLS + omega_L_to_B
            else:
                z_omega_k = omega_LLS + omega_L_to_B + omega_los_L

            if adapt:
                z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k])
                H = H2
                R = R2
            else:
                z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k])
                H = H1
                R = R1

            if i == 0:
                x_k = np.hstack([z_p_k, vT_0, z_omega_k, z_p1_k, z_q_k_1])
                P_k = P_0.copy()
                Le, We, De = get_dimensions(x_k[9:12], x_k[0:3], x_k[12:16])

            num_meas = len(z_kp1)

            if i > 0:
                x_op = x_kp1.copy()
                P_op = P_kp1.copy()
                current_difference = 1
                while current_difference > tolerance:
                    mu_sp_m = x_op.copy()
                    sigma_zz_m = P_op.copy()
                    try:
                        L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)
                    except np.linalg.LinAlgError:
                        np.fill_diagonal(sigma_zz_m, sigma_zz_m.diagonal() + epsilon)
                        L_m = scipy.linalg.cholesky(sigma_zz_m, lower=True)

                    sp_0_m = mu_sp_m
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

                    y_kp1_s_m = []
                    mu_y_kp1_m = np.zeros((num_meas,))
                    for jdx, sp_m in enumerate(sp_s_m):
                        y_kp1_m = H @ sp_m
                        if jdx == 0:
                            mu_y_kp1_m += w_0_m * y_kp1_m
                        else:
                            mu_y_kp1_m += w_j_m * y_kp1_m
                        y_kp1_s_m.append(y_kp1_m)

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

                    sigma_yy += R
                    K_kp1 = np.matmul(sigma_xy, np.linalg.inv(sigma_yy))
                    res_kp1 = z_kp1 - mu_y_kp1_m
                    x_op_prev = x_op.copy()
                    x_op = x_op + K_kp1 @ res_kp1
                    P_op = P_op - K_kp1 @ sigma_yy @ K_kp1.T
                    current_difference = np.linalg.norm(x_op - x_op_prev)

                P_k = P_op.copy()
                x_op[12:] = normalize_quat(x_op[12:])
                x_k = x_op.copy()
                x_p_k = x_k[0:3]
                x_p1_k = x_k[9:12]
                x_q_k = x_k[12:16]
                rotation_errors.append(np.rad2deg(quat_angle_diff(x_q_k, q_true[i, :])))
                Le, We, De = get_dimensions(x_p1_k, x_p_k, x_q_k)
                bbox3_dimensions.append([Le, We, De])
                P_k = 0.5 * P_k + 0.5 * P_k.T
                z_q_k_1_previous = z_q_k_1.copy()
                if not ransac_error:
                    z_q_k_2_previous = z_q_k_2.copy()

            estimated_pos.append(x_k[:3])
            z_s.append(z_kp1)
            z_pcas.append(np.hstack([z_p_k_1, z_omega_k, associatedBbox_1[:, 0], z_q_k_1]))
            if not ransac_error:
                z_rans.append(np.hstack([z_p_k_2, z_omega_k, associatedBbox_2[:, 0], z_q_k_2]))
            else:
                z_rans.append(np.hstack([np.zeros_like(z_p_k_1), np.zeros_like(z_omega_k), np.zeros_like(associatedBbox_1[:, 0]), np.zeros_like(z_q_k_1)]))
        else:
            num_points = 0
            ransac_error = True
            prev_box_B = None
            adapt = False
            z_kp1 = None
            z_pi_k_1 = None
            z_pi_k_2 = None
            z_p_k_1 = None
            z_p_k_2 = None
            z_q_k_1 = None
            z_q_k_2 = None
            z_q_k = None
            z_p_k = None
            z_p1_k = None
            associatedBbox_1 = None
            associatedBbox_2 = None
            z_p1_k_1 = None
            z_p1_k_2 = None
            omega_L_to_B = np.array([np.nan, np.nan, np.nan])
            omega_LLS = np.array([np.nan, np.nan, np.nan])
            omega_los_L = np.array([np.nan, np.nan, np.nan])
            z_omega_k = None
            ransac_pred_diff = np.nan
            pca_pred_diff = np.nan
            ransac_pca_diff = np.nan
            ransac_prev_diff = np.nan
            pca_prev_diff = np.nan
            ransac_true_diff = np.nan
            pca_true_diff = np.nan
            pred_true_diff = np.nan
            short_metric_choice = 'occluded'
            perfect_metric_choice = 'occluded'
            metric_result = {
                'metric_active': False,
                'choice_code': METHOD_PREDICTION,
                'choice_name': 'prediction',
                'stage_code': -1,
                'stage_name': 'occluded',
                'short_metric_choice': 'occluded',
                'agree_pca_ransac': False,
                'agree_pca_pred': False,
                'agree_ransac_pred': False,
                'agree_all_three': False,
                'agreement_code': -1,
                'agreement_name': 'occluded',
                'flag_ransac_ortho_pass': False,
                'flag_pca_eig_pass': False,
                'flag_pca_boresight_pass': False,
            }
            oracle_result = {
                'choice_code': METHOD_PREDICTION,
                'choice_name': 'prediction',
                'status_code': -1,
                'status_name': 'occluded',
                'perfect_metric_choice': 'occluded',
                'pass_pca': False,
                'pass_ransac': False,
                'pass_pred': False,
                'pass_all_three': False,
                'true_pass_pattern_code': -1,
                'true_pass_pattern_name': 'occluded',
            }
            metric_matches_oracle = False
            oracle_override_enabled = bool(configs['use_perfect_metric'])
            metric_overridden_by_oracle = False
            final_choice_code = METHOD_PREDICTION
            final_choice_name = 'prediction'
            ransac_orthos.append(np.nan)
            pca_ratios.append(np.nan)
            pca_angles.append(np.nan)
            without_correction.append([np.nan, np.nan, np.nan])
            bbox1_dimensions.append([np.nan, np.nan, np.nan])
            bbox2_dimensions.append([np.nan, np.nan, np.nan])

            if i == 0:
                x_k = x_0.copy()
                P_k = P_0.copy()
                x_q_k = x_k[12:16]
            else:
                x_k = x_kp1.copy()
                x_k[12:] = normalize_quat(x_k[12:])
                P_k = 0.5 * P_kp1 + 0.5 * P_kp1.T
                x_q_k = x_k[12:16]
                rotation_errors.append(np.rad2deg(quat_angle_diff(x_q_k, q_true[i, :])))
                Le, We, De = get_dimensions(x_k[9:12], x_k[0:3], x_q_k)
                bbox3_dimensions.append([Le, We, De])

            z_s.append(np.full(13, np.nan))
            z_pcas.append(np.full(13, np.nan))
            z_rans.append(np.full(13, np.nan))
        # Append for analysis
        P_s.append(P_k)
        x_s.append(x_k)


        # Append data for output file
        record = {
            'file_name': pickle_file,
            'geometry_name': task['geometry_name'],
            'combo_name': task['combo_name'],
            'combo_index': task['combo_index'],
            'run_number_for_combo': task['run_number_for_combo'],
            'ransac_pca_threshold': task['ransac_pca_threshold'],
            'orthonormal_thresh': task['orthonormal_thresh'],
            'eig_thresh': task['eig_thresh'],
            'frame': i,
            'frame_good': current_frame_good,
            'partial_occlusion': current_partial,
            'update_performed': bool(current_frame_good),
            'prediction_only': not current_frame_good,
            'ransac_error': 0 if i == 0 else ransac_true_diff,
            'pca_error': 0 if i == 0 else pca_true_diff,
            'prediction_error': 0 if i == 0 else pred_true_diff,
            'perfect_metric_choice': perfect_metric_choice,
            'short_metric_choice': short_metric_choice,
            'number_of_points': num_points,
            'pca_prev_diff': 0 if i == 0 else pca_prev_diff,
            'ransac_prev_diff': 0 if i == 0 else ransac_prev_diff,
            'ransac_pca_diff': 0 if i == 0 else ransac_pca_diff,
            'ransac_pred_diff': 0 if i == 0 else ransac_pred_diff,
            'pca_pred_diff': 0 if i == 0 else pca_pred_diff,
            'estimate_error': rotation_errors[-1] if len(rotation_errors) > 0 else np.nan,
            'ransac_orthogonality': ransac_orthos[-1] if len(ransac_orthos) > 0 else np.nan,
            'pca_ratio': pca_ratios[-1] if len(pca_ratios) > 0 else np.nan,
            'pca_angle': pca_angles[-1] if len(pca_angles) > 0 else np.nan,
            'metric_active': metric_result['metric_active'],
            'metric_choice_code': metric_result['choice_code'],
            'metric_choice_name': metric_result['choice_name'],
            'metric_stage_code': metric_result['stage_code'],
            'metric_stage_name': metric_result['stage_name'],
            'agreement_code': metric_result['agreement_code'],
            'agreement_name': metric_result['agreement_name'],
            'agree_pca_ransac': metric_result['agree_pca_ransac'],
            'agree_pca_pred': metric_result['agree_pca_pred'],
            'agree_ransac_pred': metric_result['agree_ransac_pred'],
            'agree_all_three': metric_result['agree_all_three'],
            'flag_ransac_ortho_pass': metric_result['flag_ransac_ortho_pass'],
            'flag_pca_eig_pass': metric_result['flag_pca_eig_pass'],
            'flag_pca_boresight_pass': metric_result['flag_pca_boresight_pass'],
            'oracle_choice_code': oracle_result['choice_code'],
            'oracle_choice_name': oracle_result['choice_name'],
            'oracle_status_code': oracle_result['status_code'],
            'oracle_status_name': oracle_result['status_name'],
            'oracle_pass_pca': oracle_result['pass_pca'],
            'oracle_pass_ransac': oracle_result['pass_ransac'],
            'oracle_pass_pred': oracle_result['pass_pred'],
            'oracle_pass_all_three': oracle_result['pass_all_three'],
            'true_pass_pattern_code': oracle_result['true_pass_pattern_code'],
            'true_pass_pattern_name': oracle_result['true_pass_pattern_name'],
            'metric_matches_oracle': metric_matches_oracle,
            'oracle_override_enabled': oracle_override_enabled,
            'metric_overridden_by_oracle': metric_overridden_by_oracle,
            'final_choice_code': final_choice_code,
            'final_choice_name': final_choice_name,
        }

        record = add_state_measurement_columns(
            record, i, x_k, z_p_k, z_omega_k, z_p1_k, z_q_k,
            z_q_k_1, None if ransac_error else z_q_k_2,
            z_p_k_1, None if ransac_error else z_p_k_2,
            associatedBbox_1, None if ransac_error else associatedBbox_2,
            omega_LLS, omega_L_to_B, omega_los_L, without_correction
        )
        record = add_truth_columns(record, i, debris_pos, debris_vel, omega_true, q_true)
        record = add_covariance_and_geometry_columns(record, P_k, debris_pos[i] if i < len(debris_pos) else None, Le, We, De)
        frame_records.append(record)
        if current_frame_good:
            assignment_records.append({
                'frame': i,
                'file_name': pickle_file,
                'geometry_name': task['geometry_name'],
                'combo_name': task['combo_name'],
                'combo_index': task['combo_index'],
                'run_number_for_combo': task['run_number_for_combo'],
                'ransac_pca_threshold': task['ransac_pca_threshold'],
                'orthonormal_thresh': task['orthonormal_thresh'],
                'eig_thresh': task['eig_thresh'],
                'partial_occlusion': current_partial,
            'metric_active': metric_result['metric_active'],
            'metric_choice_code': metric_result['choice_code'],
            'metric_choice_name': metric_result['choice_name'],
            'metric_stage_code': metric_result['stage_code'],
            'metric_stage_name': metric_result['stage_name'],
            'agreement_code': metric_result['agreement_code'],
            'agreement_name': metric_result['agreement_name'],
            'agree_pca_ransac': metric_result['agree_pca_ransac'],
            'agree_pca_pred': metric_result['agree_pca_pred'],
            'agree_ransac_pred': metric_result['agree_ransac_pred'],
            'agree_all_three': metric_result['agree_all_three'],
            'oracle_choice_code': oracle_result['choice_code'],
            'oracle_choice_name': oracle_result['choice_name'],
            'oracle_status_code': oracle_result['status_code'],
            'oracle_status_name': oracle_result['status_name'],
            'oracle_pass_pca': oracle_result['pass_pca'],
            'oracle_pass_ransac': oracle_result['pass_ransac'],
            'oracle_pass_pred': oracle_result['pass_pred'],
            'oracle_pass_all_three': oracle_result['pass_all_three'],
            'true_pass_pattern_code': oracle_result['true_pass_pattern_code'],
            'true_pass_pattern_name': oracle_result['true_pass_pattern_name'],
            'metric_matches_oracle': metric_matches_oracle,
            'oracle_override_enabled': oracle_override_enabled,
            'metric_overridden_by_oracle': metric_overridden_by_oracle,
            'final_choice_code': final_choice_code,
                'final_choice_name': final_choice_name,
            })
        if current_frame_good and (not RC_flag) and i > configs['start'] and metric_result['agree_pca_ransac']:
            if z_q_k is not None:
                RC_flag = True
                q_true = recalibrate_true_orientation(q_true, z_q_k, i)
                q_true = smoothen_q(q_true)

    # Create final dataframe
    master_file = pd.DataFrame(frame_records)
    if 'master_file_columns' in configs:
        existing_columns = [col for col in configs['master_file_columns'] if col in master_file.columns]
        extra_columns = [col for col in master_file.columns if col not in existing_columns]
        master_file = master_file.reindex(columns=existing_columns + extra_columns)

    os.makedirs('full_results', exist_ok=True)
    full_results_path = os.path.join('full_results', f'results_of_{output_stem}.csv')
    master_file.to_csv(full_results_path, sep=',', header=True, index=False)

    ######
    # metric assignment experiment
    #####
    assignment_results = pd.DataFrame(assignment_records)
    os.makedirs('assignment_results', exist_ok=True)
    assignment_path = os.path.join('assignment_results', output_stem + '.csv')
    assignment_results.to_csv(assignment_path, sep=',', header=True, index=False)

    if not assignment_results.empty:
        assignment_summary = (
            assignment_results
            .groupby([
                'metric_stage_name',
                'metric_choice_name',
                'oracle_choice_name',
                'true_pass_pattern_name',
                'agreement_name',
                'final_choice_name',
            ], dropna=False)
            .size()
            .reset_index(name='count')
        )
    else:
        assignment_summary = pd.DataFrame(columns=[
            'metric_stage_name', 'metric_choice_name', 'oracle_choice_name',
            'true_pass_pattern_name', 'agreement_name', 'final_choice_name', 'count'
        ])

    for col, value in {
        'geometry_name': task['geometry_name'],
        'combo_name': task['combo_name'],
        'combo_index': task['combo_index'],
        'run_number_for_combo': task['run_number_for_combo'],
        'ransac_pca_threshold': task['ransac_pca_threshold'],
        'orthonormal_thresh': task['orthonormal_thresh'],
        'eig_thresh': task['eig_thresh'],
        'file_name': pickle_file,
    }.items():
        assignment_summary[col] = value

    assignment_summary.to_csv(
        os.path.join('assignment_results', 'summary_' + output_stem + '.csv'),
        sep=',', header=True, index=False
    )

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
    rmse_q = np.sqrt(np.mean(np.array(rotation_errors[start_time_2:nframes]) ** 2))

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

    results = {
        'rmse_px': rmse_px,
        'rmse_py': rmse_py,
        'rmse_pz': rmse_pz,
        'rmse_omx': rmse_omx,
        'rmse_omy': rmse_omy,
        'rmse_omz': rmse_omz,
        'rmse_vdx': rmse_vdx,
        'rmse_vdy': rmse_vdy,
        'rmse_vdz': rmse_vdz,
        'rmse_x_before': rmse_x_before,
        'rmse_y_before': rmse_y_before,
        'rmse_z_before': rmse_z_before,
        'rmse_x_after': rmse_x_after,
        'rmse_y_after': rmse_y_after,
        'rmse_z_after': rmse_z_after,
        'rmse_q': rmse_q,
        'geometry_name': task['geometry_name'],
        'pickle_file': pickle_file,
        'pickle_path': pickle_path,
        'combo_name': task['combo_name'],
        'combo_index': task['combo_index'],
        'run_number_for_combo': task['run_number_for_combo'],
        'ransac_pca_threshold': task['ransac_pca_threshold'],
        'orthonormal_thresh': task['orthonormal_thresh'],
        'eig_thresh': task['eig_thresh'],
        'full_results_file': full_results_path,
        'assignment_results_file': assignment_path,
        'assignment_summary_file': os.path.join('assignment_results', 'summary_' + output_stem + '.csv'),
    }

    return results
