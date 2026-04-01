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
# from mpi4py import MPI
import logging
import yaml

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
    pg_k = x_k[16:19]

    # Rotation matrix - rodrigues formula
    R_k_kp1 = rodrigues(omega_k, dt)

    # Translate vertex to origin
    p1_ko = p1_k - p_k
    pg_ko = pg_k - p_k

    # Rotate vertices
    p1_kp1o = np.matmul(R_k_kp1, p1_ko.reshape(len(p1_ko), 1))
    pg_kp1o = np.matmul(R_k_kp1, pg_ko.reshape(len(pg_ko), 1))
    # Translate vertex back to new expected origin
    p1_kp1 = (p1_kp1o.T + p_k + v_k * dt).ravel()
    pg_kp1 = (pg_kp1o.T + p_k + v_k * dt).ravel()

    return p1_kp1, R_k_kp1, pg_kp1


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
                                  z_pg_k, z_pg_k_1, z_pg_k_2,
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

    # state estimate: [p, v, omega, p1, q]
    add_vec("state_est", x_k[0:3], ["x", "y", "z"])
    add_vec("state_est", x_k[3:6], ["vx", "vy", "vz"])
    add_vec("state_est", x_k[6:9], ["wx", "wy", "wz"])
    add_vec("state_est_p1", x_k[9:12], ["x", "y", "z"])
    q_est = normalize_quat(np.asarray(x_k[12:16]).copy())
    add_vec("state_est_q", q_est, ["w", "x", "y", "z"])
    add_vec("state_est_pg", x_k[16:19],["x","y","z"])

    # main measurements
    add_vec("meas_p", z_p_k, ["x", "y", "z"])
    add_vec("meas_w", z_omega_k, ["x", "y", "z"])
    add_vec("meas_p1", z_p1_k, ["x", "y", "z"])
    add_vec("meas_q", normalize_quat(np.asarray(z_q_k).copy()), ["w", "x", "y", "z"])
    add_vec("meas_pg", z_pg_k,["x","y","z"])

    raw_p = without_correction[i] if i < len(without_correction) else [np.nan, np.nan, np.nan]
    add_vec("meas_p_raw", raw_p, ["x", "y", "z"])

    # PCA-specific measurements
    add_vec("meas_pca_p", z_p_k_1, ["x", "y", "z"])
    add_vec("meas_pca_p1", associatedBbox_1[:, 0], ["x", "y", "z"])
    add_vec("meas_pca_q", normalize_quat(np.asarray(z_q_k_1).copy()), ["w", "x", "y", "z"])
    add_vec("meas_pca_pg", z_pg_k_1, ["x", "y", "z"])

    # RANSAC-specific measurements
    add_vec("meas_ransac_p", z_p_k_2, ["x", "y", "z"])
    p1_ransac = associatedBbox_2[:, 0] if associatedBbox_2 is not None else None
    add_vec("meas_ransac_p1", p1_ransac, ["x", "y", "z"])
    add_vec("meas_ransac_q", normalize_quat(np.asarray(z_q_k_2).copy()) if z_q_k_2 is not None else None, ["w", "x", "y", "z"])
    add_vec("meas_ransac_pg", z_pg_k_2, ["x", "y", "z"])

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
        "pg_x", "pg_y", "pg_z"
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

def find_center_of_mass(z_pg_k, v_k, Vs_i, PLs_i, omega_LB, omega_LD, p, threshold=0.1):
    """
    Estimate the center of mass from the geometric bounding box centroid.

    Points where the LOS velocity residual (VLs_i - v_k) is near zero lie on a
    plane passing through the COM. We fit that plane and project z_pg_k onto it.

    Args:
        z_pg_k:    (3,) geometric centroid of bounding box, in frame L
        v_k:       (3,) estimated COM translational velocity, with respect to frame L, expressed in frame L
        Vs_i:     (N, ) array of line-of-sight speeds
        PLs_i:     (N, 3) array of point cloud positions, in frame L
        omega      (3,) angular velocity of L with respect to B, from prediction?
        p          (3,) position of debris relative to B, from prediction?
        threshold: scalar, max residual magnitude to select near-zero points

    Returns:
        z_p_k: (3,) estimated center of mass position, or z_pg_k if fallback
    """
    # Select points where LOS velocity is explained by translation alone
    v_LD = v_k
    u_los = PLs_i / np.linalg.norm(PLs_i, axis=1, keepdims=True)  # (N, 3)
    centered_points = PLs_i - p    
    omega_BL = -omega_LB                                             # (N, 3) r - p
    v_com_apparent = v_LD + np.cross(omega_BL, p)                               # (3,)
    # omega_BD_cross = np.cross(omega_BL + omega_LD, centered_points)             # (N, 3)
    v_total = v_com_apparent                                  # (N, 3)
    v_projected = np.sum(u_los * v_total, axis=1)                               # (N,)

    residuals = Vs_i - v_projected                                              # (N,)
    mask = np.abs(residuals) < threshold
    near_zero_points = PLs_i[mask]

    visualize = True
    if visualize:
        import matplotlib.pyplot as plt

        Xp = PLs_i[:, 0]
        Yp = PLs_i[:, 1]

        share_color_scale = True
        point_size = 8
        alpha = 0.8
        cmap = "coolwarm"

        if share_color_scale:
            vmax_vel = max(np.max(np.abs(Vs_i)), np.max(np.abs(v_projected)))
            vmin_vel = -vmax_vel
            vmax_res = np.max(np.abs(residuals))
            vmin_res = -vmax_res
        else:
            vmin_vel = vmax_vel = None
            vmin_res = vmax_res = None

        # --- Figure 1: measured LOS velocity ---
        fig1 = plt.figure(figsize=(7, 6))
        ax1 = fig1.add_subplot(111)
        sc1 = ax1.scatter(
            Xp, Yp, c=Vs_i, s=point_size, alpha=alpha, cmap=cmap,
            vmin=vmin_vel, vmax=vmax_vel
        )
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("XY point cloud colored by measured LOS velocity Vs_i")
        ax1.set_aspect("equal", adjustable="box")
        plt.colorbar(sc1, ax=ax1, label="Vs_i [m/s]")

        # --- Figure 2: projected translational LOS velocity ---
        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111)
        sc2 = ax2.scatter(
            Xp, Yp, c=v_projected, s=point_size, alpha=alpha, cmap=cmap,
            vmin=vmin_vel, vmax=vmax_vel
        )
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.set_title("XY point cloud colored by projected velocity")
        ax2.set_aspect("equal", adjustable="box")
        plt.colorbar(sc2, ax=ax2, label="v_projected [m/s]")

        # --- Figure 3: residuals ---
        fig3 = plt.figure(figsize=(7, 6))
        ax3 = fig3.add_subplot(111)
        sc3 = ax3.scatter(
            Xp, Yp, c=residuals, s=point_size, alpha=alpha, cmap="coolwarm",
            vmin=vmin_res, vmax=vmax_res
        )
        ax3.set_xlabel("X [m]")
        ax3.set_ylabel("Y [m]")
        ax3.set_title("XY point cloud colored by residuals (Vs_i - v_projected)")
        ax3.set_aspect("equal", adjustable="box")
        plt.colorbar(sc3, ax=ax3, label="residual [m/s]")

        # --- Figure 4: near-zero residual points ---
        fig4 = plt.figure(figsize=(7, 6))
        ax4 = fig4.add_subplot(111)
        ax4.scatter(Xp, Yp, s=6, alpha=0.25, label="all points")
        if near_zero_points.shape[0] > 0:
            ax4.scatter(
                near_zero_points[:, 0],
                near_zero_points[:, 1],
                s=12,
                alpha=0.9,
                label=f"|residual| < {threshold}"
            )
        ax4.set_xlabel("X [m]")
        ax4.set_ylabel("Y [m]")
        ax4.set_title("Near-zero residual points in XY")
        ax4.set_aspect("equal", adjustable="box")
        ax4.legend(loc="best")

        plt.tight_layout()
        plt.show()

    # Fallback: not enough points to fit a plane
    if np.sum(mask) < 3:
        return z_pg_k.copy(), near_zero_points

    # Fit plane via SVD on mean-centered near-zero points
    centroid = np.mean(near_zero_points, axis=0)
    _, _, Vt = np.linalg.svd(near_zero_points - centroid)
    v1 = Vt[0]  # line vector
    v2 = centroid / np.linalg.norm(centroid) # vector from lidar to center of near-zero point cloud
    normal = np.cross(v1, v2)
    n_hat = normal / np.linalg.norm(normal) # normalize vector

    # Project z_pg_k onto the plane: move it along n_hat by signed distance d
    d = np.dot(n_hat, z_pg_k - centroid)
    z_p_k = z_pg_k - d * n_hat
    # print(near_zero_points)
    return z_p_k, near_zero_points

def run(pickle_file, configs, logger):

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
    pg_0 = np.array([0.,0.,0.]) # geometric box centroid
    x_0 = np.hstack([p_0, vT_0, omega_0, p1_0, q_ini, pg_0])
    num_states = len(x_0)

    # Initial covariance
    P_0 = np.diag(configs['ini_covariance_guess'])  # Initial Covariance matrix

    # Process noise covariance matrix
    qp = configs['ini_process_noise_cov'][0]
    qv = configs['ini_process_noise_cov'][1]
    qom = configs['ini_process_noise_cov'][2]
    qp1 = configs['ini_process_noise_cov'][3]
    qq = configs['ini_process_noise_cov'][4]
    qpg = configs['ini_process_noise_cov'][5]
    Q = np.diag([qp, qp, qp, qv, qv, qv, qom, qom, qom, qp1, qp1, qp1, qq, qq, qq, qq, qpg, qpg, qpg])

    # Measurement noise covariance matrix
    p = configs['ini_meas_noise_cov'][0]
    om = configs['ini_meas_noise_cov'][1]
    p1 = configs['ini_meas_noise_cov'][2]
    q = configs['ini_meas_noise_cov'][3]
    pg = configs['ini_meas_noise_cov'][4]
    R1 = np.diag([p, p, p, om, om, om, p1, p1, p1, q, q, q, q, pg, pg, pg])
    R2 = np.diag([p, p, p, om, om, om, p1, p1, p1, pg, pg, pg])

    z_q_k_1_previous = np.zeros((4,))
    z_q_k_2_previous = np.zeros((4,))
    # q_km1 = np.zeros((4,))

    # Measurement matrix
    H1 = np.zeros([len(P_0)-3, len(P_0)])  # no measuring of velocity
    H1[0:3,0:3] = np.eye(3)
    H1[3:,6:] = np.eye(13)
    bad_attitude_measurement_flag = False
    adapt = False

    H2 = np.zeros([12,19])
    H2[0:3,0:3] = np.eye(3)
    H2[3:6,6:9] = np.eye(3)
    H2[6:9, 9:12] = np.eye(3)
    H2[9:,16:] = np.eye(3)

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
    for i in range(nframes):
        Le = np.nan
        We = np.nan
        De = np.nan
        visualize_flag = i%20==0
        # visualize_flag = True
        print(f"Iteration {i}")
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
                q_k = sp[12:16]
                pg_k = sp[16:]

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
                p1_kp1, R_k_kp1, pg_kp1 = verticeupdate(dt, sp)

                # Orientation Update
                if i == 1:
                    q_kp1s.append(q_ini)

                q_kp1 = orientationupdate(dt, sp)
                q_kp1s.append(q_kp1)

                sp_kp1_jdx = np.hstack([p_kp1, v_kp1, omega_kp1, p1_kp1, q_kp1, pg_kp1]).ravel()
                sp_kp1s.append(sp_kp1_jdx)

                # weighted sum of sigma points to get updated state
                if jdx == 0:
                    x_kp1 += w_0_m * sp_kp1_jdx
                else:
                    x_kp1 += w_j_m * sp_kp1_jdx
            

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

        if i == 0: # initialize linear velocity for spin axis and omegaLLS
            v_k = vT_0
        else:
            v_k = x_kp1[3:6]
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
        z_pi_k_1, z_pg_k_1, R_1, evals = boundingbox.bbox3d(X_i, Y_i, Z_i, True)  # unassociated bbox
        evecs = R_1.copy()
        if i == 0:
            q_kp1 = rotm2quat(R_1)
        z_pi_k_2, z_pg_k_2, R_1_2, normal_vecs, ranking, num_planes = boundingbox.boundingbox3D_RANSAC(X_i, Y_i, Z_i, q_kp1, True, False)

        omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)

        if i == 0: 
            z_p_k_1, near_zero_points = find_center_of_mass(z_pg_k_1, debris_vel[i], VBs[i], PLs[i], omega_L_to_B, omega_true, debris_pos[i])
            z_p_k_2, near_zero_points = find_center_of_mass(z_pg_k_2, debris_vel[i], VBs[i], PLs[i], omega_L_to_B, omega_true, debris_pos[i])
        else:
            # z_p_k_1, near_zero_points = find_center_of_mass(z_pg_k_1, v_k, VBs[i], PLs[i], omega_L_to_B, x_k[6:9], x_kp1[0:3])
            # z_p_k_2, near_zero_points = find_center_of_mass(z_pg_k_2, v_k, VBs[i], PLs[i], omega_L_to_B, x_k[6:9], x_kp1[0:3])
            print(Rot_L_to_B[i] @ omega_true)
            z_p_k_1, near_zero_points = find_center_of_mass(z_pg_k_1, debris_vel[i], VBs[i], PLs[i], omega_L_to_B, omega_true, debris_pos[i])
            z_p_k_2, near_zero_points = find_center_of_mass(z_pg_k_2, debris_vel[i], VBs[i], PLs[i], omega_L_to_B, omega_true, debris_pos[i])
        
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
                estimated = np.asarray(estimated_inB)
                estimated = estimated[int(interval_time / dt):, :]
                true_inB = np.array([Rot_L_to_B[hdx] @ pos for hdx, pos in enumerate(debris_pos)])
                true = np.array(true_inB)
                true = true[int(interval_time / dt):int((interval_time + t_interval) / dt) + 1, :]

                thresh = configs['threshold']  # initial threshold to remove frequencies obtained from crosstalk with baseband frequency
                num_sin = configs['number_of_sinusoids']  # number of sinusoids to use to fit the data
                skip = configs['number_of_skips']  # when choosing frequencies from frequency according to decreasing magnitude, skips this many frequencies
                params_z, constant_z, bias_removal_success = remove_bias(interval_time, dt, z[:, 2], estimated[:, 2], num_sin, thresh, skip, true[:, 2], params_z)
                parameters = [params_x, params_y, params_z]

                constants = [0, 0, constant_z]
                done = 1

        #####################

        # Orientation association
        # R_1 is obtained from bounding box
        if i == 0:
            z_q_k_1 = rotm2quat(R_1)
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
        else:
            z_q_k_1, _, error = rotation_association(q_kp1, R_1)
            if not ransac_error:
                z_q_k_2, bad_attitude_measurement_flag_2, error_2 = rotation_association(q_kp1, R_1_2)

        if i > 0:
            LWD = 2 * quat2rotm(q_kp1).T @ (p_kp1 - p1_kp1)
            L = LWD[0];
            W = LWD[1];
            D = LWD[2]
            predictedBbox = boundingbox.from_params(p_kp1, q_kp1, L, W, D)  # just use the predicted box instead

            # first use q from R_1 to get L,W,D
            # then use z_q_k (not perfectly aligned) to get
        associatedBbox_1, Lm, Wm, Dm = boundingbox.associated(z_q_k_1, z_pi_k_1, z_pg_k_1,
                                                              R_1)  # L: along x-axis, W: along y-axis D: along z-axis
        z_p1_k_1 = associatedBbox_1[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)
        if not ransac_error:
            associatedBbox_2, Lm_2, Wm_2, Dm_2 = boundingbox.associated(z_q_k_2, z_pi_k_2, z_pg_k_2, R_1_2)
            z_p1_k_2 = associatedBbox_2[:, 0]  # represents negative x,y,z corner (i.e. bottom, left, back in axis aligned box)

        if i == 0:
            associatedBbox = associatedBbox_1.copy()
            z_p1_k = associatedBbox_1[:, 0]
            z_q_k_1_previous = z_q_k_1.copy()
            if not ransac_error:
                z_q_k_2_previous = z_q_k_2.copy()

        # Default metric inputs so the selector functions can be called safely on
        # the first frame and during startup before all comparisons are available.
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
            orthonormal_thresh = configs['orthonormal_thresh']

            if i > configs['start']:
                pca_prev_thresh = configs['previous_threshold_multiplier'] * dt * np.rad2deg(np.linalg.norm(omega_kp1))
                ran_prev_thresh = configs['previous_threshold_multiplier'] * dt * np.rad2deg(np.linalg.norm(omega_kp1))

            # pca metric
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
                # use ransac
                z_q_k = z_q_k_2.copy()
                z_pi_k = z_pi_k_2.copy()
                z_p_k = z_p_k_2.copy()
                z_pg_k = z_pg_k_2.copy()
                z_p1_k = associatedBbox_2[:, 0]
                associatedBbox = associatedBbox_2.copy()
                adapt = False
                choice = 'ransac'
            except UnboundLocalError:
                # use pca
                z_q_k = z_q_k_1.copy()
                z_pi_k = z_pi_k_1.copy()
                z_p_k = z_p_k_1.copy()
                z_pg_k = z_pg_k_1.copy()
                z_p1_k = associatedBbox_1[:, 0]
                associatedBbox = associatedBbox_1.copy()
                adapt = False
                choice = 'pca'
        elif use_measurement == 1:
            # use pca
            z_q_k = z_q_k_1.copy()
            z_pi_k = z_pi_k_1.copy()
            z_p_k = z_p_k_1.copy()
            z_pg_k = z_pg_k_1.copy()
            z_p1_k = associatedBbox_1[:, 0]
            associatedBbox = associatedBbox_1.copy()
            adapt = False
            choice = 'pca'
        else:
            # use prediction
            associatedBbox = predictedBbox.copy()
            z_p_k = z_p_k_1.copy()
            z_pg_k = z_pg_k_1.copy()
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
        if curr_t >= (t_start + t_interval) and bias_removal_success:
            z_p_k_z = correct_bias(z_p_k, i, dt, parameters, constants, Rot_L_to_B[i], Rot_B_to_L[i])
            z_p_k = z_p_k_z

        # 2. Rotation of B Frame
        omega_L_to_B = estimate_rotation_B(Rot_L_to_B, i, dt)
        B_v_BL = np.cross(-Rot_L_to_B[i] @ omega_L_to_B, Rot_L_to_B[i] @ z_p_k)

        omega_LLS = np.zeros(3)
        omega_los_L = np.zeros(3)

        # find angular velocity from LOS velocities
        if i > 0:
            # 1. Linear Least Squares
            omega_LLS_B = estimate_LLS(XBs[i], YBs[i], ZBs[i], Rot_L_to_B[i] @ z_p_k, Rot_L_to_B[i] @ v_k, VBs[i], B_v_BL)
            omega_LLS = Rot_B_to_L[i] @ omega_LLS_B



        # 3. Kabsch
        ################ to use Kabsch you need i > 0, to wait for state initializations?
        if i == 0:
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
            z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_pg_k])
            H = H2
            R = R2
        else:
            z_kp1 = np.hstack([z_p_k, z_omega_k, z_p1_k, z_q_k, z_pg_k])
            H = H1
            R = R1

        # Set initial states to measurements
        if i == 0:
            x_k = np.hstack([z_p_k, vT_0, z_omega_k, z_p1_k, z_q_k_1, z_pg_k])  # state
            P_k = P_0.copy()  # covariance matrix
            Le, We, De = get_dimensions(x_k[9:12], x_k[16:], x_k[12:16])

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
            x_op[12:16] = normalize_quat(x_op[12:16])
            x_k = x_op.copy()

            x_p_k = x_k[0:3]
            x_p1_k = x_k[9:12]
            x_q_k = x_k[12:16]
            x_pg_k = x_k[16:19]
            rotation_errors.append(np.rad2deg(quat_angle_diff(x_q_k, q_true[i, :])))
            Le, We, De = get_dimensions(x_p1_k, x_pg_k, x_q_k)
            bbox3_dimensions.append([Le, We, De])

            # smooth out covariance off diagonals
            P_k = 0.5 * P_k + 0.5 * P_k.T

            z_q_k_1_previous = z_q_k_1.copy()
            if not ransac_error:
                z_q_k_2_previous = z_q_k_2.copy()

        visualize_flag = False
        if visualize_flag:
            # if False:
            print('PCA True diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1, q_true[i, :]))))
            print('Ransac True diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, q_true[i, :]))))
            print('Pred True diff.:' + str(np.rad2deg(quat_angle_diff(q_kp1, q_true[i, :]))))
            print('PCA Pred diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1, q_kp1))))
            print('Ransac Pred diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, q_kp1))))
            print('Ransac PCA diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2, z_q_k_1))))
            print('PCA Prev. diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_1_previous, z_q_k_1))))
            print('Ransac Prev. diff.:' + str(np.rad2deg(quat_angle_diff(z_q_k_2_previous, z_q_k_2))))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.legend()
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            ax.title.set_text(
                f'Time={i * dt}s' + '\n' + f'Pred. Length={round(Lm, 2)}m ' + f'Width={round(Wm, 2)}m ' + f'Height={round(Dm, 2)}m' + '\n' + f'Meas. Length={round(Lm, 2)}m ' + f'Width={round(Wm, 2)}m ' + f'Height={round(Dm, 2)}m')
            # width = orange to green, blue to green
            # length = orange to cyan, blue to cyan
            # height = orange to magenta, blue to magenta
            ax.scatter(X_i, Y_i, Z_i, color='black', marker='o', s=2)
            u_los = PLs[i] / np.linalg.norm(PLs[i], axis=1, keepdims=True)  # (N, 3)
            VBi = np.asarray(VBs[i])
            omega_BL = -omega_L_to_B
            V_i = VBi[:, np.newaxis] * u_los - (debris_vel[i] + np.cross(omega_BL, debris_pos[i]))
            ax.quiver(X_i, Y_i, Z_i, V_i[:,0], V_i[:,1], V_i[:,2])
            # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], marker='o', color='r')


            # print(x_kp1)
            # drawrectangle(ax, p1_kp1, p2_kp1, p3_kp1, p4_kp1, p5_kp1, p6_kp1, p7_kp1, p8_kp1, 'orange', 1)
            # drawrectangle(ax, associatedBbox_1[:, 0], associatedBbox_1[:, 1], associatedBbox_1[:, 2],
            #             associatedBbox_1[:, 3],
            #             associatedBbox_1[:, 4], associatedBbox_1[:, 5], associatedBbox_1[:, 6], associatedBbox_1[:, 7],
            #             'b', 2, 'PCA')

            drawrectangle(ax, associatedBbox_2[:, 0], associatedBbox_2[:, 1], associatedBbox_2[:, 2],
                        associatedBbox_2[:, 3],
                        associatedBbox_2[:, 4], associatedBbox_2[:, 5], associatedBbox_2[:, 6], associatedBbox_2[:, 7],
                        'orange', 2, 'RANSAC')

            # drawrectangle(ax, associatedBbox[:, 0], associatedBbox[:, 1], associatedBbox[:, 2], associatedBbox[:, 3],
            #           associatedBbox[:, 4], associatedBbox[:, 5], associatedBbox[:, 6], associatedBbox[:, 7], 'orange', 2)

            # drawrectangle(ax, z_pi_k[:, 0], z_pi_k[:, 1], z_pi_k[:, 2], z_pi_k[:, 3],
            #               z_pi_k[:, 4], z_pi_k[:, 5], z_pi_k[:, 6], z_pi_k[:, 7], 'r', 1)
            # ax.scatter(p1_kp1[0], p1_kp1[1], p1_kp1[2], color='b', s=20)
            # drawrectangle(ax, predictedBbox[:, 0], predictedBbox[:, 1], predictedBbox[:, 2], predictedBbox[:, 3],
            #               predictedBbox[:, 4], predictedBbox[:, 5], predictedBbox[:, 6], predictedBbox[:, 7], 'r', 1)

            # ax.scatter(predictedBbox[0, 0], predictedBbox[1, 0], predictedBbox[2, 0], color='orange', label='Vertex 1 Pred.')
            # ax.scatter(associatedBbox[0, 0], associatedBbox[1, 0], associatedBbox[2, 0], color='blue',
            #            label='Vertex 1 Meas.')


            Rot_measured = quat2rotm(z_q_k_1)

            Rot_measured_2 = quat2rotm(z_q_k_2)
            # Rot_measured_2 = R_1_2
            # normal_vecs = normal_vecs.T

            R_estimated = quat2rotm(q_kp1)

            R_true = quat2rotm(q_true[i, :])

            # plot measured
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 0]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured[2, 0]],
            #         color='blue', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 1]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured[2, 1]],
            #         color='blue', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured[1, 2]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured[2, 2]],
            #         color='b', linewidth=4)

            # plot measured
            ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 0]],
                    [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 0]],
                    color='orange', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 1]],
                    [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 1]],
                    color='orange', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 2]],
                    [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 2]],
                    color='orange', linewidth=4)
            #
            # Rot_measured_2 = R_1_2
            # plot measured
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 0]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 0]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 0]],
            #         color='red', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 1]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 1]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 1]],
            #         color='red', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + Rot_measured_2[0, 2]], [z_p_k[1], z_p_k[1] + Rot_measured_2[1, 2]],
            #         [z_p_k[2], z_p_k[2] + Rot_measured_2[2, 2]],
            #         color='red', linewidth=4)

            # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 0]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 0]],
            #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 0]],
            #         color='blue', linewidth=4)
            # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 1]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 1]],
            #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 1]],
            #         color='blue', linewidth=4)
            # ax.plot([z_p_k_2[0], z_p_k_2[0] + normal_vecs[0, 2]], [z_p_k_2[1], z_p_k_2[1] + normal_vecs[1, 2]],
            #         [z_p_k_2[2], z_p_k_2[2] + normal_vecs[2, 2]],
            #         color='blue', linewidth=4)
            #

            # plot current estimate of ekf
            # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 0]], [z_p_k[1], z_p_k[1] + R_estimated[1, 0]],
            #         [z_p_k[2], z_p_k[2] + R_estimated[2, 0]],
            #         color='red', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 1]], [z_p_k[1], z_p_k[1] + R_estimated[1, 1]],
            #         [z_p_k[2], z_p_k[2] + R_estimated[2, 1]],
            #         color='red', linewidth=4)
            # ax.plot([z_p_k[0], z_p_k[0] + R_estimated[0, 2]], [z_p_k[1], z_p_k[1] + R_estimated[1, 2]],
            #         [z_p_k[2], z_p_k[2] + R_estimated[2, 2]],
            #         color='red', linewidth=4, label='Predicted')
            #
            # plot true
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 0]], [z_p_k[1], z_p_k[1] + R_true[1, 0]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 0]],
                    color='green', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 1]], [z_p_k[1], z_p_k[1] + R_true[1, 1]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 1]],
                    color='green', linewidth=4)
            ax.plot([z_p_k[0], z_p_k[0] + R_true[0, 2]], [z_p_k[1], z_p_k[1] + R_true[1, 2]],
                    [z_p_k[2], z_p_k[2] + R_true[2, 2]],
                    color='green', linewidth=4, label='True')

            # plot b_frame
            # ax.plot([0., 0. + Rot_B_to_L[i][0, 0]], [0., 0. + Rot_B_to_L[i][1, 0]],
            #         [0., 0. + Rot_B_to_L[i][2, 0]],
            #         color='r', linewidth=1)
            # ax.plot([0., 0. + Rot_B_to_L[i][0, 1]], [0., 0. + Rot_B_to_L[i][1, 1]],
            #         [0., 0. + Rot_B_to_L[i][2, 1]],
            #         color='g', linewidth=1)
            # ax.plot([0., 0. + Rot_B_to_L[i][0, 2]], [0., 0. + Rot_B_to_L[i][1, 2]],
            #         [0., 0. + Rot_B_to_L[i][2, 2]],
            #         color='b', linewidth=1)

            # black is axis of rotation
            # ax.plot([z_p_k[0], z_p_k[0] + 1], [z_p_k[1], z_p_k[1] + 1],
            #         [z_p_k[2], z_p_k[2] + 1],
            #         color='black', linewidth=4)

            # outlier_cloud = pcd.select_by_index(inliers, invert=True)

            # Visualize the inliers (plane) and outliers
            # inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red plane
            # outlier_cloud.paint_uniform_color([0.0, 1, 0])  # Green remaining points
            # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

            # ax.scatter(x_k[0], x_k[1], x_k[2], color='orange' )
            ax.scatter(z_p_k_1[0], z_p_k_1[1], z_p_k_1[2], color='b', label='Box Centroid')
            ax.scatter(z_p_k[0], z_p_k[1], z_p_k[2], color='r', label='COM')
            ax.scatter(near_zero_points[:,0], near_zero_points[:,1], near_zero_points[:,2], color='c', label="near zero")
            ax.scatter(debris_pos[i,0], debris_pos[i,1], debris_pos[i,2], color='g', label='True Position')
            ax.legend()
            ax.set_aspect('equal', 'box')
            plt.show()

        z_s.append(z_kp1)
        z_pcas.append(np.hstack([z_p_k_1, z_omega_k, associatedBbox_1[:, 0], z_q_k_1, z_pg_k_1]))
        if not ransac_error:
            z_rans.append(np.hstack([z_p_k_2, z_omega_k, associatedBbox_2[:, 0], z_q_k_2, z_pg_k_2]))
        else:
            z_rans.append(np.hstack([np.zeros_like(z_p_k_1), np.zeros_like(z_omega_k), np.zeros_like(associatedBbox_1[:, 0]), np.zeros_like(z_q_k_1), np.zeros_like(z_pg_k_1)]))

        # Append for analysis
        P_s.append(P_k)
        x_s.append(x_k)
        estimated_pos.append(x_k[:3])

        # Append data for output file
        record = {
            'file_name': pickle_file,
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
            z_pg_k, z_pg_k_1, z_pg_k_2,
            z_q_k_1, None if ransac_error else z_q_k_2,
            z_p_k_1, None if ransac_error else z_p_k_2,
            associatedBbox_1, None if ransac_error else associatedBbox_2,
            omega_LLS, omega_L_to_B, omega_los_L, without_correction
        )
        record = add_truth_columns(record, i, debris_pos, debris_vel, omega_true, q_true)
        record = add_covariance_and_geometry_columns(record, P_k, debris_pos[i] if i < len(debris_pos) else None, Le, We, De)
        frame_records.append(record)
        assignment_records.append({
            'frame': i,
            'file_name': pickle_file,
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
        if (not RC_flag) and i > configs['start'] and metric_result['agree_pca_ransac']:
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
    master_file.to_csv('full_results/results_of_' + pickle_file.split('.')[0] + '.csv', sep=',', header=True, index=False)

    ######
    # metric assignment experiment
    #####
    assignment_results = pd.DataFrame(assignment_records)
    os.makedirs('assignment_results', exist_ok=True)
    assignment_path = 'assignment_results/' + configs['assignment_results_file_name'] + pickle_file.split('.')[0] + '.csv'
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
    assignment_summary.to_csv(
        'assignment_results/summary_' + configs['assignment_results_file_name'] + pickle_file.split('.')[0] + '.csv',
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

        plt.rcParams.update({'font.size': 12})
        plt.rcParams['text.usetex'] = True

        # fig = plt.figure()
        # plt.plot(np.arange(0, dt*nframes, dt), centroids_inB[:, 0] - true_pos_inB[:, 0])
        # plt.xlabel('Time (s)')
        # plt.ylabel('$\displaystyle p_x$ (m)')
        #
        # fig = plt.figure()
        # plt.plot(np.arange(0, dt*nframes, dt), centroids_inB[:, 1] - true_pos_inB[:, 1])
        # plt.xlabel('Time (s)')
        # plt.ylabel('$\displaystyle p_y$ (m)')

        fig = plt.figure()
        z_s = np.array(z_s)
        centroids_inB = np.array(centroids_inB)
        true_pos_inB = np.array(true_pos_inB)
        plt.plot(np.arange(0, dt * nframes, dt), centroids_inB[:, 2] - true_pos_inB[:, 2], label='Computed',
                 color='blue')
        plt.plot(np.arange(0, dt * nframes, dt), np.zeros_like(true_pos_inB), label='True', color='Green',
                 linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_z$ (m)')
        plt.legend()

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 0] - debris_pos[:, 0], label='Computed', linewidth=1,
                 color='blue')
        plt.plot(np.arange(0, dt * nframes, dt), without_correction[:, 0] - debris_pos[:, 0], label='Original',
                 linewidth=1, color='brown')
        plt.plot(np.arange(0, dt * nframes, dt), np.zeros_like(z_s[:, 0]), label='True', color='green', linestyle='--',
                 linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_x$ (m)')
        plt.legend()
        # plt.title('X Position')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 1] - debris_pos[:, 1], label='Computed', linewidth=1,
                 color='blue')
        plt.plot(np.arange(0, dt * nframes, dt), without_correction[:, 1] - debris_pos[:, 1], label='Original',
                 linewidth=1, color='brown')
        plt.plot(np.arange(0, dt * nframes, dt), np.zeros_like(z_s[:, 1]), label='True', color='green', linestyle='--',
                 linewidth=1)
        # plt.plot(np.arange(0, dt*nframes, dt), x_s[:m1-1,1] - debris_pos[:,1], label='Estimated', linewidth=2)
        # plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,1], label='True', linewidth=1, linestyle='dashed')

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_y$ (m)')
        # plt.title('Y Position')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 2] - debris_pos[:, 2], label='Computed', linewidth=1,
                 color='blue')
        plt.plot(np.arange(0, dt * nframes, dt), without_correction[:, 2] - debris_pos[:, 2], label='Original',
                 linewidth=1, color='brown')
        plt.plot(np.arange(0, dt * nframes, dt), np.zeros_like(z_s[:, 2]), label='True', color='green', linestyle='--',
                 linewidth=1)

        # plt.plot(np.arange(0, dt*nframes, dt), debris_pos[:,2], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_z$ (m)')
        # plt.title('Z Position')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.scatter(debris_pos[1, 0], debris_pos[1, 1], debris_pos[1, 2], color='orange', marker='o', s=20)
        ax.scatter(debris_pos[-1, 0], debris_pos[-1, 1], debris_pos[-1, 2], color='k', marker='o', s=20)
        ax.scatter(z_s[:, 0], z_s[:, 1], z_s[:, 2], color='b', s=0.3, linewidths=0)
        ax.plot(debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2], color='g')
        ax.legend(['Start', 'End', 'Computed Centroid Positions', 'True Centroid Positions'])
        # plt.xlim([-170.5, -167.5])
        # plt.ylim([-351, -306])
        # ax.set_zlim(-20, -9)

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 3], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 6], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), omega_true[0] * np.ones([nframes, 1]), label='True', linewidth=1,
                 linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle \Omega_x$ (rad/s)')
        # plt.title('$\displaystyle\Omega_x$')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 4], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 7], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), omega_true[1] * np.ones([nframes, 1]), label='True', linewidth=1,
                 linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle \Omega_y$ (rad/s)')
        # plt.title('Omega Y')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 5], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 8], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), omega_true[2] * np.ones([nframes, 1]), label='True', linewidth=1,
                 linestyle='dashed')
        plt.legend()

        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle \Omega_z$ (rad/s)')
        # plt.title('Omega Z')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 6] - omega_true[0], label='Error $\displaystyle \Omega_x$',
                 linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 7] - omega_true[1], label='Error $\displaystyle \Omega_y$',
                 linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 8] - omega_true[2], label='Error $\displaystyle \Omega_z$',
                 linewidth=2)
        # plt.plot(np.arange(0, dt*nframes, dt), np.zeros([nframes,1]), linewidth = 1) # draw line at zero
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity Error (rad/s)')
        # plt.title('Angular Velocity Errors')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 0] - debris_pos[:nframes, 0], label='Error $\displaystyle p_x$',
                 linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 1] - debris_pos[:nframes, 1], label='Error $\displaystyle p_y$',
                 linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 2] - debris_pos[:nframes, 2], label='Error $\displaystyle p_z$',
                 linewidth=2)

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Position Error (m)')
        # plt.title('Position Errors')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 0], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 0], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 0], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_x$ (m)')
        # plt.title('X Position')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 1], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 1], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 1], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_y$ (m)')
        # plt.title('Y Position')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 2], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 2], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_pos[:nframes, 2], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle p_z$ (m)')
        # plt.title('Z Position')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 3] - debris_vel[:nframes, 0],
                 label='Error $\displaystyle v_{Dx}$',
                 linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 4] - debris_vel[:nframes, 1],
                 label='Error $\displaystyle v_{Dy}$',
                 linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 5] - debris_vel[:nframes, 2],
                 label='Error $\displaystyle v_{Dz}$',
                 linewidth=1)

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity Error (m/s)')
        # plt.title('Velocity Errors')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 3], label='Estimated', color='Orange', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 0], label='True', color='green', linewidth=1,
                 linestyle='--')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle v_{Dx}$ (m/s)')
        # plt.title('Velocity in X')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 4], label='Estimated', color='Orange', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 1], label='True', color='green', linewidth=1,
                 linestyle='--')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle v_{Dy}$ (m/s)')
        # plt.title('Velocity in y')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 5], label='Estimated', color='Orange', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), debris_vel[:nframes, 2], label='True', color='green', linewidth=1,
                 linestyle='--')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle v_{Dz}$ (m/s)')
        # plt.title('Velocity in z')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 9], label='$\displaystyle p_{1x}$')
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 10], label='$\displaystyle p_{1y}$')
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 11], label='$\displaystyle p_{1z}$')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Vertex $\displaystyle p_{1}$ Position (m)')
        # plt.title('Position of Vertice P1 overt time')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 9], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 12], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 0], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle q_w$')
        # plt.title('Orientation $\displaystyle q_0$')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 10], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 13], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 1], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle q_x$')
        # plt.title('Orientation $\displaystyle q_1$')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 11], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 14], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 2], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle q_y$')
        # plt.title('Orientation $\displaystyle q_2$')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), z_s[:, 12], label='Computed', linewidth=1)
        plt.plot(np.arange(0, dt * nframes, dt), x_s[:, 15], label='Estimated', linewidth=2)
        plt.plot(np.arange(0, dt * nframes, dt), q_true[:nframes, 3], label='True', linewidth=1, linestyle='dashed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('$\displaystyle q_z$')
        # plt.title('Orientation $\displaystyle q_3$')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 0], label='Length')
        plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 1], label='Width')
        plt.plot(np.arange(0, dt * nframes, dt), bbox1_dimensions[:, 2], label='Height')
        plt.legend()
        plt.title('PCA Box Dimensions')
        plt.xlabel('Time (s)')
        plt.ylabel('Size (m)')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 0], label='Length')
        plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 1], label='Width')
        plt.plot(np.arange(0, dt * nframes, dt), bbox2_dimensions[:, 2], label='Height')
        plt.legend()
        plt.title('RANSAC Box Dimensions')
        plt.xlabel('Time (s)')
        plt.ylabel('Size (m)')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 0], label='Length')
        plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 1], label='Width')
        plt.plot(np.arange(0, dt * nframes, dt), bbox3_dimensions[:, 2], label='Height')
        # Add horizontal lines at 5.3, 1.3, and 2.4 meters
        plt.axhline(y=5.25, color='k', linestyle='--', label='True')  # Horizontal line at 5.3m
        plt.axhline(y=1.25, color='k', linestyle='--')  # Horizontal line at 1.3m
        plt.axhline(y=2.4, color='k', linestyle='--')  # Horizontal line at 2.4m
        plt.legend()
        plt.title('Filtered Box Dimensions')
        plt.xlabel('Time (s)')
        plt.ylabel('Size (m)')

        fig = plt.figure()
        plt.plot(np.arange(0, dt * nframes, dt), np.array(rotation_errors), label='Rotation Error')
        plt.title('Rotation Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle Error (deg)')


        """
        fig = plt.figure()
        true_b = []
        for i in range(nframes):
            true_b.append(Rot_L_to_B[i] @ [1,1,1])
        true_b = np.array(true_b)
        plt.plot(np.arange(0, dt*nframes, dt), true_b[:,2], label='True')

        plt.plot(np.arange(0, dt*nframes, dt), omega_kabsch_b[:,2], label='Computed')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Angular velocity (rad/s)')
        plt.title('$\displaystyle {}^B \Omega_z$ from Kabsch')
        """

        plt.show()


    results = [rmse_px, rmse_py, rmse_pz, rmse_omx, rmse_omy, rmse_omz, rmse_vdx, rmse_vdy, rmse_vdz, rmse_x_before, rmse_y_before, rmse_z_before,
               rmse_x_after, rmse_y_after, rmse_z_after, rmse_q]

    return results

def run_single(config):

    # Configure logging only for the process with rank 0
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    

    # get all file names
    pickle_files = os.listdir(config['pickle_directory_name'])
    checked = []
    simulation_data = []
    for file_name in os.listdir(config['pickle_directory_name']):
        if file_name in checked:
            pass
        else:
            results = run(file_name, config, logger)
            simulation_data.append(results)
            results_as_df = pd.DataFrame(simulation_data)
            results_as_df['pickle_file'] = file_name

            # save as csv
            results_as_df.to_csv(os.path.join(config['top_level_dir'], config['results_file_name']), sep=',', header=True,
                             index=False)
    return


with open('configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)

run_single(configs)