import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import numpy as np
import pandas as pd
import trimesh
import yaml

import dynamics_revamped as dynamics
import lidarScan3_revamped as lidarScan3


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config_revamped.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tilde(v):
    v = np.asarray(v).reshape(3)
    vx, vy, vz = v
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]])


def getR(x, y, z):
    p = np.array([x, y, z], dtype=float)
    z_L = np.array([0.0, 0.0, 1.0], dtype=float)
    z_B = p / np.linalg.norm(p)

    cross_vec = np.cross(z_L, z_B)
    cross_norm = np.linalg.norm(cross_vec)
    dot_val = np.clip(np.dot(z_B, z_L), -1.0, 1.0)

    if cross_norm < 1e-12:
        if dot_val > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])

    e = (cross_vec / cross_norm)[:, np.newaxis]
    phi = np.arccos(dot_val)
    R = e @ e.T + (np.identity(3) - (e @ e.T)) * np.cos(phi) + tilde(e) * np.sin(phi)
    return R.T


def process_frame(rank, i, mesh_template, debris_pos, debris_vel, angle_0, omega_L, cfg, config_path):
    lidar_cfg = cfg["lidar"]
    noise_cfg = cfg["noise"]
    sim_cfg = cfg["simulation"]

    x, y, z = debris_pos[i]
    d = np.linalg.norm(debris_pos[i])

    fov_req = np.rad2deg(2.0 * np.arctan2(float(lidar_cfg["res_box_m"]) / 2.0, d))
    fov_h = min(fov_req, float(lidar_cfg["max_horizontal_fov_deg"]))
    fov_v = min(fov_req, float(lidar_cfg["max_vertical_fov_deg"]))
    partial = bool(fov_req > fov_h or fov_req > fov_v)

    ang_res = float(lidar_cfg["angular_resolution_deg"])
    h_resolution = min(int(fov_h / ang_res), int(lidar_cfg["max_horizontal_rays"]))
    v_resolution = min(int(fov_v / ang_res), int(lidar_cfg["max_vertical_rays"]))
    h_resolution = max(h_resolution, 1)
    v_resolution = max(v_resolution, 1)

    debris = mesh_template.copy()
    Rot_L_to_B = getR(x, y, z)
    x_prev, y_prev, z_prev = debris_pos[i - 1] if i > 0 else debris_pos[i]
    Rot_L_to_B_prev = getR(x_prev, y_prev, z_prev)

    debris_pos_B = Rot_L_to_B @ debris_pos[i]
    debris_vel_B = Rot_L_to_B @ debris_vel[i]

    Rot_4by4 = np.eye(4)
    Rot_4by4[:3, :3] = Rot_L_to_B
    debris.apply_transform(Rot_4by4)

    axis = Rot_L_to_B @ (omega_L / np.linalg.norm(omega_L)) if np.linalg.norm(omega_L) > 0 else np.array([0.0, 0.0, 1.0])
    angle_0_rad = np.deg2rad(angle_0)
    angle = np.linalg.norm(omega_L * float(sim_cfg["dt"]) * i)
    debris.apply_transform(trimesh.transformations.rotation_matrix(angle_0_rad + angle, axis))
    debris.apply_transform(trimesh.transformations.translation_matrix(debris_pos_B))

    omega_B = Rot_L_to_B @ omega_L
    O_B = np.asarray(lidar_cfg.get("sensor_origin_B_m", [0.0, 0.0, 0.0]), dtype=float)
    X, Y, Z, V_los = lidarScan3.point_cloud(
        O_B,
        h_resolution,
        v_resolution,
        fov_h,
        fov_v,
        debris,
        debris_pos_B,
        debris_vel_B,
        omega_B,
        Rot_L_to_B,
        Rot_L_to_B_prev,
        float(sim_cfg["dt"]),
        config_path=config_path,
        noise_cfg=noise_cfg,
    )
    P = np.vstack([X, Y, Z]).T if len(X) > 0 else np.empty((0, 3))
    print(f"Process {rank} processing frame: {i}")
    return X, Y, Z, P, V_los, Rot_L_to_B, partial


def sample_vector(min_vals, max_vals):
    min_vals = np.asarray(min_vals, dtype=float)
    max_vals = np.asarray(max_vals, dtype=float)
    return np.random.uniform(min_vals, max_vals)


def get_initial_conditions(cfg, conditions_count=100, config_path: Optional[str] = None):
    starts_dict = []
    sim_cfg = cfg["simulation"]
    ic_cfg = cfg["initial_conditions"]
    dyn_cfg = cfg["dynamics"]

    dt = float(sim_cfg["dt"])
    nframes = int(sim_cfg["nframes"])
    mu = float(dyn_cfg["mu_km3_s2"])
    valid_distance_min = float(ic_cfg["valid_distance_m"]["min"])
    valid_distance_max = float(ic_cfg["valid_distance_m"]["max"])

    while len(starts_dict) < conditions_count:
        r0 = sample_vector(ic_cfg["position_km"]["min"], ic_cfg["position_km"]["max"])
        rdot0 = sample_vector(ic_cfg["velocity_km_s"]["min"], ic_cfg["velocity_km_s"]["max"])
        omega_L = sample_vector(ic_cfg["omega_rad_s"]["min"], ic_cfg["omega_rad_s"]["max"])
        angle_0 = np.random.uniform(ic_cfg["angle0_deg"]["min"], ic_cfg["angle0_deg"]["max"])
        altitude = np.random.uniform(ic_cfg["altitude_km"]["min"], ic_cfg["altitude_km"]["max"])

        earth_radius_km = 6378.0
        r = altitude + earth_radius_km
        mean_motion = np.sqrt(mu / r ** 3)
        _, _, _, _, _, _, d, _ = dynamics.propagate(dt, nframes, r0, rdot0, mean_motion, config_path=config_path)
        if max(d) > valid_distance_max or min(d) < valid_distance_min:
            continue

        starts_dict.append(
            {
                "px": float(r0[0]),
                "py": float(r0[1]),
                "pz": float(r0[2]),
                "vx": float(rdot0[0]),
                "vy": float(rdot0[1]),
                "vz": float(rdot0[2]),
                "angle_0": float(angle_0),
                "omx": float(omega_L[0]),
                "omy": float(omega_L[1]),
                "omz": float(omega_L[2]),
                "mean_motion": float(mean_motion),
                "nframes": nframes,
                "use_frames": bool(cfg["frame_dropout"]["enabled"]),
            }
        )
    return starts_dict


def build_frame_usage(nframes, frame_dropout_cfg):
    if frame_dropout_cfg.get("enabled", False):
        alpha = float(frame_dropout_cfg["alpha"])
        beta = float(frame_dropout_cfg["beta"])
        p_drop_run = np.random.beta(alpha, beta)
        use_frames = np.random.random(nframes) > p_drop_run
    else:
        p_drop_run = 0.0
        use_frames = np.ones(nframes, dtype=bool)
    return use_frames, p_drop_run


def run_single_simulation(rank, sim_parameters, sim_index, cfg, config_path: Optional[str] = None):
    r0 = np.array([sim_parameters["px"], sim_parameters["py"], sim_parameters["pz"]], dtype=float)
    rdot0 = np.array([sim_parameters["vx"], sim_parameters["vy"], sim_parameters["vz"]], dtype=float)
    omega_L = np.array([sim_parameters["omx"], sim_parameters["omy"], sim_parameters["omz"]], dtype=float)
    angle_0 = float(sim_parameters["angle_0"])
    mean_motion = float(sim_parameters["mean_motion"])
    nframes = int(sim_parameters["nframes"])

    use_frames, p_drop_run = build_frame_usage(nframes, cfg["frame_dropout"])
    dt = float(cfg["simulation"]["dt"])

    x, y, z, vx, vy, vz, d, v = dynamics.propagate(dt, nframes, r0, rdot0, mean_motion, config_path=config_path)
    debris_pos = np.vstack([x, y, z]).T
    debris_vel = np.vstack([vx, vy, vz]).T

    debris_file = cfg["mesh"]["debris_file"]
    debris_mesh = trimesh.load_mesh(debris_file)

    XBs, YBs, ZBs, PBs, VBs, Omega_Ls, Rot_L_to_Bs, partials = [], [], [], [], [], [], [], []
    for i in range(nframes):
        angle_i = mean_motion * dt * i
        axis = np.array([0.0, 0.0, 1.0])
        R_L = (
            np.cos(angle_i) * np.eye(3)
            + (1 - np.cos(angle_i)) * np.outer(axis, axis)
            + np.sin(angle_i) * tilde(axis)
        )
        omega_L_i = R_L @ omega_L
        X, Y, Z, P, V_los, Rot_L_to_B, partial = process_frame(
            rank,
            i,
            debris_mesh,
            debris_pos,
            debris_vel,
            angle_0,
            omega_L_i,
            cfg,
            config_path,
        )
        XBs.append(X)
        YBs.append(Y)
        ZBs.append(Z)
        PBs.append(P)
        VBs.append(V_los)
        Omega_Ls.append(omega_L_i)
        Rot_L_to_Bs.append(Rot_L_to_B)
        partials.append(partial)

    simulation_data = {
        "XBs": XBs,
        "YBs": YBs,
        "ZBs": ZBs,
        "PBs": PBs,
        "VBs": VBs,
        "debris_pos": debris_pos,
        "debris_vel": debris_vel,
        "Rot_L_to_B": Rot_L_to_Bs,
        "omega_L": omega_L,
        "Omega_Ls": Omega_Ls,
        "dt": dt,
        "angle_0": angle_0,
        "partial": partials,
        "use_frame": use_frames,
        "p_drop_run": p_drop_run,
        "mean_motion": mean_motion,
        "config": cfg,
    }

    output_cfg = cfg["output"]
    results_dir = Path(output_cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    sim_filename = output_cfg["sim_filename_template"].format(sim_index=sim_index)
    with open(results_dir / sim_filename, "wb") as sim_data:
        pickle.dump(simulation_data, sim_data)

    print(f"Process {rank} completed and saved simulation {sim_index}")
    return sim_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default=None)
    args = parser.parse_args()

    config_path = (args.config_path or os.environ.get("LIDAR_SIM_CONFIG", str(DEFAULT_CONFIG_PATH)))
    cfg = load_config(config_path)

    seed = cfg["simulation"].get("seed")
    if seed is not None:
        np.random.seed(int(seed))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    try:
        num_tests = int(cfg["simulation"]["num_tests"])
        initial_conditions_list = list(get_initial_conditions(cfg, num_tests, config_path=config_path))
        total_conditions = len(initial_conditions_list)
        local_results = []

        for i in range(rank, total_conditions, size):
            conditions = initial_conditions_list[i]
            sim_index = run_single_simulation(rank, conditions, i, cfg, config_path=config_path)
            local_results.append(sim_index)

        all_completed = comm.gather(local_results, root=0)
        if rank == 0:
            ini_cond_df = pd.DataFrame(initial_conditions_list)
            flat_completed = sorted(item for sublist in all_completed for item in sublist)
            ini_cond_df["file index"] = flat_completed
            ini_cond_df.to_csv(cfg["output"]["initial_conditions_csv"], sep=",", header=True, index=False)
            print(f"Total completed simulations: {len(flat_completed)}")
            print("Completed simulation indices:", flat_completed)

    except Exception as e:
        print(f"Error on process {rank}: {str(e)}")
        comm.Abort(1)
