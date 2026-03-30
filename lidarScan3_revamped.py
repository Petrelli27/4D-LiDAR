import numpy as np
import trimesh
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

import lidarNoise_revamped as lidarNoise


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def point_cloud(
    O_B,
    horizontal_resolution,
    vertical_resolution,
    h_range,
    v_range,
    sat_mesh: trimesh.Trimesh,
    sat_pos,
    v_rel,
    omega,
    Rot_L_to_B,
    Rot_L_to_B_prev,
    dt,
    config_path: Optional[str] = None,
    noise_cfg: Optional[Dict[str, Any]] = None,
):
    theta_r = np.deg2rad(np.linspace(-h_range / 2.0, h_range / 2.0, horizontal_resolution))
    phi_r = np.deg2rad(np.linspace(-v_range / 2.0, v_range / 2.0, vertical_resolution))
    Theta, Phi = np.meshgrid(theta_r, phi_r)

    rays = np.stack(
        [
            np.cos(Phi) * np.sin(Theta),
            np.sin(Phi),
            np.cos(Theta) * np.cos(Phi),
        ],
        axis=-1,
    ).reshape(-1, 3)

    locations, index_ray, index_tri = sat_mesh.ray.intersects_location(
        ray_origins=np.tile(O_B, (len(rays), 1)),
        ray_directions=rays,
        multiple_hits=False,
    )

    if len(locations) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    useful_rel_locations = locations - O_B
    r = useful_rel_locations - sat_pos
    u_los = -useful_rel_locations / np.linalg.norm(useful_rel_locations, axis=1)[:, np.newaxis]

    Rlb = Rot_L_to_B_prev.T @ Rot_L_to_B
    angle_B_to_B = 2.0 * np.arctan2(np.linalg.norm(Rlb - Rlb.T) / 2.0, 1.0)

    if angle_B_to_B < 1e-3:
        omega_L_to_B = np.array([0.0, 0.0, 0.0])
    else:
        axis_B_to_B = (1.0 / (2.0 * np.sin(angle_B_to_B))) * np.array(
            [
                Rlb[2, 1] - Rlb[1, 2],
                Rlb[0, 2] - Rlb[2, 0],
                Rlb[1, 0] - Rlb[0, 1],
            ]
        )
        axis_B_to_B = Rot_L_to_B.T @ axis_B_to_B
        axis_B_to_B /= np.linalg.norm(axis_B_to_B)
        omega_L_to_B = (angle_B_to_B * axis_B_to_B) / dt

    v_rel_B = v_rel + np.cross(-omega_L_to_B, Rot_L_to_B @ sat_pos)
    v_los_s = np.dot(v_rel_B, u_los.T) + np.sum(np.cross(omega - omega_L_to_B, r) * u_los, axis=1)

    Xs, Ys, Zs = useful_rel_locations.T
    if noise_cfg is None:
        noise_cfg = load_config(config_path)["noise"]

    Xn, Yn, Zn, Vn = lidarNoise.apply_noise(
        Xs,
        Ys,
        Zs,
        v_los_s,
        noise_cfg=noise_cfg,
        config_path=config_path,
        return_dropout=False,
    )
    return Xn, Yn, Zn, Vn
