import numpy as np
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_global_pointing_omega(noise_cfg: Dict[str, Any]) -> np.ndarray:
    omega_xyz_deg = noise_cfg.get("pointing", {}).get("omega_xyz_deg")
    if omega_xyz_deg is not None:
        return np.deg2rad(np.asarray(omega_xyz_deg, dtype=float))

    point_sigma_deg = float(noise_cfg.get("point_sigma_deg", 0.002))
    angular_mode = noise_cfg.get("angular_mode", "gaussian")
    sig = np.deg2rad(point_sigma_deg)

    if angular_mode == "gaussian":
        return np.random.normal(0.0, sig, size=3)
    if angular_mode == "uniform":
        return np.random.uniform(-sig, sig, size=3)
    raise ValueError("noise.angular_mode must be 'gaussian' or 'uniform'")


def _compute_sigmas(distances_m: np.ndarray, noise_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    base_range_sigma = float(noise_cfg.get("range_sigma_m", 0.02))
    base_vel_sigma = float(noise_cfg.get("vel_sigma_m_s", 0.03))

    range_dep_cfg = noise_cfg.get("range_dependent", {})
    enabled = bool(range_dep_cfg.get("enabled", True))
    if not enabled:
        return (
            np.full_like(distances_m, base_range_sigma, dtype=float),
            np.full_like(distances_m, base_vel_sigma, dtype=float),
        )

    reference_range_m = float(range_dep_cfg.get("reference_range_m", 10.0))
    slope_m_per_m = float(range_dep_cfg.get("slope_m_per_m", 0.002))
    excess_range = np.maximum(distances_m - reference_range_m, 0.0)
    sigma_range = base_range_sigma + slope_m_per_m * excess_range
    sigma_vel = base_vel_sigma + slope_m_per_m * excess_range
    return sigma_range, sigma_vel


def _apply_point_dropout(
    X_n: np.ndarray,
    Y_n: np.ndarray,
    Z_n: np.ndarray,
    V_n: np.ndarray,
    noise_cfg: Dict[str, Any],
    return_dropout: bool,
):
    dropout_cfg = noise_cfg.get("point_dropout", {})
    enabled = bool(dropout_cfg.get("enabled", False))
    p_drop_frame = 0.0

    if enabled:
        alpha = float(dropout_cfg.get("alpha", 1.0))
        beta = float(dropout_cfg.get("beta", 20.0))
        if alpha <= 0 or beta <= 0:
            raise ValueError("noise.point_dropout alpha and beta must be > 0")
        p_drop_frame = np.random.beta(alpha, beta)
        keep_mask = np.random.random(size=X_n.shape[0]) > p_drop_frame
        X_n = X_n[keep_mask]
        Y_n = Y_n[keep_mask]
        Z_n = Z_n[keep_mask]
        V_n = V_n[keep_mask]

    if return_dropout:
        return X_n, Y_n, Z_n, V_n, p_drop_frame
    return X_n, Y_n, Z_n, V_n


def apply_noise(
    X,
    Y,
    Z,
    V,
    noise_cfg: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    return_dropout: bool = False,
):
    """
    Apply config-driven LiDAR noise.

    Expected config structure: config["noise"].
    """
    if noise_cfg is None:
        cfg = load_config(config_path)
        noise_cfg = cfg["noise"]

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    V = np.asarray(V, dtype=float)

    if not (X.shape == Y.shape == Z.shape == V.shape):
        raise ValueError("X, Y, Z, and V must have the same shape")

    p = np.column_stack((X, Y, Z))
    d = np.linalg.norm(p, axis=1)
    d_safe = np.where(d == 0.0, 1e-12, d)
    u = p / d_safe[:, None]

    omega = _sample_global_pointing_omega(noise_cfg)
    u_tilted = u + np.cross(omega[None, :], u)
    u_tilted /= np.linalg.norm(u_tilted, axis=1, keepdims=True)

    sigma_range_i, sigma_vel_i = _compute_sigmas(d, noise_cfg)
    d_noisy = np.maximum(d + np.random.normal(0.0, sigma_range_i, size=d.shape), 0.0)
    V_n = V + np.random.normal(0.0, sigma_vel_i, size=V.shape)

    p_noisy = u_tilted * d_noisy[:, None]
    X_n = p_noisy[:, 0]
    Y_n = p_noisy[:, 1]
    Z_n = p_noisy[:, 2]

    return _apply_point_dropout(X_n, Y_n, Z_n, V_n, noise_cfg, return_dropout)
