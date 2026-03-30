# !/usr/bin/env python3

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from scipy.integrate import solve_ivp


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def CW2(r0, rdot0, omeg, t):
    x0 = r0[0]
    y0 = r0[1]
    z0 = r0[2]
    xdot0 = rdot0[0]
    ydot0 = rdot0[1]
    zdot0 = rdot0[2]

    xt = (4 * x0 + (2 * ydot0) / omeg) + (xdot0 / omeg) * math.sin(omeg * t) - (3 * x0 + (2 * ydot0) / omeg) * math.cos(omeg * t)
    yt = (y0 - (2 * xdot0) / omeg) + ((2 * xdot0) / omeg) * math.cos(omeg * t) + (6 * x0 + (4 * ydot0) / omeg) * math.sin(omeg * t) - (6 * omeg * x0 + 3 * ydot0) * t
    zt = z0 * math.cos(omeg * t) + (zdot0 / omeg) * math.sin(omeg * t)

    xdott = (3 * omeg * x0 + 2 * ydot0) * math.sin(omeg * t) + xdot0 * math.cos(omeg * t)
    ydott = (6 * omeg * x0 + 4 * ydot0) * math.cos(omeg * t) - 2 * xdot0 * math.sin(omeg * t) - (6 * omeg * x0 + 3 * ydot0)
    zdott = zdot0 * math.cos(omeg * t) - z0 * omeg * math.sin(omeg * t)

    return [xt, yt, zt], [xdott, ydott, zdott]


def two_body_ode(t, state, mu):
    r = state[:3]
    v = state[3:]
    r_norm3 = np.dot(r, r) ** 1.5
    accel = -mu / r_norm3 * r
    return np.concatenate([v, accel])


def lvlh_axes(r_c, v_c):
    x_hat = r_c / np.linalg.norm(r_c)
    h = np.cross(r_c, v_c)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat


def eci_to_lvlh(r_c, v_c, r_d_eci, v_d_eci):
    x_hat, y_hat, z_hat = lvlh_axes(r_c, v_c)
    R_lvlh = np.array([x_hat, y_hat, z_hat])
    dr_eci = r_d_eci - r_c
    dv_eci = v_d_eci - v_c
    h_vec = np.cross(r_c, v_c)
    omega_vec = h_vec / np.dot(r_c, r_c)
    dr_lvlh = R_lvlh @ dr_eci
    dv_lvlh = R_lvlh @ (dv_eci - np.cross(omega_vec, dr_eci))
    return dr_lvlh, dv_lvlh


def lvlh_to_eci_state(R_km, omeg, mu, dr_lvlh, dv_lvlh):
    r_c = np.array([R_km, 0.0, 0.0])
    v_circ = math.sqrt(mu / R_km)
    v_c = np.array([0.0, v_circ, 0.0])

    x_hat, y_hat, z_hat = lvlh_axes(r_c, v_c)
    r_d = r_c + x_hat * dr_lvlh[0] + y_hat * dr_lvlh[1] + z_hat * dr_lvlh[2]

    h_vec = np.cross(r_c, v_c)
    omega_vec = h_vec / np.dot(r_c, r_c)

    R_lvlh = np.array([x_hat, y_hat, z_hat])
    dv_eci = R_lvlh.T @ dv_lvlh + np.cross(omega_vec, r_d - r_c)
    v_d = v_c + dv_eci
    return r_c, v_c, r_d, v_d


def _propagate_cw(dt, nframes, r0, rdot0, omeg):
    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []
    ds, vs = [], []
    km_to_m = 1000.0

    for i in range(nframes):
        t = dt * i
        r_vec, rdot_vec = CW2(r0, rdot0, omeg, t)
        d = math.sqrt(r_vec[0] ** 2 + r_vec[1] ** 2 + r_vec[2] ** 2)
        v = math.sqrt(rdot_vec[0] ** 2 + rdot_vec[1] ** 2 + rdot_vec[2] ** 2)
        xs.append(r_vec[0] * km_to_m)
        ys.append(r_vec[1] * km_to_m)
        zs.append(r_vec[2] * km_to_m)
        vxs.append(rdot_vec[0] * km_to_m)
        vys.append(rdot_vec[1] * km_to_m)
        vzs.append(rdot_vec[2] * km_to_m)
        ds.append(d * km_to_m)
        vs.append(v * km_to_m)

    return xs, ys, zs, vxs, vys, vzs, ds, vs


def _propagate_nonlinear(dt, nframes, r0, rdot0, omeg, mu, R, integrator_cfg):
    if R is None:
        R = (mu / omeg ** 2) ** (1.0 / 3.0)

    r0 = np.asarray(r0, dtype=float)
    rdot0 = np.asarray(rdot0, dtype=float)
    r_c0_eci, v_c0_eci, r_d0_eci, v_d0_eci = lvlh_to_eci_state(R, omeg, mu, r0, rdot0)
    state0 = np.concatenate([r_c0_eci, v_c0_eci, r_d0_eci, v_d0_eci])

    def ode(t, state):
        ds_chaser = two_body_ode(t, state[0:6], mu)
        ds_debris = two_body_ode(t, state[6:12], mu)
        return np.concatenate([ds_chaser, ds_debris])

    t_span = (0.0, dt * (nframes - 1))
    t_eval = np.linspace(0.0, dt * (nframes - 1), nframes)

    sol = solve_ivp(
        ode,
        t_span,
        state0,
        method=integrator_cfg.get("method", "RK45"),
        t_eval=t_eval,
        rtol=float(integrator_cfg.get("rtol", 1e-10)),
        atol=float(integrator_cfg.get("atol", 1e-12)),
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    km_to_m = 1000.0
    xs, ys, zs = [], [], []
    vxs, vys, vzs = [], [], []
    ds, vs = [], []

    for i in range(nframes):
        r_chaser = sol.y[0:3, i]
        v_chaser = sol.y[3:6, i]
        r_debris = sol.y[6:9, i]
        v_debris = sol.y[9:12, i]
        dr_lvlh, dv_lvlh = eci_to_lvlh(r_chaser, v_chaser, r_debris, v_debris)
        d = np.linalg.norm(dr_lvlh)
        v = np.linalg.norm(dv_lvlh)
        xs.append(dr_lvlh[0] * km_to_m)
        ys.append(dr_lvlh[1] * km_to_m)
        zs.append(dr_lvlh[2] * km_to_m)
        vxs.append(dv_lvlh[0] * km_to_m)
        vys.append(dv_lvlh[1] * km_to_m)
        vzs.append(dv_lvlh[2] * km_to_m)
        ds.append(d * km_to_m)
        vs.append(v * km_to_m)

    return xs, ys, zs, vxs, vys, vzs, ds, vs


def propagate(
    dt,
    nframes,
    r0,
    rdot0,
    omeg,
    R=None,
    mu=None,
    config_path: Optional[str] = None,
):
    cfg = load_config(config_path)
    dyn_cfg = cfg["dynamics"]

    if mu is None:
        mu = float(dyn_cfg.get("mu_km3_s2", 398600.5))

    orbit_cfg = dyn_cfg.get("orbit", {})
    if R is None and not orbit_cfg.get("derive_radius_from_mean_motion", True):
        R = orbit_cfg.get("radius_km")

    model = dyn_cfg.get("model", "nonlinear_two_body")
    if model == "cw":
        return _propagate_cw(dt, nframes, r0, rdot0, omeg)
    if model == "nonlinear_two_body":
        return _propagate_nonlinear(dt, nframes, r0, rdot0, omeg, mu, R, dyn_cfg.get("integrator", {}))
    raise ValueError("dynamics.model must be 'nonlinear_two_body' or 'cw'")
