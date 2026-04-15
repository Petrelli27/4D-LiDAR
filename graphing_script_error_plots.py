import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _frame_good_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _rmse(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))


def _prepare_df(csv_path: str, fps: float = 20.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "frame_good" not in df.columns:
        raise ValueError("Missing required column: frame_good")
    if "frame" not in df.columns:
        raise ValueError("Missing required column: frame")

    df = df.loc[_frame_good_mask(df["frame_good"])].copy()
    df = df.sort_values("frame").reset_index(drop=True)
    df["time_sec"] = df["frame"] / fps

    if df.empty:
        raise ValueError("No rows remain after filtering frame_good == True")

    return df


def _save_fig(fig, out_path: str, show: bool = False):
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _validate_rmse_start_idx(df: pd.DataFrame, rmse_start_idx: int) -> int:
    if not isinstance(rmse_start_idx, int):
        raise TypeError("rmse_start_idx must be an integer")
    if rmse_start_idx < 0:
        raise ValueError("rmse_start_idx must be >= 0")
    if rmse_start_idx >= len(df):
        raise ValueError(
            f"rmse_start_idx={rmse_start_idx} is out of range for dataframe of length {len(df)}"
        )
    return rmse_start_idx


def plot_position_error(
    df: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    rmse_start_idx: int = 0,
):
    ex = df["state_est_x"] - df["truth_p_x"]
    ey = df["state_est_y"] - df["truth_p_y"]
    ez = df["state_est_z"] - df["truth_p_z"]

    i0 = _validate_rmse_start_idx(df, rmse_start_idx)
    ex_rmse = ex.iloc[i0:]
    ey_rmse = ey.iloc[i0:]
    ez_rmse = ez.iloc[i0:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], ex, label=r"Error $p_x$")
    ax.plot(df["time_sec"], ey, label=r"Error $p_y$")
    ax.plot(df["time_sec"], ez, label=r"Error $p_z$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (m)")
    ax.legend(loc="upper right")

    txt = (
        rf"$p_x$  RMSE={_rmse(ex_rmse):.4f}m" "\n"
        rf"$p_y$  RMSE={_rmse(ey_rmse):.4f}m" "\n"
        rf"$p_z$  RMSE={_rmse(ez_rmse):.4f}m"
    )
    ax.text(0.62, 0.23, txt, transform=ax.transAxes, fontsize=11)

    _save_fig(fig, os.path.join(out_dir, "position_error.svg"), show=show)


def plot_linear_velocity_error(
    df: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    rmse_start_idx: int = 0,
):
    evx = df["state_est_vx"] - df["truth_v_x"]
    evy = df["state_est_vy"] - df["truth_v_y"]
    evz = df["state_est_vz"] - df["truth_v_z"]

    i0 = _validate_rmse_start_idx(df, rmse_start_idx)
    evx_rmse = evx.iloc[i0:]
    evy_rmse = evy.iloc[i0:]
    evz_rmse = evz.iloc[i0:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], evx, label=r"Error $v_{Dx}$")
    ax.plot(df["time_sec"], evy, label=r"Error $v_{Dy}$")
    ax.plot(df["time_sec"], evz, label=r"Error $v_{Dz}$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity Error (m/s)")
    ax.legend(loc="upper right")

    txt = (
        rf"$v_{{Dx}}$  RMSE={_rmse(evx_rmse):.4f}m/s" "\n"
        rf"$v_{{Dy}}$  RMSE={_rmse(evy_rmse):.4f}m/s" "\n"
        rf"$v_{{Dz}}$  RMSE={_rmse(evz_rmse):.4f}m/s"
        # rf"(from sample {i0})"
    )
    ax.text(0.66, 0.24, txt, transform=ax.transAxes, fontsize=11)

    _save_fig(fig, os.path.join(out_dir, "linear_velocity_error.svg"), show=show)


def plot_angular_velocity_error(
    df: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    rmse_start_idx: int = 0,
):
    ewx = df["state_est_wx"] - df["truth_w_x"]
    ewy = df["state_est_wy"] - df["truth_w_y"]
    ewz = df["state_est_wz"] - df["truth_w_z"]

    i0 = _validate_rmse_start_idx(df, rmse_start_idx)
    ewx_rmse = ewx.iloc[i0:]
    ewy_rmse = ewy.iloc[i0:]
    ewz_rmse = ewz.iloc[i0:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], ewx, label=r"Error $\Omega_x$")
    ax.plot(df["time_sec"], ewy, label=r"Error $\Omega_y$")
    ax.plot(df["time_sec"], ewz, label=r"Error $\Omega_z$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity Error (rad/s)")
    ax.legend(loc="upper left")

    txt = (
        rf"$\Omega_x$  RMSE={_rmse(ewx_rmse):.4f} rad/s" "\n"
        rf"$\Omega_y$  RMSE={_rmse(ewy_rmse):.4f} rad/s" "\n"
        rf"$\Omega_z$  RMSE={_rmse(ewz_rmse):.4f} rad/s"
        # rf"(from sample {i0})"
    )
    ax.text(0.62, 0.23, txt, transform=ax.transAxes, fontsize=11)

    _save_fig(fig, os.path.join(out_dir, "angular_velocity_error.svg"), show=show)


def plot_orientation_error(
    df: pd.DataFrame,
    out_dir: str,
    show: bool = False,
    rmse_start_idx: int = 0,
):
    e = df["estimate_error"]

    i0 = _validate_rmse_start_idx(df, rmse_start_idx)
    e_rmse = e.iloc[i0:]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], e)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Screw Angle Error (deg)")
    ax.text(
        0.63,
        0.86,
        f"RMSE={_rmse(e_rmse):.2f} deg",
        transform=ax.transAxes,
        fontsize=11,
    )

    _save_fig(fig, os.path.join(out_dir, "orientation_error.svg"), show=show)


def plot_pz_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["meas_p_z"], label="Computed")
    ax.plot(df["time_sec"], df["state_est_z"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_p_z"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$p_z$ (m)")
    ax.legend(loc="upper right")

    _save_fig(fig, os.path.join(out_dir, "pz_comparison.svg"), show=show)


def plot_px_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["meas_p_x"], label="Computed")
    ax.plot(df["time_sec"], df["state_est_x"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_p_x"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$p_x$ (m)")
    ax.legend(loc="upper right")

    _save_fig(fig, os.path.join(out_dir, "px_comparison.svg"), show=show)


def plot_vdy_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["state_est_vy"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_v_y"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$v_{Dy}$ (m/s)")
    ax.legend(loc="upper right")

    _save_fig(fig, os.path.join(out_dir, "vdy_comparison.svg"), show=show)


def plot_omegaz_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["meas_w_x"], label="Computed")
    ax.plot(df["time_sec"], df["state_est_wx"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_w_x"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Omega_x$ (rad/s)")
    ax.legend(loc="upper right")

    _save_fig(fig, os.path.join(out_dir, "omegay_comparison.svg"), show=show)


def plot_vertex_p1_components(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["state_est_p1_x"], label=r"$p_{1x}$")
    ax.plot(df["time_sec"], df["state_est_p1_y"], label=r"$p_{1y}$")
    ax.plot(df["time_sec"], df["state_est_p1_z"], label=r"$p_{1z}$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Vertex $p_1$ Position (m)")
    ax.legend(loc="best")

    _save_fig(fig, os.path.join(out_dir, "vertex_p1_components.svg"), show=show)


def plot_qw_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["meas_q_w"], label="Computed")
    ax.plot(df["time_sec"], df["state_est_q_w"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_q_w"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$q_w$")
    ax.legend(loc="best")

    _save_fig(fig, os.path.join(out_dir, "qw_comparison.svg"), show=show)


def plot_qz_comparison(df: pd.DataFrame, out_dir: str, show: bool = False):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["meas_q_z"], label="Computed")
    ax.plot(df["time_sec"], df["state_est_q_z"], label="Estimated")
    ax.plot(df["time_sec"], df["truth_q_z"], "--", label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$q_z$")
    ax.legend(loc="best")

    _save_fig(fig, os.path.join(out_dir, "qz_comparison.svg"), show=show)


def plot_size_estimation(
    df: pd.DataFrame,
    out_dir: str,
    true_L: float,
    true_W: float,
    true_D: float,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["time_sec"], df["Le"], label="Length")
    ax.plot(df["time_sec"], df["We"], label="Width")
    ax.plot(df["time_sec"], df["De"], label="Height")

    ax.axhline(true_L, linestyle="--", color="black", label="True")
    ax.axhline(true_W, linestyle="--", color="black")
    ax.axhline(true_D, linestyle="--", color="black")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Size (m)")
    ax.legend(loc="best")

    ax.text(
        1.01, true_L, f"{true_L:.1f}m",
        transform=ax.get_yaxis_transform(),
        va="center", fontsize=11
    )
    ax.text(
        1.01, true_W, f"{true_W:.1f}m",
        transform=ax.get_yaxis_transform(),
        va="center", fontsize=11
    )
    ax.text(
        1.01, true_D, f"{true_D:.1f}m",
        transform=ax.get_yaxis_transform(),
        va="center", fontsize=11
    )

    _save_fig(fig, os.path.join(out_dir, "size_estimation.svg"), show=show)


def make_all_plots(
    csv_path: str,
    out_dir: str,
    fps: float = 20.0,
    true_L: float = 5.3,
    true_W: float = 2.4,
    true_D: float = 1.3,
    rmse_start_idx: int = 0,
    show: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    df = _prepare_df(csv_path, fps=fps)
    _validate_rmse_start_idx(df, rmse_start_idx)

    required_cols = [
        "state_est_x", "state_est_y", "state_est_z",
        "truth_p_x", "truth_p_y", "truth_p_z",
        "state_est_vx", "state_est_vy", "state_est_vz",
        "truth_v_x", "truth_v_y", "truth_v_z",
        "state_est_wx", "state_est_wy", "state_est_wz",
        "truth_w_x", "truth_w_y", "truth_w_z",
        "estimate_error",
        "meas_p_x", "meas_p_z",
        "meas_w_z",
        "state_est_p1_x", "state_est_p1_y", "state_est_p1_z",
        "meas_q_w", "state_est_q_w", "truth_q_w",
        "meas_q_z", "state_est_q_z", "truth_q_z",
        "Le", "We", "De",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    plot_position_error(df, out_dir, show=show, rmse_start_idx=rmse_start_idx)
    plot_linear_velocity_error(df, out_dir, show=show, rmse_start_idx=rmse_start_idx)
    plot_angular_velocity_error(df, out_dir, show=show, rmse_start_idx=rmse_start_idx)
    plot_orientation_error(df, out_dir, show=show, rmse_start_idx=rmse_start_idx)
    plot_px_comparison(df, out_dir, show=show)
    plot_pz_comparison(df, out_dir, show=show)
    plot_vdy_comparison(df, out_dir, show=show)
    plot_omegaz_comparison(df, out_dir, show=show)
    plot_vertex_p1_components(df, out_dir, show=show)
    plot_qw_comparison(df, out_dir, show=show)
    plot_qz_comparison(df, out_dir, show=show)
    plot_size_estimation(df, out_dir, true_L=true_L, true_W=true_W, true_D=true_D, show=show)

    print(f"Saved plots to: {out_dir}")
    print(f"Annotated RMSE values computed from filtered sample index: {rmse_start_idx}")


if __name__ == "__main__":
    file = "results_of_ass_res_results_cube-circular-panels__rpca_15__ortho_0p6__eig_1__brecal_90__run_039__sim_debris_trimesh_test_cube-circular-panels_132.csv"
    csv_path = "D:\\phd\\4d-lidar\\to_sync\\to_sync\\to_sync\\final_res_final_res\\full_results\\" + file
    out_dir = r"figs"

    make_all_plots(
        csv_path=csv_path,
        out_dir=out_dir,
        fps=20.0,
        true_L=5.3,
        true_W=2.4,
        true_D=1.3,
        rmse_start_idx=2000,   # RMSE annotations use samples 50:end
        show=True,           # set False for HPC / batch use
    )