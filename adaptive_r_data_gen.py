import os
import glob
import argparse
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "true_range",
    "frame_good",

    "meas_p_x", "meas_p_y", "meas_p_z",
    "truth_p_x", "truth_p_y", "truth_p_z",

    "meas_w_x", "meas_w_y", "meas_w_z",
    "truth_w_x", "truth_w_y", "truth_w_z",

    "meas_q_w", "meas_q_x", "meas_q_y", "meas_q_z",
    "truth_q_w", "truth_q_x", "truth_q_y", "truth_q_z",
]


def robust_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def quaternion_angle_error_deg(q_meas, q_truth, eps=1e-12):
    q_meas = q_meas.astype(float)
    q_truth = q_truth.astype(float)

    q_meas /= np.maximum(np.linalg.norm(q_meas, axis=1, keepdims=True), eps)
    q_truth /= np.maximum(np.linalg.norm(q_truth, axis=1, keepdims=True), eps)

    dots = np.sum(q_meas * q_truth, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    return np.degrees(2.0 * np.arccos(dots))


def align_quaternion_signs(q_meas, q_truth):
    """
    Align measured quaternion sign to truth quaternion sign.

    q and -q represent the same attitude, so this avoids artificial
    component-wise jumps.
    """
    dots = np.sum(q_meas * q_truth, axis=1)
    q_meas_aligned = q_meas.copy()
    q_meas_aligned[dots < 0.0] *= -1.0
    return q_meas_aligned


def make_accumulator(n_bins):
    keys = [
        "pos_err_norm",
        "pos_err_x", "pos_err_y", "pos_err_z",

        "w_err_norm",
        "w_err_x", "w_err_y", "w_err_z",

        "q_angle_err_deg",
        "q_err_w", "q_err_x", "q_err_y", "q_err_z",
    ]

    acc = {"count": np.zeros(n_bins, dtype=np.int64)}

    for key in keys:
        acc[f"{key}_sum"] = np.zeros(n_bins, dtype=float)
        acc[f"{key}_sumsq"] = np.zeros(n_bins, dtype=float)
        acc[f"{key}_count"] = np.zeros(n_bins, dtype=np.int64)

    return acc


def update_stats(acc, bin_idx, values, key):
    values = np.asarray(values, dtype=float)

    finite = np.isfinite(values)
    if not np.any(finite):
        return

    bin_idx = bin_idx[finite]
    values = values[finite]

    for b in np.unique(bin_idx):
        vals = values[bin_idx == b]
        acc[f"{key}_count"][b] += len(vals)
        acc[f"{key}_sum"][b] += vals.sum()
        acc[f"{key}_sumsq"][b] += np.sum(vals ** 2)


def std_from_sum(count, total, total_squares):
    out = np.full_like(total, np.nan, dtype=float)
    valid = count > 1

    mean = np.zeros_like(total, dtype=float)
    mean[valid] = total[valid] / count[valid]

    variance = np.zeros_like(total, dtype=float)
    variance[valid] = (
        total_squares[valid] - count[valid] * mean[valid] ** 2
    ) / (count[valid] - 1)

    variance = np.maximum(variance, 0.0)
    out[valid] = np.sqrt(variance[valid])

    return out

def filter_excluded_files(csv_files, exclude_contains):
    if not exclude_contains:
        return csv_files

    excluded = []
    kept = []

    for path in csv_files:
        name = os.path.basename(path)
        if any(s in name for s in exclude_contains):
            excluded.append(path)
        else:
            kept.append(path)

    print(f"Excluded {len(excluded)} files based on exclude strings.")
    return kept

def compute_range_binned_uncertainties(
    input_folder,
    output_csv,
    bin_size=25.0,
    max_range=200.0,
    exclude_contains=None,
):
    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    csv_files = filter_excluded_files(csv_files, exclude_contains)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_folder} after applying exclusions."
        )

    bin_edges = np.arange(0.0, max_range + bin_size, bin_size)
    n_bins = len(bin_edges) - 1

    acc = make_accumulator(n_bins)

    for i, path in enumerate(csv_files, start=1):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in REQUIRED_COLS)
            df = df.loc[:, ~df.columns.duplicated()]

            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                print(f"Skipping {os.path.basename(path)}; missing columns: {missing}")
                continue

            df = df[REQUIRED_COLS].copy()

            df = df[robust_bool(df["frame_good"])]
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=REQUIRED_COLS)

            if df.empty:
                continue

            true_range = df["true_range"].to_numpy(dtype=float)

            # Bins: [0,25), [25,50), ..., [175,200]
            bin_idx = np.digitize(true_range, bin_edges, right=False) - 1

            # Include exactly max_range in final bin
            bin_idx[true_range == max_range] = n_bins - 1

            valid = (bin_idx >= 0) & (bin_idx < n_bins)

            if not np.any(valid):
                continue

            df = df.iloc[np.flatnonzero(valid)].copy()
            bin_idx = bin_idx[valid]

            for b in np.unique(bin_idx):
                acc["count"][b] += np.sum(bin_idx == b)

            # ==================================================
            # Position measurement error
            # ==================================================
            pos_err_x = df["meas_p_x"].to_numpy(float) - df["truth_p_x"].to_numpy(float)
            pos_err_y = df["meas_p_y"].to_numpy(float) - df["truth_p_y"].to_numpy(float)
            pos_err_z = df["meas_p_z"].to_numpy(float) - df["truth_p_z"].to_numpy(float)

            pos_err_norm = np.sqrt(pos_err_x**2 + pos_err_y**2 + pos_err_z**2)

            update_stats(acc, bin_idx, pos_err_norm, "pos_err_norm")
            update_stats(acc, bin_idx, pos_err_x, "pos_err_x")
            update_stats(acc, bin_idx, pos_err_y, "pos_err_y")
            update_stats(acc, bin_idx, pos_err_z, "pos_err_z")

            # ==================================================
            # Angular velocity measurement error
            # ==================================================
            w_err_x = df["meas_w_x"].to_numpy(float) - df["truth_w_x"].to_numpy(float)
            w_err_y = df["meas_w_y"].to_numpy(float) - df["truth_w_y"].to_numpy(float)
            w_err_z = df["meas_w_z"].to_numpy(float) - df["truth_w_z"].to_numpy(float)

            w_err_norm = np.sqrt(w_err_x**2 + w_err_y**2 + w_err_z**2)

            update_stats(acc, bin_idx, w_err_norm, "w_err_norm")
            update_stats(acc, bin_idx, w_err_x, "w_err_x")
            update_stats(acc, bin_idx, w_err_y, "w_err_y")
            update_stats(acc, bin_idx, w_err_z, "w_err_z")

            # ==================================================
            # Quaternion / orientation measurement error
            # ==================================================
            q_meas = df[
                ["meas_q_w", "meas_q_x", "meas_q_y", "meas_q_z"]
            ].to_numpy(float)

            q_truth = df[
                ["truth_q_w", "truth_q_x", "truth_q_y", "truth_q_z"]
            ].to_numpy(float)

            q_angle_err_deg = quaternion_angle_error_deg(q_meas, q_truth)

            q_meas_aligned = align_quaternion_signs(q_meas, q_truth)

            q_err_w = q_meas_aligned[:, 0] - q_truth[:, 0]
            q_err_x = q_meas_aligned[:, 1] - q_truth[:, 1]
            q_err_y = q_meas_aligned[:, 2] - q_truth[:, 2]
            q_err_z = q_meas_aligned[:, 3] - q_truth[:, 3]

            update_stats(acc, bin_idx, q_angle_err_deg, "q_angle_err_deg")
            update_stats(acc, bin_idx, q_err_w, "q_err_w")
            update_stats(acc, bin_idx, q_err_x, "q_err_x")
            update_stats(acc, bin_idx, q_err_y, "q_err_y")
            update_stats(acc, bin_idx, q_err_z, "q_err_z")

            if i % 100 == 0 or i == len(csv_files):
                print(f"Processed {i}/{len(csv_files)} files")

        except Exception as e:
            print(f"Failed on {path}: {e}")

    count = acc["count"]

    result = pd.DataFrame({
        "range_bin_min_m": bin_edges[:-1],
        "range_bin_max_m": bin_edges[1:],
        "n_rows": count,

        "std_position_error_norm_m": std_from_sum(
            acc["pos_err_norm_count"],
            acc["pos_err_norm_sum"],
            acc["pos_err_norm_sumsq"],
        ),
        "std_position_error_x_m": std_from_sum(
            acc["pos_err_x_count"],
            acc["pos_err_x_sum"],
            acc["pos_err_x_sumsq"],
        ),
        "std_position_error_y_m": std_from_sum(
            acc["pos_err_y_count"],
            acc["pos_err_y_sum"],
            acc["pos_err_y_sumsq"],
        ),
        "std_position_error_z_m": std_from_sum(
            acc["pos_err_z_count"],
            acc["pos_err_z_sum"],
            acc["pos_err_z_sumsq"],
        ),

        "std_angular_velocity_error_norm_rad_s": std_from_sum(
            acc["w_err_norm_count"],
            acc["w_err_norm_sum"],
            acc["w_err_norm_sumsq"],
        ),
        "std_angular_velocity_error_x_rad_s": std_from_sum(
            acc["w_err_x_count"],
            acc["w_err_x_sum"],
            acc["w_err_x_sumsq"],
        ),
        "std_angular_velocity_error_y_rad_s": std_from_sum(
            acc["w_err_y_count"],
            acc["w_err_y_sum"],
            acc["w_err_y_sumsq"],
        ),
        "std_angular_velocity_error_z_rad_s": std_from_sum(
            acc["w_err_z_count"],
            acc["w_err_z_sum"],
            acc["w_err_z_sumsq"],
        ),

        "std_orientation_error_deg": std_from_sum(
            acc["q_angle_err_deg_count"],
            acc["q_angle_err_deg_sum"],
            acc["q_angle_err_deg_sumsq"],
        ),
        "std_quaternion_error_w": std_from_sum(
            acc["q_err_w_count"],
            acc["q_err_w_sum"],
            acc["q_err_w_sumsq"],
        ),
        "std_quaternion_error_x": std_from_sum(
            acc["q_err_x_count"],
            acc["q_err_x_sum"],
            acc["q_err_x_sumsq"],
        ),
        "std_quaternion_error_y": std_from_sum(
            acc["q_err_y_count"],
            acc["q_err_y_sum"],
            acc["q_err_y_sumsq"],
        ),
        "std_quaternion_error_z": std_from_sum(
            acc["q_err_z_count"],
            acc["q_err_z_sum"],
            acc["q_err_z_sumsq"],
        ),
    })

    result.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute range-binned measurement uncertainty statistics."
    )

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing CSV result files.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--bin_size",
        type=float,
        default=25.0,
        help="Range bin size in metres.",
    )
    parser.add_argument(
        "--max_range",
        type=float,
        default=200.0,
        help="Maximum range in metres.",
    )

    args = parser.parse_args()

    compute_range_binned_uncertainties(
        input_folder=args.input_folder,
        output_csv=args.output_csv,
        bin_size=args.bin_size,
        max_range=args.max_range,
    )


if __name__ == "__main__":
    input_folder='to_sync/final_res_final_res/full_results'
    output_csv='range_binned_uncertainties.csv'
    bin_size=25
    max_range=200
    exclude_list = ['results_cube-dish', 'results_cube-single-panel', 'results_cube-tilted-panels',
                    'results_cylinder-four-panels', 'results_cylinder-no-panels-booster',
                    'results_cylinder-two-panel-tilted', 'results_hex-tilded-panels', 'results_kompsat',
                    'results_obsever-cubesat-scaled-v2', ]

    compute_range_binned_uncertainties(
        input_folder=input_folder,
        output_csv=output_csv,
        bin_size=bin_size,
        max_range=max_range,
        exclude_contains=exclude_list
    )