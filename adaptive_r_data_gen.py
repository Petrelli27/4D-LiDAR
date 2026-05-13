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


STAT_KEYS = [
    "pos_err_norm",
    "pos_err_x", "pos_err_y", "pos_err_z",

    "w_err_norm",
    "w_err_x", "w_err_y", "w_err_z",

    "q_angle_err_deg",
    "q_err_w", "q_err_x", "q_err_y", "q_err_z",

    "true_range",
]


def robust_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def normalize_quaternions(q, eps=1e-12):
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)


def align_quaternion_signs(q_meas, q_truth):
    dots = np.sum(q_meas * q_truth, axis=1)
    q_meas_aligned = q_meas.copy()
    q_meas_aligned[dots < 0.0] *= -1.0
    return q_meas_aligned


def quaternion_angle_error_deg(q_meas, q_truth):
    q_meas = normalize_quaternions(q_meas.astype(float))
    q_truth = normalize_quaternions(q_truth.astype(float))

    dots = np.sum(q_meas * q_truth, axis=1)
    dots = np.clip(np.abs(dots), 0.0, 1.0)

    return np.degrees(2.0 * np.arccos(dots))


def filter_excluded_files(csv_files, exclude_contains):
    if not exclude_contains:
        return csv_files

    kept = []
    excluded = []

    for path in csv_files:
        name = os.path.basename(path)
        if any(s in name for s in exclude_contains):
            excluded.append(path)
        else:
            kept.append(path)

    print(f"Excluded {len(excluded)} files based on exclude strings.")
    return kept


def make_accumulator(n_bins):
    acc = {}

    for key in STAT_KEYS:
        acc[f"{key}_count"] = np.zeros(n_bins, dtype=np.int64)
        acc[f"{key}_sum"] = np.zeros(n_bins, dtype=float)
        acc[f"{key}_sumsq"] = np.zeros(n_bins, dtype=float)
        acc[f"{key}_min"] = np.full(n_bins, np.nan, dtype=float)
        acc[f"{key}_max"] = np.full(n_bins, np.nan, dtype=float)

    acc["n_rows"] = np.zeros(n_bins, dtype=np.int64)

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

        vmin = vals.min()
        vmax = vals.max()

        if np.isnan(acc[f"{key}_min"][b]):
            acc[f"{key}_min"][b] = vmin
            acc[f"{key}_max"][b] = vmax
        else:
            acc[f"{key}_min"][b] = min(acc[f"{key}_min"][b], vmin)
            acc[f"{key}_max"][b] = max(acc[f"{key}_max"][b], vmax)


def mean_from_sum(count, total):
    out = np.full_like(total, np.nan, dtype=float)
    valid = count > 0
    out[valid] = total[valid] / count[valid]
    return out


def var_from_sum(count, total, total_squares):
    out = np.full_like(total, np.nan, dtype=float)
    valid = count > 1

    mean = np.zeros_like(total, dtype=float)
    mean[valid] = total[valid] / count[valid]

    var = np.zeros_like(total, dtype=float)
    var[valid] = (total_squares[valid] - count[valid] * mean[valid] ** 2) / (count[valid] - 1)
    var = np.maximum(var, 0.0)

    out[valid] = var[valid]
    return out


def rmse_from_sumsq(count, total_squares):
    out = np.full_like(total_squares, np.nan, dtype=float)
    valid = count > 0
    out[valid] = np.sqrt(total_squares[valid] / count[valid])
    return out


def add_metric_columns(result, acc, key, prefix):
    count = acc[f"{key}_count"]
    total = acc[f"{key}_sum"]
    total_squares = acc[f"{key}_sumsq"]

    var = var_from_sum(count, total, total_squares)

    result[f"mean_{prefix}"] = mean_from_sum(count, total)
    result[f"rmse_{prefix}"] = rmse_from_sumsq(count, total_squares)
    result[f"std_{prefix}"] = np.sqrt(var)
    result[f"var_{prefix}"] = var
    result[f"min_{prefix}"] = acc[f"{key}_min"]
    result[f"max_{prefix}"] = acc[f"{key}_max"]


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

            bin_idx = np.digitize(true_range, bin_edges, right=False) - 1
            bin_idx[true_range == max_range] = n_bins - 1

            valid = (bin_idx >= 0) & (bin_idx < n_bins)

            if not np.any(valid):
                continue

            df = df.iloc[np.flatnonzero(valid)].copy()
            bin_idx = bin_idx[valid]
            true_range = true_range[valid]

            for b in np.unique(bin_idx):
                acc["n_rows"][b] += np.sum(bin_idx == b)

            update_stats(acc, bin_idx, true_range, "true_range")

            pos_err_x = df["meas_p_x"].to_numpy(float) - df["truth_p_x"].to_numpy(float)
            pos_err_y = df["meas_p_y"].to_numpy(float) - df["truth_p_y"].to_numpy(float)
            pos_err_z = df["meas_p_z"].to_numpy(float) - df["truth_p_z"].to_numpy(float)
            pos_err_norm = np.sqrt(pos_err_x**2 + pos_err_y**2 + pos_err_z**2)

            update_stats(acc, bin_idx, pos_err_norm, "pos_err_norm")
            update_stats(acc, bin_idx, pos_err_x, "pos_err_x")
            update_stats(acc, bin_idx, pos_err_y, "pos_err_y")
            update_stats(acc, bin_idx, pos_err_z, "pos_err_z")

            w_err_x = df["meas_w_x"].to_numpy(float) - df["truth_w_x"].to_numpy(float)
            w_err_y = df["meas_w_y"].to_numpy(float) - df["truth_w_y"].to_numpy(float)
            w_err_z = df["meas_w_z"].to_numpy(float) - df["truth_w_z"].to_numpy(float)
            w_err_norm = np.sqrt(w_err_x**2 + w_err_y**2 + w_err_z**2)

            update_stats(acc, bin_idx, w_err_norm, "w_err_norm")
            update_stats(acc, bin_idx, w_err_x, "w_err_x")
            update_stats(acc, bin_idx, w_err_y, "w_err_y")
            update_stats(acc, bin_idx, w_err_z, "w_err_z")

            q_meas = df[["meas_q_w", "meas_q_x", "meas_q_y", "meas_q_z"]].to_numpy(float)
            q_truth = df[["truth_q_w", "truth_q_x", "truth_q_y", "truth_q_z"]].to_numpy(float)

            q_meas = normalize_quaternions(q_meas)
            q_truth = normalize_quaternions(q_truth)

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

    result = pd.DataFrame({
        "range_bin_min_m": bin_edges[:-1],
        "range_bin_max_m": bin_edges[1:],
        "n_rows": acc["n_rows"],
    })

    add_metric_columns(result, acc, "true_range", "true_range_m")

    add_metric_columns(result, acc, "pos_err_norm", "position_error_norm_m")
    add_metric_columns(result, acc, "pos_err_x", "position_error_x_m")
    add_metric_columns(result, acc, "pos_err_y", "position_error_y_m")
    add_metric_columns(result, acc, "pos_err_z", "position_error_z_m")

    add_metric_columns(result, acc, "w_err_norm", "angular_velocity_error_norm_rad_s")
    add_metric_columns(result, acc, "w_err_x", "angular_velocity_error_x_rad_s")
    add_metric_columns(result, acc, "w_err_y", "angular_velocity_error_y_rad_s")
    add_metric_columns(result, acc, "w_err_z", "angular_velocity_error_z_rad_s")

    add_metric_columns(result, acc, "q_angle_err_deg", "orientation_error_deg")
    add_metric_columns(result, acc, "q_err_w", "quaternion_error_w")
    add_metric_columns(result, acc, "q_err_x", "quaternion_error_x")
    add_metric_columns(result, acc, "q_err_y", "quaternion_error_y")
    add_metric_columns(result, acc, "q_err_z", "quaternion_error_z")

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
    input_folder='to_sync/final_R_res/full_results'
    output_csv='range_binned_uncertainties3.csv'
    bin_size=25
    max_range=200
    # exclude_list = ['results_cube-dish', 'results_cube-single-panel', 'results_cube-tilted-panels',
    #                 'results_cylinder-four-panels', 'results_cylinder-no-panels-booster',
    #                 'results_cylinder-two-panel-tilted', 'results_hex-tilded-panels', 'results_kompsat',
    #                 'results_obsever-cubesat-scaled-v2', ]

    exclude_list = []

    compute_range_binned_uncertainties(
        input_folder=input_folder,
        output_csv=output_csv,
        bin_size=bin_size,
        max_range=max_range,
        exclude_contains=exclude_list
    )