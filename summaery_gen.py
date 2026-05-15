import os
import glob
import numpy as np
import pandas as pd


def compute_vector_rmse(df, est_cols, truth_cols):
    """
    Compute 3D RMSE between estimated and truth vectors over all rows in df.

    RMSE = sqrt(mean(dx^2 + dy^2 + dz^2))
    """
    est = df[est_cols].to_numpy(dtype=float)
    truth = df[truth_cols].to_numpy(dtype=float)
    diff = est - truth
    sq_norm = np.sum(diff**2, axis=1)
    return np.sqrt(np.mean(sq_norm))


def compute_scalar_rmse(df, col):
    """
    Compute scalar RMSE = sqrt(mean(x^2)).
    """
    x = df[col].to_numpy(dtype=float)
    return np.sqrt(np.mean(x**2))


def process_single_file(csv_path, n_skip_valid=200):
    """
    Process one run-history CSV and return a dict with summary metrics.

    Filters:
      1) frame_good == True
      2) exclude first n_skip_valid valid rows after filtering

    Returns:
      dict with identifiers and RMSE values, or None if insufficient data.
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "geometry_name",
        "run_number_for_combo",
        "frame_good",
        "state_est_x", "state_est_y", "state_est_z",
        "truth_p_x", "truth_p_y", "truth_p_z",
        "state_est_vx", "state_est_vy", "state_est_vz",
        "truth_v_x", "truth_v_y", "truth_v_z",
        "state_est_wx", "state_est_wy", "state_est_wz",
        "truth_w_x", "truth_w_y", "truth_w_z",
        "estimate_error",
        "meas_p_raw_x", "meas_p_raw_y", "meas_p_raw_z",
        "meas_p_x", "meas_p_y", "meas_p_z",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    # Robust handling in case frame_good is stored as bool, 0/1, or string
    frame_good = df["frame_good"]
    if frame_good.dtype == bool:
        mask_good = frame_good
    else:
        mask_good = frame_good.astype(str).str.strip().str.lower().isin(
            ["true", "1", "yes", "y"]
        )

    df_good = df.loc[mask_good].copy()

    n_valid_total = len(df_good)
    if n_valid_total <= n_skip_valid:
        print(
            f"Skipping {os.path.basename(csv_path)}: "
            f"only {n_valid_total} valid rows, need > {n_skip_valid}."
        )
        return None

    # Exclude first 200 valid samples
    df_used = df_good.iloc[n_skip_valid:].copy()

    if df_used.empty:
        print(f"Skipping {os.path.basename(csv_path)}: no rows left after filtering.")
        return None

    # Drop rows with NaNs in the metrics we actually need
    metric_cols = [
        "state_est_x", "state_est_y", "state_est_z",
        "truth_p_x", "truth_p_y", "truth_p_z",
        "state_est_vx", "state_est_vy", "state_est_vz",
        "truth_v_x", "truth_v_y", "truth_v_z",
        "state_est_wx", "state_est_wy", "state_est_wz",
        "truth_w_x", "truth_w_y", "truth_w_z",
        "estimate_error",
        "meas_p_raw_x", "meas_p_raw_y", "meas_p_raw_z",
        "meas_p_x", "meas_p_y", "meas_p_z",
    ]
    df_used = df_used.dropna(subset=metric_cols)

    if df_used.empty:
        print(f"Skipping {os.path.basename(csv_path)}: all filtered rows contain NaNs.")
        return None

    # Pull identifiers from the file
    geometry_name = df["geometry_name"].iloc[0]
    run_number_for_combo = df["run_number_for_combo"].iloc[0]

    # Optional traceability / hyperparameter columns if present
    summary = {
        "file_name": os.path.basename(csv_path),
        "geometry_name": geometry_name,
        "run_number_for_combo": run_number_for_combo,
        "n_valid_rows_total": n_valid_total,
        "n_rows_used_after_skip": len(df_used),
    }

    optional_firstrow_cols = [
        "combo_name",
        "combo_index",
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "eig_thresh",
        "bias_recalibration_thresh",
    ]
    for col in optional_firstrow_cols:
        if col in df.columns:
            summary[col] = df[col].iloc[0]

    # State RMSEs
    summary["rmse_position"] = compute_vector_rmse(
        df_used,
        ["state_est_x", "state_est_y", "state_est_z"],
        ["truth_p_x", "truth_p_y", "truth_p_z"],
    )

    summary["rmse_linear_velocity"] = compute_vector_rmse(
        df_used,
        ["state_est_vx", "state_est_vy", "state_est_vz"],
        ["truth_v_x", "truth_v_y", "truth_v_z"],
    )

    summary["rmse_angular_velocity"] = compute_vector_rmse(
        df_used,
        ["state_est_wx", "state_est_wy", "state_est_wz"],
        ["truth_w_x", "truth_w_y", "truth_w_z"],
    )

    # Orientation RMSE: use estimate_error directly
    summary["rmse_orientation"] = compute_scalar_rmse(df_used, "estimate_error")

    # Position measurement RMSE before bias
    summary["rmse_position_before_bias"] = compute_vector_rmse(
        df_used,
        ["meas_p_raw_x", "meas_p_raw_y", "meas_p_raw_z"],
        ["truth_p_x", "truth_p_y", "truth_p_z"],
    )

    # Position measurement RMSE after bias
    summary["rmse_position_after_bias"] = compute_vector_rmse(
        df_used,
        ["meas_p_x", "meas_p_y", "meas_p_z"],
        ["truth_p_x", "truth_p_y", "truth_p_z"],
    )

    return summary


def compile_run_histories(input_folder, output_csv, pattern="*.csv", n_skip_valid=200):
    """
    Compile all matching CSV files in input_folder into one summary CSV.
    """
    search_pattern = os.path.join(input_folder, pattern)
    csv_files = sorted(glob.glob(search_pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching: {search_pattern}")

    results = []
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Processing: {os.path.basename(csv_path)}")
        try:
            row = process_single_file(csv_path, n_skip_valid=n_skip_valid)
            if row is not None:
                results.append(row)
        except Exception as e:
            print(f"Error processing {os.path.basename(csv_path)}: {e}")

    if not results:
        raise RuntimeError("No valid files were processed successfully.")

    summary_df = pd.DataFrame(results)

    # Nice ordering if columns exist
    preferred_order = [
        "file_name",
        "geometry_name",
        "run_number_for_combo",
        "combo_name",
        "combo_index",
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "eig_thresh",
        "bias_recalibration_thresh",
        "n_valid_rows_total",
        "n_rows_used_after_skip",
        "rmse_position",
        "rmse_linear_velocity",
        "rmse_angular_velocity",
        "rmse_orientation",
        "rmse_position_before_bias",
        "rmse_position_after_bias",
    ]
    existing_cols = [c for c in preferred_order if c in summary_df.columns]
    other_cols = [c for c in summary_df.columns if c not in existing_cols]
    summary_df = summary_df[existing_cols + other_cols]

    summary_df.to_csv(output_csv, index=False)
    print(f"\nSaved compiled summary to: {output_csv}")
    print(f"Total rows written: {len(summary_df)}")

    return summary_df


if __name__ == "__main__":
    input_folder =r"to_sync/adaptive_R_res/full_results"
    output_csv = r"compiled_rmse_summary_final_adaptive.csv"

    compile_run_histories(
        input_folder=input_folder,
        output_csv=output_csv,
        pattern="results_of_ass_res_*.csv",
        n_skip_valid=2000,
    )