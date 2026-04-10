import os
import re
import pandas as pd


def frame_good_mask(series):
    """
    Robust boolean parser for frame_good column.
    Accepts True/False, 1/0, yes/no, etc.
    """
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def compile_results_per_combo(source_folder, dest_folder):
    """
    Read all per-run results CSVs in source_folder, keep only frame_good rows,
    retain a selected subset of columns, and write one compiled CSV per
    (rpca, ortho) combo into dest_folder.

    Expected filename format:
    results_of_ass_res_results_<geometry>__rpca_<rpca>__ortho_<ortho>__eig_<eig>__run_<run>__*.csv
    """

    os.makedirs(dest_folder, exist_ok=True)

    pattern = re.compile(
        r"^results_of_ass_res_results_(?P<geometry>.+?)"
        r"__rpca_(?P<rpca>.+?)"
        r"__ortho_(?P<ortho>.+?)"
        r"__eig_(?P<eig>.+?)"
        r"__run_(?P<run>\d+)"
        r"__.*\.csv$"
    )

    required_columns = [
        "geometry_name",
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "eig_thresh",
        "metric_stage_name",
        "estimate_error",
        "ransac_error",
        "pca_error",
        "truth_p_x",
        "truth_p_y",
        "truth_p_z",
        "meas_p_x",
        "meas_p_y",
        "meas_p_z",
        "meas_q_w",
        "meas_q_x",
        "meas_q_y",
        "meas_q_z",
        "truth_q_w",
        "truth_q_x",
        "truth_q_y",
        "truth_q_z",
        "frame_good",
        "meas_w_x",
        "meas_w_y",
        "meas_w_z",
        "truth_w_x",
        "truth_w_y",
        "truth_w_z",
    ]

    keep_columns = [
        "geometry_name",
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "eig_thresh",
        "metric_stage_name",
        "estimate_error",
        "ransac_error",
        "pca_error",
        "truth_p_x",
        "truth_p_y",
        "truth_p_z",
        "meas_p_x",
        "meas_p_y",
        "meas_p_z",
        "meas_q_w",
        "meas_q_x",
        "meas_q_y",
        "meas_q_z",
        "truth_q_w",
        "truth_q_x",
        "truth_q_y",
        "truth_q_z",
        "meas_w_x",
        "meas_w_y",
        "meas_w_z",
        "truth_w_x",
        "truth_w_y",
        "truth_w_z",
    ]

    combo_to_dfs = {}
    processed_files = 0
    skipped_files = 0

    for fname in sorted(os.listdir(source_folder)):
        if not fname.endswith(".csv"):
            continue

        match = pattern.match(fname)
        if not match:
            print(f"Skipping file with unexpected name format: {fname}")
            skipped_files += 1
            continue

        full_path = os.path.join(source_folder, fname)

        rpca = match.group("rpca")
        ortho = match.group("ortho")

        combo_key = f"rpca_{rpca}__ortho_{ortho}"

        try:
            df = pd.read_csv(full_path)

            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Skipping {fname} because missing columns: {missing}")
                skipped_files += 1
                continue

            df = df.loc[frame_good_mask(df["frame_good"])].copy()

            if df.empty:
                print(f"Processed {fname} | combo={combo_key} | kept 0 frame_good rows")
                processed_files += 1
                continue

            df = df[keep_columns].copy()
            df["source_file"] = fname
            df["combo_key"] = combo_key

            if combo_key not in combo_to_dfs:
                combo_to_dfs[combo_key] = []

            combo_to_dfs[combo_key].append(df)

            print(f"Processed {fname} | combo={combo_key} | kept {len(df)} frame_good rows")
            processed_files += 1

        except Exception as e:
            print(f"Error reading {fname}: {e}")
            skipped_files += 1

    print("\nWriting compiled combo files...")

    for combo_key, dfs in combo_to_dfs.items():
        if not dfs:
            continue

        compiled = pd.concat(dfs, ignore_index=True)

        out_name = f"compiled_{combo_key}.csv"
        out_path = os.path.join(dest_folder, out_name)

        compiled.to_csv(out_path, index=False)
        print(f"Wrote {out_path} | rows={len(compiled)}")

    print("\nDone.")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files  : {skipped_files}")
    print(f"Output combos  : {len(combo_to_dfs)}")


if __name__ == "__main__":
    source_folder = r"D:\phd\4d-lidar\to_sync\final_res_brecal_30_hp"
    dest_folder = r"compiled_hyperparameter_combo_results_final"

    compile_results_per_combo(
        source_folder=source_folder,
        dest_folder=dest_folder,
    )