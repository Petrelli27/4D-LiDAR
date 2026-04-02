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
    retain a selected subset of columns, and write one compiled CSV per combo
    into dest_folder.

    Filenames are expected to look like:
    results_of_ass_res_results_booster_200s__rpca_5__ortho_0p4__eig_0p04__run_001__sim_debris_trimesh_test_booster_103.csv
    """

    os.makedirs(dest_folder, exist_ok=True)

    # Regex to parse combo from filename
    pattern = re.compile(
        r"^results_of_ass_res_results_(?P<geometry>.+?)"
        r"__rpca_(?P<rpca>[^_]+)"
        r"__ortho_(?P<ortho>[^_]+)"
        r"__eig_(?P<eig>[^_]+)"
        r"__run_(?P<run>\d+)"
        r"__.*\.csv$"
    )

    required_columns = [
        "geometry_name",
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "eig_thresh",
        "metric_stage_name",
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

        combo_key = (
            f"rpca_{match.group('rpca')}__"
            f"ortho_{match.group('ortho')}__"
            f"eig_{match.group('eig')}"
        )

        try:
            df = pd.read_csv(full_path)

            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                print(f"Skipping {fname} because missing columns: {missing}")
                skipped_files += 1
                continue

            df = df.loc[frame_good_mask(df["frame_good"])].copy()

            if df.empty:
                print(f"Processed {fname} | kept 0 frame_good rows")
                processed_files += 1
                continue

            keep_columns = [
                "geometry_name",
                "ransac_pca_threshold",
                "orthonormal_thresh",
                "eig_thresh",
                "metric_stage_name",
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
            ]

            df = df[keep_columns].copy()
            df["source_file"] = fname

            if combo_key not in combo_to_dfs:
                combo_to_dfs[combo_key] = []

            combo_to_dfs[combo_key].append(df)

            print(f"Processed {fname} | kept {len(df)} frame_good rows")
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
    source_folder = r"concord_hyperparameter_results"
    dest_folder = r"compiled_hyperparameter_combo_results"

    compile_results_per_combo(
        source_folder=source_folder,
        dest_folder=dest_folder,
    )