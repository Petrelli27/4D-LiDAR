import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def lighten_color(color, amount=0.55):
    """
    Lighten a matplotlib color by mixing it with white.
    amount=0 -> original color
    amount=1 -> white
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple(c + (white - c) * amount)


def compute_position_error(df, pred_cols, truth_cols):
    """
    Compute Euclidean position error row-wise:
    ||pred - truth||
    """
    pred = df[pred_cols].to_numpy(dtype=float)
    truth = df[truth_cols].to_numpy(dtype=float)
    return np.linalg.norm(pred - truth, axis=1)


def make_compiled_csv_and_plot(
    root_folder,
    output_csv,
    output_figure=None,
    bin_size=25
):
    """
    1) Search recursively for CSV files whose names contain
       'results_of_sim_debris_trimesh'
    2) Keep only rows with frame_good == True
    3) Compute PCA and RANSAC position errors
    4) Save compiled CSV
    5) Bin by true_range and plot mean position error per bin
    """

    target_substring = "results_of_sim_debris_trimesh"

    required_cols = [
        "true_range",
        "file_name",
        "frame_good",
        "truth_p_x", "truth_p_y", "truth_p_z",
        "meas_pca_p_x", "meas_pca_p_y", "meas_pca_p_z",
        "meas_ransac_p_x", "meas_ransac_p_y", "meas_ransac_p_z",
    ]

    compiled_dfs = []

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".csv") and target_substring in fname:
                full_path = os.path.join(dirpath, fname)

                try:
                    df = pd.read_csv(full_path)

                    missing = [col for col in required_cols if col not in df.columns]
                    if missing:
                        print(f"Skipping {full_path} because missing columns: {missing}")
                        continue

                    # Robust frame_good handling
                    frame_good_mask = df["frame_good"].astype(str).str.strip().str.lower().isin(
                        ["true", "1", "yes"]
                    )

                    df_good = df.loc[frame_good_mask].copy()

                    if df_good.empty:
                        print(f"Processed: {full_path} | kept 0 rows")
                        continue

                    # Convert relevant numeric columns
                    numeric_cols = [
                        "true_range",
                        "truth_p_x", "truth_p_y", "truth_p_z",
                        "meas_pca_p_x", "meas_pca_p_y", "meas_pca_p_z",
                        "meas_ransac_p_x", "meas_ransac_p_y", "meas_ransac_p_z",
                    ]
                    for col in numeric_cols:
                        df_good[col] = pd.to_numeric(df_good[col], errors="coerce")

                    # Drop rows with NaNs in required numeric columns
                    df_good = df_good.dropna(subset=numeric_cols).copy()

                    if df_good.empty:
                        print(f"Processed: {full_path} | kept 0 valid numeric rows")
                        continue

                    # Compute position errors
                    df_good["pca_position_error"] = compute_position_error(
                        df_good,
                        ["meas_pca_p_x", "meas_pca_p_y", "meas_pca_p_z"],
                        ["truth_p_x", "truth_p_y", "truth_p_z"]
                    )

                    df_good["ransac_position_error"] = compute_position_error(
                        df_good,
                        ["meas_ransac_p_x", "meas_ransac_p_y", "meas_ransac_p_z"],
                        ["truth_p_x", "truth_p_y", "truth_p_z"]
                    )

                    compiled = df_good[[
                        "ransac_position_error",
                        "pca_position_error",
                        "true_range",
                        "file_name"
                    ]].copy()

                    compiled["source_csv"] = full_path
                    compiled_dfs.append(compiled)

                    print(f"Processed: {full_path} | kept {len(compiled)} rows")

                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    if compiled_dfs:
        compiled_all = pd.concat(compiled_dfs, ignore_index=True)
    else:
        compiled_all = pd.DataFrame(columns=[
            "ransac_position_error",
            "pca_position_error",
            "true_range",
            "file_name",
            "source_csv"
        ])

    compiled_all.to_csv(output_csv, index=False)
    print(f"\nSaved compiled CSV to: {output_csv}")
    print(f"Total rows: {len(compiled_all)}")

    if compiled_all.empty:
        print("No data available for plotting.")
        return

    # ----- Bin by true_range -----
    min_range = np.floor(compiled_all["true_range"].min() / bin_size) * bin_size
    max_range = np.ceil(compiled_all["true_range"].max() / bin_size) * bin_size
    bin_edges = np.arange(min_range, max_range + bin_size, bin_size)

    compiled_all["range_bin"] = pd.cut(
        compiled_all["true_range"],
        bins=bin_edges,
        right=False,
        include_lowest=True
    )

    grouped = compiled_all.groupby("range_bin", observed=False).agg(
        mean_ransac_position_error=("ransac_position_error", "mean"),
        mean_pca_position_error=("pca_position_error", "mean"),
        count=("true_range", "size")
    ).reset_index()

    grouped = grouped[grouped["count"] > 0].copy()

    if grouped.empty:
        print("No non-empty bins available for plotting.")
        return

    labels = [
        f"{int(interval.left)}–{int(interval.right)}"
        for interval in grouped["range_bin"]
    ]

    x = np.arange(len(grouped))
    width = 0.38

    ransac_edge = "tab:blue"
    pca_edge = "tab:orange"
    ransac_face = lighten_color(ransac_edge, amount=0.55)
    pca_face = lighten_color(pca_edge, amount=0.55)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        grouped["mean_ransac_position_error"],
        width=width,
        label="RANSAC position error",
        edgecolor=ransac_edge,
        color=ransac_face,
        linewidth=1.5
    )

    ax.bar(
        x + width / 2,
        grouped["mean_pca_position_error"],
        width=width,
        label="PCA position error",
        edgecolor=pca_edge,
        color=pca_face,
        linewidth=1.5
    )

    ax.set_xlabel("True range bin (m)")
    ax.set_ylabel("Mean position error")
    ax.set_title(f"Mean position error by {bin_size} m true-range bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    root_folder = r"range_pca_ransac_data"
    output_csv = r"compiled_position_errors.csv"
    output_figure = None

    make_compiled_csv_and_plot(
        root_folder=root_folder,
        output_csv=output_csv,
        output_figure=output_figure,
        bin_size=25
    )