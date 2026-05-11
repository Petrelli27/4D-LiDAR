import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def lighten_color(color, amount=0.5):
    """
    Lighten a matplotlib color by mixing it with white.

    amount=0   -> original color
    amount=1   -> white
    """
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)


def plot_binned_mean_errors_from_root(
    root_folder,
    bin_size=25,
    target_substring=None,
    output_csv=None,
    output_figure=None
):
    """
    Search recursively through root_folder for CSV files, compile valid rows,
    bin by true_range, and plot mean RANSAC/PCA error per bin with +/- 1 sigma whiskers.

    Parameters
    ----------
    root_folder : str
        Root folder containing CSV files.
    bin_size : float
        Range bin width in meters.
    target_substring : str or None
        If given, only CSV files whose filename contains this substring are processed.
        If None, all CSV files are processed.
    output_csv : str or None
        If given, saves the compiled dataframe to this CSV.
    output_figure : str or None
        If given, saves the figure to this path.
    """

    required_cols = ["ransac_error", "pca_error", "true_range"]

    compiled_dfs = []

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if not fname.endswith(".csv"):
                continue

            if target_substring is not None and target_substring not in fname:
                continue

            full_path = os.path.join(dirpath, fname)

            try:
                df = pd.read_csv(full_path)

                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    print(f"Skipping {full_path} because missing columns: {missing}")
                    continue

                df = df[required_cols].copy()

                # Ensure numeric
                df["ransac_error"] = pd.to_numeric(df["ransac_error"], errors="coerce")
                df["pca_error"] = pd.to_numeric(df["pca_error"], errors="coerce")
                df["true_range"] = pd.to_numeric(df["true_range"], errors="coerce")

                # Drop invalid rows
                df = df.dropna(subset=required_cols).copy()

                if df.empty:
                    print(f"Processed: {full_path} | kept 0 valid rows")
                    continue

                df["source_csv"] = full_path
                compiled_dfs.append(df)

                print(f"Processed: {full_path} | kept {len(df)} rows")

            except Exception as e:
                print(f"Error reading {full_path}: {e}")

    if not compiled_dfs:
        raise ValueError("No valid CSV rows found in root_folder.")

    compiled_all = pd.concat(compiled_dfs, ignore_index=True)

    if output_csv is not None:
        compiled_all.to_csv(output_csv, index=False)
        print(f"\nSaved compiled CSV to: {output_csv}")

    print(f"\nTotal compiled rows: {len(compiled_all)}")

    # Keep naming as degrees
    compiled_all["ransac_error_deg"] = compiled_all["ransac_error"]
    compiled_all["pca_error_deg"] = compiled_all["pca_error"]

    # Build bin edges
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
        mean_ransac_error_deg=("ransac_error_deg", "mean"),
        std_ransac_error_deg=("ransac_error_deg", "std"),
        mean_pca_error_deg=("pca_error_deg", "mean"),
        std_pca_error_deg=("pca_error_deg", "std"),
        count=("true_range", "size")
    ).reset_index()

    grouped = grouped[grouped["count"] > 0].copy()

    if grouped.empty:
        raise ValueError("No non-empty bins available for plotting.")

    # If a bin has only one sample, std is NaN.
    # For plotting, set those 1-sigma whiskers to zero.
    grouped["std_ransac_error_deg"] = grouped["std_ransac_error_deg"].fillna(0.0)
    grouped["std_pca_error_deg"] = grouped["std_pca_error_deg"].fillna(0.0)

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
        grouped["mean_ransac_error_deg"],
        yerr=grouped["std_ransac_error_deg"],
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        width=width,
        label="RANSAC error",
        edgecolor=ransac_edge,
        color=ransac_face,
        linewidth=1.5
    )

    ax.bar(
        x + width / 2,
        grouped["mean_pca_error_deg"],
        yerr=grouped["std_pca_error_deg"],
        capsize=4,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        width=width,
        label="PCA error",
        edgecolor=pca_edge,
        color=pca_face,
        linewidth=1.5
    )

    ax.set_xlabel("True range bin (m)")
    ax.set_ylabel("Error (deg)")
    ax.set_title(f"Mean RANSAC and PCA error ± 1σ by {bin_size} m true-range bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_figure is not None:
        plt.savefig(output_figure, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_figure}")

    plt.show()


if __name__ == "__main__":
    root_folder = "asr_paper_results/range_pca_ransac_data"

    plot_binned_mean_errors_from_root(
        root_folder=root_folder,
        bin_size=25,
        target_substring=None,
        output_csv=None,
        output_figure=None
    )