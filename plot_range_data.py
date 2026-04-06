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


def plot_binned_mean_errors(csv_path, bin_size=25):
    """
    Read compiled CSV, bin by true_range, and plot mean ransac/pca error per bin.

    Parameters
    ----------
    csv_path : str
        Path to compiled CSV.
    bin_size : float
        Range bin width in meters.
    """

    df = pd.read_csv(csv_path)

    required_cols = ["ransac_error", "pca_error", "true_range"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure numeric
    df["ransac_error"] = pd.to_numeric(df["ransac_error"], errors="coerce")
    df["pca_error"] = pd.to_numeric(df["pca_error"], errors="coerce")
    df["true_range"] = pd.to_numeric(df["true_range"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["ransac_error", "pca_error", "true_range"]).copy()

    if len(df) == 0:
        raise ValueError("No valid rows found after cleaning.")

    # Convert errors to degrees
    df["ransac_error_deg"] = df["ransac_error"]
    df["pca_error_deg"] = df["pca_error"]

    # Build bin edges
    min_range = np.floor(df["true_range"].min() / bin_size) * bin_size
    max_range = np.ceil(df["true_range"].max() / bin_size) * bin_size
    bin_edges = np.arange(min_range, max_range + bin_size, bin_size)

    # Bin the ranges
    df["range_bin"] = pd.cut(
        df["true_range"],
        bins=bin_edges,
        right=False,
        include_lowest=True
    )

    # Group and compute mean
    grouped = df.groupby("range_bin", observed=False).agg(
        mean_ransac_error_deg=("ransac_error_deg", "mean"),
        mean_pca_error_deg=("pca_error_deg", "mean"),
        count=("true_range", "size")
    ).reset_index()

    # Remove empty bins
    grouped = grouped[grouped["count"] > 0].copy()

    # X labels like [0,25), [25,50), ...
    labels = [
        f"{int(interval.left)}–{int(interval.right)}"
        for interval in grouped["range_bin"]
    ]

    x = np.arange(len(grouped))
    width = 0.38

    # Base colors
    ransac_edge = "tab:blue"
    pca_edge = "tab:orange"
    ransac_face = lighten_color(ransac_edge, amount=0.55)
    pca_face = lighten_color(pca_edge, amount=0.55)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width / 2,
        grouped["mean_ransac_error_deg"],
        width=width,
        label="RANSAC error",
        edgecolor=ransac_edge,
        color=ransac_face,
        linewidth=1.5
    )

    ax.bar(
        x + width / 2,
        grouped["mean_pca_error_deg"],
        width=width,
        label="PCA error",
        edgecolor=pca_edge,
        color=pca_face,
        linewidth=1.5
    )

    ax.set_xlabel("True range bin (m)")
    ax.set_ylabel("Mean error (deg)")
    ax.set_title("Mean RANSAC and PCA error by 25 m true-range bin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_path = r"range_data_compiled_results.csv"
    plot_binned_mean_errors(csv_path, bin_size=25)