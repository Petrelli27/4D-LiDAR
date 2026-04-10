import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_orientation_error_heatmap(
    file_path,
    output_path="orientation_error_heatmap.png",
    show_plot=True,
    figsize=(9, 6),
    cmap="Blues"
):
    """
    Create a heatmap where:
      - x-axis = ransac_pca_threshold
      - y-axis = orthonormal_thresh
      - cell color = mean_orientation_error_deg
      - cell annotation = discard_pct (shown in black)

    The threshold values are inferred directly from the CSV.
    eig_thresh is ignored.

    Expected CSV columns:
        ransac_pca_threshold
        orthonormal_thresh
        mean_orientation_error_deg
        discard_pct
    """

    # -----------------------------
    # Load and validate
    # -----------------------------
    df = pd.read_csv(file_path)

    required_cols = [
        "ransac_pca_threshold",
        "orthonormal_thresh",
        "mean_orientation_error_deg",
        "discard_pct",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna(subset=required_cols)

    if df.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    # -----------------------------
    # Infer threshold values directly from file
    # -----------------------------
    x_vals = np.array(sorted(df["ransac_pca_threshold"].unique()), dtype=float)
    y_vals = np.array(sorted(df["orthonormal_thresh"].unique()), dtype=float)

    # -----------------------------
    # Aggregate in case there are duplicate rows per (x, y)
    # -----------------------------
    grouped = (
        df.groupby(["orthonormal_thresh", "ransac_pca_threshold"], as_index=False)
          .agg(
              mean_orientation_error_deg=("mean_orientation_error_deg", "mean"),
              discard_pct=("discard_pct", "mean")
          )
    )

    # Heatmap values
    heat = (
        grouped.pivot(
            index="orthonormal_thresh",
            columns="ransac_pca_threshold",
            values="mean_orientation_error_deg"
        )
        .reindex(index=y_vals, columns=x_vals)
    )

    # Annotation values
    discard = (
        grouped.pivot(
            index="orthonormal_thresh",
            columns="ransac_pca_threshold",
            values="discard_pct"
        )
        .reindex(index=y_vals, columns=x_vals)
    )

    # Build annotation strings
    annot = discard.copy()
    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            val = annot.iat[i, j]
            annot.iat[i, j] = "" if pd.isna(val) else f"{val:.1f}%"

    # -----------------------------
    # Color normalization
    # -----------------------------
    finite_vals = heat.to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]

    if finite_vals.size == 0:
        raise ValueError("No finite mean_orientation_error_deg values found.")

    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))

    # Avoid zero dynamic range
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heat,
        cmap=cmap,
        norm=norm,
        annot=annot,
        fmt="",
        annot_kws={"color": "black", "fontsize": 10, "fontweight": "bold"},
        cbar_kws={"label": "Mean Orientation Error [deg]"},
        linewidths=0.5,
        linecolor="white"
    )

    # Put ticks at cell centers and label with actual threshold values
    ax.set_xticks(np.arange(len(x_vals)) + 0.5)
    ax.set_yticks(np.arange(len(y_vals)) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=0)
    ax.set_yticklabels([f"{v:g}" for v in y_vals], rotation=0)

    ax.set_xlabel("RANSAC-PCA Threshold")
    ax.set_ylabel("Orthonormal Threshold")
    ax.set_title("Mean Orientation Error Heatmap")

    # Optional: invert y-axis if you want smaller threshold at the top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    file_path = "quartile_combo_orientation_error_summary.csv"
    plot_orientation_error_heatmap(
        file_path=file_path,
        output_path="orientation_error_heatmap.png",
        show_plot=True,
        figsize=(9, 6),
        cmap="Blues"
    )