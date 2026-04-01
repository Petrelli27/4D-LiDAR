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


def frame_good_mask(series):
    """
    Robust boolean parser for frame_good column.
    """
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_orientation_stage_plot(
    root_folder,
    output_csv,
    output_figure=None,
):
    """
    Recursively search for matching CSV files, keep only frame_good rows,
    compile ransac_error / pca_error / metric_stage_name, and generate
    the staged bar chart requested.

    Stages considered:
      - agree_ransac
      - ortho_ransac
      - eig_pca
      - pred_fallback

    Plot categories:
      1) Raw Measurements
         - mean ransac_error over all 4 stages
         - mean pca_error over all 4 stages
         - discarded % = 0
      2) RANSAC and PCA Agreement
         - mean ransac_error over agree_ransac
         - discarded % based on only keeping agree_ransac
      3) + RANSAC Orthonormality
         - mean ransac_error over agree_ransac + ortho_ransac
         - discarded % based on keeping those two
      4) + Distinct Eigenvalues
         - mean mixed error:
             ransac_error for agree_ransac and ortho_ransac
             pca_error for eig_pca
         - discarded % based on keeping agree_ransac + ortho_ransac + eig_pca
    """

    target_substring = "results_of_sim_debris_trimesh"
    stages_of_interest = ["agree_ransac", "ortho_ransac", "eig_pca", "pred_fallback"]

    required_cols = [
        "frame_good",
        "ransac_error",
        "pca_error",
        "metric_stage_name",
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

                    df = df.loc[frame_good_mask(df["frame_good"])].copy()

                    if df.empty:
                        print(f"Processed: {full_path} | kept 0 frame_good rows")
                        continue

                    df = df[df["metric_stage_name"].isin(stages_of_interest)].copy()

                    if df.empty:
                        print(f"Processed: {full_path} | no rows in stages of interest")
                        continue

                    df["ransac_error"] = pd.to_numeric(df["ransac_error"], errors="coerce")
                    df["pca_error"] = pd.to_numeric(df["pca_error"], errors="coerce")

                    df = df.dropna(subset=["ransac_error", "pca_error", "metric_stage_name"]).copy()

                    if df.empty:
                        print(f"Processed: {full_path} | no valid numeric rows")
                        continue

                    df["source_csv"] = full_path
                    compiled_dfs.append(df[[
                        "ransac_error",
                        "pca_error",
                        "metric_stage_name",
                        "source_csv"
                    ]].copy())

                    print(f"Processed: {full_path} | kept {len(df)} rows")

                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    if compiled_dfs:
        compiled = pd.concat(compiled_dfs, ignore_index=True)
    else:
        compiled = pd.DataFrame(columns=[
            "ransac_error", "pca_error", "metric_stage_name", "source_csv"
        ])

    compiled.to_csv(output_csv, index=False)
    print(f"\nSaved compiled CSV to: {output_csv}")
    print(f"Total rows: {len(compiled)}")

    if compiled.empty:
        print("No data available for plotting.")
        return

    # ---- Counts by stage ----
    stage_counts = compiled["metric_stage_name"].value_counts()
    n_agree = int(stage_counts.get("agree_ransac", 0))
    n_ortho = int(stage_counts.get("ortho_ransac", 0))
    n_eig = int(stage_counts.get("eig_pca", 0))
    n_pred = int(stage_counts.get("pred_fallback", 0))
    n_total = n_agree + n_ortho + n_eig + n_pred

    if n_total == 0:
        print("No rows found in the stages of interest.")
        return

    # ---- Category 1: Raw Measurements ----
    raw_ransac_mean = compiled["ransac_error"].mean()
    raw_pca_mean = compiled["pca_error"].mean()
    raw_discard_pct = 0.0

    # ---- Category 2: RANSAC and PCA Agreement ----
    df_agree = compiled[compiled["metric_stage_name"] == "agree_ransac"].copy()
    agree_error_mean = df_agree["ransac_error"].mean() if not df_agree.empty else np.nan
    agree_keep = len(df_agree)
    agree_discard_pct = 100.0 * (1.0 - agree_keep / n_total)

    # ---- Category 3: + RANSAC Orthonormality ----
    df_agree_ortho = compiled[compiled["metric_stage_name"].isin([
        "agree_ransac", "ortho_ransac"
    ])].copy()
    agree_ortho_error_mean = (
        df_agree_ortho["ransac_error"].mean() if not df_agree_ortho.empty else np.nan
    )
    agree_ortho_keep = len(df_agree_ortho)
    agree_ortho_discard_pct = 100.0 * (1.0 - agree_ortho_keep / n_total)

    # ---- Category 4: + Distinct Eigenvalues ----
    df_final = compiled[compiled["metric_stage_name"].isin([
        "agree_ransac", "ortho_ransac", "eig_pca"
    ])].copy()

    if not df_final.empty:
        df_final["selected_error"] = np.where(
            df_final["metric_stage_name"] == "eig_pca",
            df_final["pca_error"],
            df_final["ransac_error"]
        )
        final_error_mean = df_final["selected_error"].mean()
    else:
        final_error_mean = np.nan

    final_keep = len(df_final)
    final_discard_pct = 100.0 * (1.0 - final_keep / n_total)

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(13, 7))
    ax2 = ax1.twinx()

    # Colors
    ransac_color = "tab:blue"
    pca_color = "tab:orange"
    discard_color = "tab:green"

    ransac_face = lighten_color(ransac_color, amount=0.60)
    pca_face = lighten_color(pca_color, amount=0.60)
    discard_face = lighten_color(discard_color, amount=0.60)

    # Make axis line/ticks/label colors match requested style
    ax1.spines["left"].set_color(ransac_color)
    ax1.tick_params(axis="y", colors=ransac_color)
    ax1.yaxis.label.set_color(ransac_color)

    ax2.spines["right"].set_color(discard_color)
    ax2.tick_params(axis="y", colors=discard_color)
    ax2.yaxis.label.set_color(discard_color)

    category_centers = np.array([0.0, 1.8, 3.6, 5.4])
    width = 0.28

    # Category 1: Raw Measurements -> 3 bars
    c0 = category_centers[0]
    b1 = ax1.bar(
        c0 - width, raw_ransac_mean, width=width,
        edgecolor=ransac_color, color=ransac_face, linewidth=1.8,
        label="Raw RANSAC mean error"
    )
    b2 = ax1.bar(
        c0, raw_pca_mean, width=width,
        edgecolor=pca_color, color=pca_face, linewidth=1.8,
        label="Raw PCA mean error"
    )
    b3 = ax2.bar(
        c0 + width, raw_discard_pct, width=width,
        edgecolor=discard_color, color=discard_face, linewidth=1.8,
        label="Discarded (%)"
    )

    # Categories 2-4: 2 bars each
    error_means = [agree_error_mean, agree_ortho_error_mean, final_error_mean]
    discard_pcts = [agree_discard_pct, agree_ortho_discard_pct, final_discard_pct]
    error_labels = [
        "Selected mean error",
        "Selected mean error",
        "Selected mean error",
    ]

    for i, (err_val, disc_val) in enumerate(zip(error_means, discard_pcts), start=1):
        c = category_centers[i]
        ax1.bar(
            c - width / 2, err_val, width=width,
            edgecolor=ransac_color, color=ransac_face, linewidth=1.8,
            label=error_labels[i - 1] if i == 1 else None
        )
        ax2.bar(
            c + width / 2, disc_val, width=width,
            edgecolor=discard_color, color=discard_face, linewidth=1.8,
            label="Discarded (%)" if i == 1 else None
        )

    # Labels and formatting
    ax1.set_xticks(category_centers)
    ax1.set_xticklabels([
        "Raw Measurements",
        "RANSAC and PCA\nAgreement",
        "+ RANSAC\nOrthonormality",
        "+ Distinct\nEigenvalues"
    ])

    ax1.set_ylabel("Mean orientation error (deg)")
    ax2.set_ylabel("Discarded points (%)")
    ax1.set_title("Orientation error and discarded percentage by metric stage grouping")

    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    # Right axis limits a bit padded
    ax2.set_ylim(0, max(100, raw_discard_pct, agree_discard_pct, agree_ortho_discard_pct, final_discard_pct) * 1.08)

    # Combined legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, edgecolor=ransac_color, facecolor=ransac_face, linewidth=1.8),
        plt.Rectangle((0, 0), 1, 1, edgecolor=pca_color, facecolor=pca_face, linewidth=1.8),
        plt.Rectangle((0, 0), 1, 1, edgecolor=discard_color, facecolor=discard_face, linewidth=1.8),
    ]
    labels = [
        "RANSAC-based mean error",
        "PCA-based mean error",
        "Discarded (%)",
    ]
    ax1.legend(handles, labels, loc="upper right")

    plt.tight_layout()

    if output_figure is not None:
        plt.savefig(output_figure, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_figure}")

    plt.show()

    # Optional console summary
    print("\nStage counts:")
    print(f"  agree_ransac   : {n_agree}")
    print(f"  ortho_ransac   : {n_ortho}")
    print(f"  eig_pca        : {n_eig}")
    print(f"  pred_fallback  : {n_pred}")
    print(f"  total          : {n_total}")

    print("\nPlotted values:")
    print(f"  Raw Measurements -> raw RANSAC mean error: {raw_ransac_mean:.6f} deg")
    print(f"  Raw Measurements -> raw PCA mean error   : {raw_pca_mean:.6f} deg")
    print(f"  Raw Measurements -> discarded            : {raw_discard_pct:.2f}%")
    print(f"  Agreement only  -> mean error            : {agree_error_mean:.6f} deg")
    print(f"  Agreement only  -> discarded             : {agree_discard_pct:.2f}%")
    print(f"  + Ortho         -> mean error            : {agree_ortho_error_mean:.6f} deg")
    print(f"  + Ortho         -> discarded             : {agree_ortho_discard_pct:.2f}%")
    print(f"  + Eig           -> mean error            : {final_error_mean:.6f} deg")
    print(f"  + Eig           -> discarded             : {final_discard_pct:.2f}%")


if __name__ == "__main__":
    root_folder = r"concord_data_200m_200s"
    output_csv = r"compiled_orientation_stage_data.csv"
    output_figure = None

    build_orientation_stage_plot(
        root_folder=root_folder,
        output_csv=output_csv,
        output_figure=output_figure,
    )