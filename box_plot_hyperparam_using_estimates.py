import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def format_combo_label(rpca, ortho, eig):
    return f"({rpca:g}, {ortho:g}, {eig:g})"


def evenly_spaced_indices(n_items, n_select):
    """
    Return n_select evenly spaced integer indices from [0, n_items-1].
    If n_items <= n_select, return all indices.
    """
    if n_items <= n_select:
        return list(range(n_items))
    return np.linspace(0, n_items - 1, n_select).round().astype(int).tolist()


def select_evenly_spaced_per_quartile(summary_df, n_per_quartile=5):
    """
    Split summary_df (already sorted by mean error ascending) into 4 quartiles,
    then select n_per_quartile evenly spaced combos from each quartile.
    """
    n_total = len(summary_df)
    if n_total == 0:
        return summary_df.copy()

    quartile_chunks = np.array_split(summary_df, 4)
    selected_parts = []

    for q_idx, chunk in enumerate(quartile_chunks, start=1):
        if len(chunk) == 0:
            continue

        idxs = evenly_spaced_indices(len(chunk), n_per_quartile)
        picked = chunk.iloc[idxs].copy()
        picked["quartile"] = q_idx
        selected_parts.append(picked)

    if not selected_parts:
        return summary_df.iloc[0:0].copy()

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values("mean_orientation_error_deg", ascending=True).reset_index(drop=True)
    return selected


def build_quartile_combo_orientation_plot(
    combo_folder,
    n_per_quartile=5,
    output_figure=None,
    output_summary_csv=None,
):
    """
    Read one compiled CSV per combo from combo_folder.

    For each combo:
      - ignore inactive_startup entirely
      - relevant universe = agree_ransac, ortho_ransac, eig_pca, pred_fallback
      - selected orientation error:
            agree_ransac / ortho_ransac -> ransac_error
            eig_pca                     -> pca_error
      - discard % = pred_fallback / (agree_ransac + ortho_ransac + eig_pca + pred_fallback)

    Then:
      - compute mean selected orientation error per combo
      - split combos into quartiles by mean error
      - select n_per_quartile evenly spaced combos from each quartile
      - plot horizontal boxplots with discard annotations
    """

    stage_universe = ["agree_ransac", "ortho_ransac", "eig_pca", "pred_fallback"]
    kept_stages = ["agree_ransac", "ortho_ransac", "eig_pca"]

    combo_summaries = []
    combo_distributions = []

    for fname in sorted(os.listdir(combo_folder)):
        if not fname.endswith(".csv"):
            continue

        full_path = os.path.join(combo_folder, fname)

        try:
            df = pd.read_csv(full_path)
        except Exception as e:
            print(f"Skipping {fname} due to read error: {e}")
            continue

        required_cols = [
            "ransac_pca_threshold",
            "orthonormal_thresh",
            "eig_thresh",
            "metric_stage_name",
            "estimate_error",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Skipping {fname} because missing columns: {missing}")
            continue

        # Ignore inactive_startup completely
        df_stage = df[df["metric_stage_name"].isin(stage_universe)].copy()

        total_count = len(df_stage)
        if total_count == 0:
            print(f"Skipping {fname} because it has 0 rows in the relevant stages")
            continue

        rpca = pd.to_numeric(df["ransac_pca_threshold"], errors="coerce").dropna()
        ortho = pd.to_numeric(df["orthonormal_thresh"], errors="coerce").dropna()
        eig = pd.to_numeric(df["eig_thresh"], errors="coerce").dropna()

        if len(rpca) == 0 or len(ortho) == 0 or len(eig) == 0:
            print(f"Skipping {fname} because combo thresholds could not be parsed")
            continue

        rpca_val = float(rpca.iloc[0])
        ortho_val = float(ortho.iloc[0])
        eig_val = float(eig.iloc[0])

        combo_label = format_combo_label(rpca_val, ortho_val, eig_val)

        # Keep only rows that contribute to selected orientation error
        df_kept = df_stage[df_stage["metric_stage_name"].isin(kept_stages)].copy()
        if df_kept.empty:
            print(f"Skipping {fname} because it has 0 kept rows")
            continue

        df_kept["estimate_error"] = pd.to_numeric(df_kept["estimate_error"], errors="coerce")

        df_kept["selected_orientation_error"] = df_kept["estimate_error"]

        df_kept = df_kept.dropna(subset=["selected_orientation_error"]).copy()
        if df_kept.empty:
            print(f"Skipping {fname} because selected orientation error is all NaN")
            continue

        mean_error = df_kept["selected_orientation_error"].mean()
        discard_pct = 100.0 * (1.0 - len(df_kept) / total_count)

        combo_summaries.append({
            "combo_file": fname,
            "combo_label": combo_label,
            "ransac_pca_threshold": rpca_val,
            "orthonormal_thresh": ortho_val,
            "eig_thresh": eig_val,
            "mean_orientation_error_deg": mean_error,
            "n_kept": len(df_kept),
            "n_total_relevant": total_count,
            "discard_pct": discard_pct,
        })

        dist_df = df_kept[["selected_orientation_error"]].copy()
        dist_df["combo_label"] = combo_label
        dist_df["combo_file"] = fname
        dist_df["discard_pct"] = discard_pct
        combo_distributions.append(dist_df)

        print(
            f"Processed {fname} | kept {len(df_kept)}/{total_count} "
            f"| mean={mean_error:.4f} deg | discard={discard_pct:.1f}%"
        )

    if not combo_summaries:
        print("No valid combo files found.")
        return

    summary_df = pd.DataFrame(combo_summaries)
    summary_df = summary_df.sort_values("mean_orientation_error_deg", ascending=True).reset_index(drop=True)

    if output_summary_csv is not None:
        summary_df.to_csv(output_summary_csv, index=False)
        print(f"Saved combo summary to: {output_summary_csv}")

    selected_summary = select_evenly_spaced_per_quartile(
        summary_df,
        n_per_quartile=n_per_quartile,
    )

    selected_labels = selected_summary["combo_label"].tolist()

    dist_df = pd.concat(combo_distributions, ignore_index=True)
    plot_df = dist_df[dist_df["combo_label"].isin(selected_labels)].copy()

    # Preserve global ascending order on y-axis
    plot_df["combo_label"] = pd.Categorical(
        plot_df["combo_label"],
        categories=selected_labels,
        ordered=True
    )

    plt.figure(figsize=(10, max(6, 0.5 * len(selected_labels))))

    ax = sns.boxplot(
        x="selected_orientation_error",
        y="combo_label",
        data=plot_df,
        orient="h",
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "^",
            "markerfacecolor": "orange",
            "markeredgecolor": "orange",
        },
        medianprops={
            "color": "orange",
            "linewidth": 1,
        },
        boxprops={
            "facecolor": "none",
            "edgecolor": "black",
        },
    )

    ax.set_xlabel("Orientation error [deg]")
    ax.set_ylabel(r"Hyperparameter combination $(\theta_{rp}^{max}, vol_r^{max}, r_{p}^{max})$")
    ax.set_title(
        f"{n_per_quartile} evenly spaced combinations per quartile by mean orientation error"
    )

    # x_max = plot_df["selected_orientation_error"].max()
    # x_min = plot_df["selected_orientation_error"].min()
    x_max = 100
    x_min = 0
    x_span = x_max - x_min if x_max > x_min else 1.0
    x_text = x_max + 0.03 * x_span

    for y_idx, (_, row) in enumerate(selected_summary.iterrows()):
        ax.text(
            x_text,
            y_idx,
            f"-{row['discard_pct']:.0f}%",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_xlim(x_min, x_max + 0.18 * x_span)

    plt.tight_layout()

    if output_figure is not None:
        plt.savefig(output_figure, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output_figure}")

    plt.show()

    print("\nSelected combinations:")
    print(selected_summary[[
        "combo_label",
        "quartile",
        "mean_orientation_error_deg",
        "discard_pct",
        "n_kept",
        "n_total_relevant",
    ]].to_string(index=False))


if __name__ == "__main__":
    combo_folder = r"D:\phd\4d-lidar\to_sync\final_hp_compiled"
    output_figure = r"quartile_combo_orientation_error.png"
    output_summary_csv = r"quartile_combo_orientation_error_summary.csv"

    build_quartile_combo_orientation_plot(
        combo_folder=combo_folder,
        n_per_quartile=5,
        output_figure=output_figure,
        output_summary_csv=output_summary_csv,
    )