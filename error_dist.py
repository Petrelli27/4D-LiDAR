import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_combo_file(combo_csv_path):
    df = pd.read_csv(combo_csv_path)

    required_cols = [
        "truth_p_x", "truth_p_y", "truth_p_z",
        "meas_p_x", "meas_p_y", "meas_p_z",
        "truth_q_w", "truth_q_x", "truth_q_y", "truth_q_z",
        "meas_q_w", "meas_q_x", "meas_q_y", "meas_q_z",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def align_measured_quaternion_sign(df):
    """
    Align measured quaternion sign to truth quaternion sign row-by-row.
    If dot(meas_q, truth_q) < 0, flip measured quaternion.
    """
    meas_q = df[["meas_q_w", "meas_q_x", "meas_q_y", "meas_q_z"]].to_numpy(dtype=float)
    truth_q = df[["truth_q_w", "truth_q_x", "truth_q_y", "truth_q_z"]].to_numpy(dtype=float)

    dots = np.sum(meas_q * truth_q, axis=1)
    flip_mask = dots < 0.0

    meas_q_aligned = meas_q.copy()
    meas_q_aligned[flip_mask] *= -1.0

    df = df.copy()
    df["meas_q_w_aligned"] = meas_q_aligned[:, 0]
    df["meas_q_x_aligned"] = meas_q_aligned[:, 1]
    df["meas_q_y_aligned"] = meas_q_aligned[:, 2]
    df["meas_q_z_aligned"] = meas_q_aligned[:, 3]

    return df


def compute_component_errors(df):
    df = df.copy()

    # Position errors
    df["err_p_x"] = pd.to_numeric(df["meas_p_x"], errors="coerce") - pd.to_numeric(df["truth_p_x"], errors="coerce")
    df["err_p_y"] = pd.to_numeric(df["meas_p_y"], errors="coerce") - pd.to_numeric(df["truth_p_y"], errors="coerce")
    df["err_p_z"] = pd.to_numeric(df["meas_p_z"], errors="coerce") - pd.to_numeric(df["truth_p_z"], errors="coerce")

    # Quaternion sign alignment first
    df = align_measured_quaternion_sign(df)

    # Quaternion component errors
    df["err_q_w"] = pd.to_numeric(df["meas_q_w_aligned"], errors="coerce") - pd.to_numeric(df["truth_q_w"], errors="coerce")
    df["err_q_x"] = pd.to_numeric(df["meas_q_x_aligned"], errors="coerce") - pd.to_numeric(df["truth_q_x"], errors="coerce")
    df["err_q_y"] = pd.to_numeric(df["meas_q_y_aligned"], errors="coerce") - pd.to_numeric(df["truth_q_y"], errors="coerce")
    df["err_q_z"] = pd.to_numeric(df["meas_q_z_aligned"], errors="coerce") - pd.to_numeric(df["truth_q_z"], errors="coerce")

    return df


def save_histogram(series, title, xlabel, output_path, bins=50):
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        print(f"Skipping histogram {output_path} because there is no valid data.")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(clean, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram: {output_path}")


def summarize_error_statistics(df, output_csv_path):
    error_cols = [
        "err_p_x", "err_p_y", "err_p_z",
        "err_q_w", "err_q_x", "err_q_y", "err_q_z",
    ]

    rows = []
    for col in error_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) == 0:
            rows.append({
                "error_component": col,
                "count": 0,
                "mean": np.nan,
                "variance": np.nan,
                "std_dev": np.nan,
            })
            continue

        rows.append({
            "error_component": col,
            "count": int(len(values)),
            "mean": float(values.mean()),
            "variance": float(values.var(ddof=1)),
            "std_dev": float(values.std(ddof=1)),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_csv_path, index=False)
    print(f"Saved variance summary: {output_csv_path}")

    return summary_df


def plot_and_save_combo_error_distributions(combo_csv_path, output_folder, bins=50):
    """
    Given one compiled combo CSV:
      - compute component-wise position and quaternion errors
      - save one histogram PNG per component
      - save one CSV summary containing mean / variance / std dev
      - save an augmented CSV with error columns if desired
    """
    os.makedirs(output_folder, exist_ok=True)

    df = load_combo_file(combo_csv_path)
    df = compute_component_errors(df)

    combo_base = os.path.splitext(os.path.basename(combo_csv_path))[0]

    # Save summary CSV
    summary_csv = os.path.join(output_folder, f"{combo_base}__error_variances.csv")
    summary_df = summarize_error_statistics(df, summary_csv)

    # Save augmented CSV with error columns
    augmented_csv = os.path.join(output_folder, f"{combo_base}__with_errors.csv")
    df.to_csv(augmented_csv, index=False)
    print(f"Saved augmented CSV: {augmented_csv}")

    # Individual histogram files
    histogram_specs = [
        ("err_p_x", "Position error distribution: $p_x$", "Error in $p_x$"),
        ("err_p_y", "Position error distribution: $p_y$", "Error in $p_y$"),
        ("err_p_z", "Position error distribution: $p_z$", "Error in $p_z$"),
        ("err_q_w", "Quaternion component error distribution: $q_w$", "Error in $q_w$"),
        ("err_q_x", "Quaternion component error distribution: $q_x$", "Error in $q_x$"),
        ("err_q_y", "Quaternion component error distribution: $q_y$", "Error in $q_y$"),
        ("err_q_z", "Quaternion component error distribution: $q_z$", "Error in $q_z$"),
    ]

    for col, title, xlabel in histogram_specs:
        out_png = os.path.join(output_folder, f"{combo_base}__{col}_hist.png")
        save_histogram(
            series=df[col],
            title=title,
            xlabel=xlabel,
            output_path=out_png,
            bins=bins,
        )

    print("\nVariance summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    combo_csv_path = r"asr_paper_results/compiled_hyperparameter_combo_results/compiled_rpca_20__ortho_0p5__eig_0p24.csv"
    output_folder = r"output_error_distributions"

    plot_and_save_combo_error_distributions(
        combo_csv_path=combo_csv_path,
        output_folder=output_folder,
        bins=50,
    )