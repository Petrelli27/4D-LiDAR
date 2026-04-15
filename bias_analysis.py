import os
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

COLUMNS_TO_KEEP = [
    "geometry_name",
    "state_est_x", "state_est_y", "state_est_z",
    "truth_p_x", "truth_p_y", "truth_p_z",
    "bias_recalibration_thresh",
    "bias_correction",
    "bias_current",
]

STAGE1_FILENAME = "stage1_extracted.parquet"
STAGE2_STATS_FILENAME = "stage2_bias_effect_summary.csv"
STAGE3_SUMMARY_FILENAME = "stage3_per_file_correction_analysis.csv"
STAGE3_GOOD_FILES_FILENAME = "stage3_good_files.txt"
STAGE3_CLOSEST_FILENAME = "stage3_closest_result.txt"
STAGE2_FIGURE_FILENAME = "stage2_bias_effect_plot.png"


# ============================================================
# Utilities
# ============================================================

def robust_bool(series: pd.Series) -> pd.Series:
    """
    Robust parser for boolean-like columns.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def safe_read_csv(csv_path: str) -> pd.DataFrame:
    """
    Read CSV and remove duplicated columns if needed.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_position_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-row position error columns.
    """
    out = df.copy()
    out["err_x"] = out["state_est_x"] - out["truth_p_x"]
    out["err_y"] = out["state_est_y"] - out["truth_p_y"]
    out["err_z"] = out["state_est_z"] - out["truth_p_z"]
    out["pos_err_3d"] = np.sqrt(out["err_x"]**2 + out["err_y"]**2 + out["err_z"]**2)
    return out


# ============================================================
# Stage 1
# ============================================================

def stage1_extract(input_folder: str, output_folder: str, force: bool = False) -> str:
    """
    Parse all CSVs in input_folder, keep only relevant columns, add source_file,
    and save combined result.
    """
    ensure_dir(output_folder)
    output_path = os.path.join(output_folder, STAGE1_FILENAME)

    if os.path.exists(output_path) and not force:
        print(f"[Stage 1] Using existing file: {output_path}")
        return output_path

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    extracted = []

    for i, csv_file in enumerate(csv_files, start=1):
        try:
            df = safe_read_csv(csv_file)

            missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
            if missing:
                print(f"[Stage 1] Skipping {csv_file} (missing columns: {missing})")
                continue

            sub = df[COLUMNS_TO_KEEP].copy()
            sub["bias_correction"] = robust_bool(sub["bias_correction"])
            sub["source_file"] = os.path.basename(csv_file)

            # Keep row order as file order and make frame index explicit
            sub["row_in_file"] = np.arange(len(sub))

            extracted.append(sub)

            if i % 100 == 0 or i == len(csv_files):
                print(f"[Stage 1] Processed {i}/{len(csv_files)} files")

        except Exception as e:
            print(f"[Stage 1] Failed on {csv_file}: {e}")

    if not extracted:
        raise RuntimeError("No valid CSV files were extracted.")

    combined = pd.concat(extracted, ignore_index=True)
    combined = compute_position_error(combined)

    # Prefer parquet for speed and size
    combined.to_parquet(output_path, index=False)
    print(f"[Stage 1] Saved: {output_path}")
    print(f"[Stage 1] Rows: {len(combined)}")

    return output_path


# ============================================================
# Stage 2
# ============================================================

def stage2_bias_effect(stage1_file: str, output_folder: str, force: bool = False) -> tuple[str, str]:
    """
    For each file:
      - before bias = rows before the first True in bias_correction
      - after bias  = rows from the first True onward
    Then group by bias_recalibration_thresh and plot mean 3D error before/after.
    """
    ensure_dir(output_folder)

    summary_csv = os.path.join(output_folder, STAGE2_STATS_FILENAME)
    figure_path = os.path.join(output_folder, STAGE2_FIGURE_FILENAME)

    if os.path.exists(summary_csv) and os.path.exists(figure_path) and not force:
        print(f"[Stage 2] Using existing outputs:\n  {summary_csv}\n  {figure_path}")
        return summary_csv, figure_path

    df = pd.read_parquet(stage1_file)

    per_file_records = []

    for source_file, g in df.groupby("source_file", sort=False):
        g = g.sort_values("row_in_file").reset_index(drop=True)

        correction_idx = np.flatnonzero(g["bias_correction"].values)
        if len(correction_idx) == 0:
            # No correction event -> skip for before/after comparison
            continue

        first_idx = int(correction_idx[0])

        before = g.iloc[:first_idx]
        after = g.iloc[first_idx:]

        if len(before) == 0 or len(after) == 0:
            continue

        per_file_records.append({
            "source_file": source_file,
            "geometry_name": g["geometry_name"].iloc[0],
            "bias_recalibration_thresh": g["bias_recalibration_thresh"].iloc[0],
            "mean_pos_err_before": before["pos_err_3d"].mean(),
            "mean_pos_err_after": after["pos_err_3d"].mean(),
            "n_before": len(before),
            "n_after": len(after),
            "first_correction_row": first_idx,
        })

    if not per_file_records:
        raise RuntimeError("Stage 2 found no files with a valid first bias correction event.")

    per_file_df = pd.DataFrame(per_file_records)

    grouped = (
        per_file_df
        .groupby("bias_recalibration_thresh", as_index=False)
        .agg(
            mean_pos_err_before=("mean_pos_err_before", "mean"),
            mean_pos_err_after=("mean_pos_err_after", "mean"),
            n_files=("source_file", "nunique"),
        )
        .sort_values("bias_recalibration_thresh")
    )

    grouped.to_csv(summary_csv, index=False)
    print(f"[Stage 2] Saved summary: {summary_csv}")

    plt.figure()
    # plt.plot(
    #     grouped["bias_recalibration_thresh"],
    #     grouped["mean_pos_err_before"],
    #     marker="o",
    #     label="Before first bias correction",
    # )
    plt.plot(
        grouped["bias_recalibration_thresh"],
        grouped["mean_pos_err_after"],
        marker="o"
    )
    plt.xlabel("Bias recalibration threshold (degrees)")
    plt.ylabel("Mean 3D position error (m)")
    # plt.title("Effect of Bias Recalibration Threshold on Position Error")
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print(f"[Stage 2] Saved figure: {figure_path}")

    return summary_csv, figure_path


# ============================================================
# Stage 3
# ============================================================

def stage3_per_correction_analysis(
    stage1_file: str,
    output_folder: str,
    chosen_bias_thresh: float,
    fps: float = 20.0,
    window_seconds: float = 20.0,
    force: bool = False,
) -> tuple[str, str, str]:
    """
    For a chosen bias_recalibration_thresh, inspect each correction event in each file.

    For every correction event:
      compare mean 3D position error in the 20 seconds before vs 20 seconds after.
    A file is considered 'good' if every valid correction event improves error.

    If no file satisfies that, output the closest result:
      the file with the highest number of improving corrections,
      then best improvement ratio,
      then largest net improvement magnitude.
    """
    ensure_dir(output_folder)

    summary_csv = os.path.join(output_folder, STAGE3_SUMMARY_FILENAME)
    good_files_txt = os.path.join(output_folder, STAGE3_GOOD_FILES_FILENAME)
    closest_txt = os.path.join(output_folder, STAGE3_CLOSEST_FILENAME)

    if (
        os.path.exists(summary_csv)
        and os.path.exists(good_files_txt)
        and os.path.exists(closest_txt)
        and not force
    ):
        print(f"[Stage 3] Using existing outputs:\n  {summary_csv}\n  {good_files_txt}\n  {closest_txt}")
        return summary_csv, good_files_txt, closest_txt

    window_n = int(round(window_seconds * fps))
    if window_n <= 0:
        raise ValueError("window_seconds * fps must be positive.")

    df = pd.read_parquet(stage1_file)

    # Float comparisons can be annoying, so use np.isclose
    thresh_mask = np.isclose(df["bias_recalibration_thresh"].astype(float), float(chosen_bias_thresh))
    df = df.loc[thresh_mask].copy()

    if df.empty:
        raise RuntimeError(
            f"No rows found for bias_recalibration_thresh = {chosen_bias_thresh}"
        )

    per_file_results = []
    perfect_files = []

    for source_file, g in df.groupby("source_file", sort=False):
        g = g.sort_values("row_in_file").reset_index(drop=True)

        correction_rows = np.flatnonzero(g["bias_correction"].values)
        if len(correction_rows) == 0:
            continue

        event_records = []

        for corr_idx in correction_rows:
            before_start = corr_idx - window_n
            before_end = corr_idx
            after_start = corr_idx
            after_end = corr_idx + window_n

            # Require full windows on both sides
            if before_start < 0 or after_end > len(g):
                continue

            before_err = g.iloc[before_start:before_end]["pos_err_3d"].mean()
            after_err = g.iloc[after_start:after_end]["pos_err_3d"].mean()

            improved = after_err < before_err
            delta = before_err - after_err  # positive means improvement

            event_records.append({
                "correction_row": int(corr_idx),
                "before_mean_err": float(before_err),
                "after_mean_err": float(after_err),
                "delta_before_minus_after": float(delta),
                "improved": bool(improved),
            })

        if len(event_records) == 0:
            continue

        event_df = pd.DataFrame(event_records)

        n_events = len(event_df)
        n_improved = int(event_df["improved"].sum())
        improve_ratio = n_improved / n_events
        net_delta = event_df["delta_before_minus_after"].sum()
        all_improved = bool(event_df["improved"].all())

        per_file_results.append({
            "source_file": source_file,
            "geometry_name": g["geometry_name"].iloc[0],
            "bias_recalibration_thresh": g["bias_recalibration_thresh"].iloc[0],
            "n_valid_corrections": n_events,
            "n_improved_corrections": n_improved,
            "improve_ratio": improve_ratio,
            "all_valid_corrections_improved": all_improved,
            "net_delta_before_minus_after": net_delta,
            "mean_delta_before_minus_after": event_df["delta_before_minus_after"].mean(),
            "correction_rows": ";".join(map(str, event_df["correction_row"].tolist())),
        })

        if all_improved:
            perfect_files.append(source_file)

    if not per_file_results:
        raise RuntimeError(
            f"No files with valid correction windows were found for threshold {chosen_bias_thresh}."
        )

    per_file_df = pd.DataFrame(per_file_results)
    per_file_df = per_file_df.sort_values(
        by=[
            "all_valid_corrections_improved",
            "n_improved_corrections",
            "improve_ratio",
            "net_delta_before_minus_after",
        ],
        ascending=[False, False, False, False],
    )

    per_file_df.to_csv(summary_csv, index=False)
    print(f"[Stage 3] Saved summary: {summary_csv}")

    with open(good_files_txt, "w", encoding="utf-8") as f:
        if perfect_files:
            f.write("Files where every valid correction improved the 20-second mean position error:\n")
            for name in perfect_files:
                f.write(f"{name}\n")
        else:
            f.write("No files found where every valid correction improved the 20-second mean position error.\n")

    print(f"[Stage 3] Saved good-file list: {good_files_txt}")

    with open(closest_txt, "w", encoding="utf-8") as f:
        if perfect_files:
            f.write("At least one perfect file exists; closest-result file not needed.\n")
        else:
            best = per_file_df.iloc[0]
            f.write("No perfect file found.\n")
            f.write("Closest result:\n")
            f.write(f"source_file: {best['source_file']}\n")
            f.write(f"geometry_name: {best['geometry_name']}\n")
            f.write(f"bias_recalibration_thresh: {best['bias_recalibration_thresh']}\n")
            f.write(f"n_valid_corrections: {best['n_valid_corrections']}\n")
            f.write(f"n_improved_corrections: {best['n_improved_corrections']}\n")
            f.write(f"improve_ratio: {best['improve_ratio']:.4f}\n")
            f.write(f"net_delta_before_minus_after: {best['net_delta_before_minus_after']:.6f}\n")
            f.write(f"mean_delta_before_minus_after: {best['mean_delta_before_minus_after']:.6f}\n")
            f.write(f"correction_rows: {best['correction_rows']}\n")

    print(f"[Stage 3] Saved closest-result file: {closest_txt}")

    return summary_csv, good_files_txt, closest_txt


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Bias-correction staged analysis pipeline")

    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing the 1200 CSV files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder where stage outputs will be stored")

    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3],
                        help="Start from this stage")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if cached outputs exist")

    parser.add_argument("--chosen_bias_thresh", type=float, default=15.0,
                        help="Bias recalibration threshold to inspect in stage 3")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Frames per second for the time-window conversion")
    parser.add_argument("--window_seconds", type=float, default=20.0,
                        help="Seconds before/after each correction for stage 3")

    args = parser.parse_args()

    ensure_dir(args.output_folder)

    stage1_file = os.path.join(args.output_folder, STAGE1_FILENAME)

    if args.start_stage <= 1:
        stage1_file = stage1_extract(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            force=args.force,
        )

    if args.start_stage <= 2:
        if not os.path.exists(stage1_file):
            raise FileNotFoundError(
                f"Stage 1 output not found: {stage1_file}. "
                f"Either run stage 1 first or set --start_stage 1."
            )
        stage2_bias_effect(
            stage1_file=stage1_file,
            output_folder=args.output_folder,
            force=args.force,
        )

    if args.start_stage <= 3:
        if not os.path.exists(stage1_file):
            raise FileNotFoundError(
                f"Stage 1 output not found: {stage1_file}. "
                f"Either run stage 1 first or set --start_stage 1."
            )
        stage3_per_correction_analysis(
            stage1_file=stage1_file,
            output_folder=args.output_folder,
            chosen_bias_thresh=args.chosen_bias_thresh,
            fps=args.fps,
            window_seconds=args.window_seconds,
            force=args.force,
        )


if __name__ == "__main__":
    main()