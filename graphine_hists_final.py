import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def get_geometry_from_filename(filename):
    """
    Extract geometry from filenames like:
    results_of_ass_res_results_cube-dish__rpca_15__ortho_0p6__eig_1__brecal_90__run_030__sim_debris_trimesh_test_cube-dish_124.csv
    """
    base = os.path.basename(filename)
    m = re.match(r"results_of_ass_res_results_(.*?)__rpca_", base)
    if not m:
        raise ValueError(f"Could not parse geometry from filename: {filename}")
    return m.group(1)


def plot_estimate_error_histograms(input_folder, output_folder, bins_all=60, bins_runmean=30):
    """
    For each geometry:
      1) Plot histogram of all estimate_error values across all frames and runs
      2) Plot histogram of mean estimate_error per run

    Uses ALL frames, not just frame_good.
    """

    os.makedirs(output_folder, exist_ok=True)

    # geometry -> {"all_errors": [...], "run_means": [...]}
    geometry_data = {}

    csv_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".csv")
    ]

    for fname in csv_files:
        fpath = os.path.join(input_folder, fname)

        try:
            geometry = get_geometry_from_filename(fname)
        except ValueError as e:
            print(f"Skipping file: {e}")
            continue

        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"Skipping unreadable file {fname}: {e}")
            continue

        if "estimate_error" not in df.columns:
            print(f"Skipping {fname}: missing 'estimate_error' column.")
            continue

        # Use all frames; just drop NaNs/non-numeric
        errors = pd.to_numeric(df["estimate_error"], errors="coerce").dropna()

        if errors.empty:
            print(f"Skipping {fname}: no valid estimate_error values.")
            continue

        if geometry not in geometry_data:
            geometry_data[geometry] = {
                "all_errors": [],
                "run_means": []
            }

        geometry_data[geometry]["all_errors"].extend(errors.tolist())
        geometry_data[geometry]["run_means"].append(errors.mean())

    # Create plots per geometry
    for geometry, data in geometry_data.items():
        all_errors = data["all_errors"]
        run_means = data["run_means"]

        if len(all_errors) == 0 or len(run_means) == 0:
            print(f"Skipping geometry {geometry}: insufficient data.")
            continue

        # 1) Histogram of all estimate_error values
        plt.figure(figsize=(8, 5))
        plt.hist(all_errors, bins=bins_all)
        plt.xlabel("Estimate Error")
        plt.ylabel("Count")
        plt.title(f"{geometry}: All Frame Estimate Errors")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, f"{geometry}_estimate_error_hist_all_frames.svg"),
            format="svg"
        )
        plt.close()

        # 2) Histogram of mean estimate_error per run
        plt.figure(figsize=(8, 5))
        plt.hist(run_means, bins=bins_runmean)
        plt.xlabel("Mean Estimate Error per Run")
        plt.ylabel("Count")
        plt.title(f"{geometry}: Mean Estimate Error per Run")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, f"{geometry}_estimate_error_hist_run_means.svg"),
            format="svg"
        )
        plt.close()

        print(
            f"Saved plots for geometry '{geometry}' "
            f"({len(all_errors)} frame errors, {len(run_means)} runs)."
        )


if __name__ == "__main__":
    input_folder = "D:\\phd\\4d-lidar\\to_sync\\to_sync\\to_sync\\final_res_final_res\\full_results"
    output_folder = "D:\\phd\\4d-lidar\\to_sync\\to_sync\\to_sync\\final_res_final_res\\estimate_error_histograms"

    plot_estimate_error_histograms(
        input_folder=input_folder,
        output_folder=output_folder,
        bins_all=60,
        bins_runmean=30,
    )