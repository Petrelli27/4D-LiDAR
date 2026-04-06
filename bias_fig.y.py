import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_bias_effect(csv_file):

    # --- Load ---
    df = pd.read_csv(csv_file)

    # --- Remove duplicate columns ---
    df = df.loc[:, ~df.columns.duplicated()]

    # --- Compute 3D RMSE ---
    df["rmse_before"] = np.sqrt(
        df["rmse_x_before"]**2 +
        df["rmse_y_before"]**2 +
        df["rmse_z_before"]**2
    )

    df["rmse_after"] = np.sqrt(
        df["rmse_x_after"]**2 +
        df["rmse_y_after"]**2 +
        df["rmse_z_after"]**2
    )

    # --- Group by bias threshold ---
    grouped = df.groupby("bias_recalibration_thresh").agg({
        "rmse_before": "mean",
        "rmse_after": "mean"
    }).reset_index()

    # --- Sort for clean plotting ---
    grouped = grouped.sort_values("bias_recalibration_thresh")

    # --- Plot ---
    plt.figure()

    # Before bias
    plt.plot(
        grouped["bias_recalibration_thresh"],
        grouped["rmse_before"],
        marker='o',
        label="Before bias"
    )

    # After bias
    plt.plot(
        grouped["bias_recalibration_thresh"],
        grouped["rmse_after"],
        marker='o',
        label="After bias"
    )

    plt.xlabel("Bias recalibration threshold")
    plt.ylabel("Mean 3D Position RMSE")
    plt.title("Effect of Bias Recalibration Threshold on Position Error")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_bias_effect("to_sync/to_sync/bias_res/mc_res_combo.csv")