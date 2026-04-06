import pandas as pd


def compute_geometry_means(input_csv, output_csv):
    """
    Takes the compiled run-level RMSE CSV and computes mean metrics per geometry.
    """

    df = pd.read_csv(input_csv)

    required_cols = [
        "geometry_name",
        "rmse_position",
        "rmse_linear_velocity",
        "rmse_angular_velocity",
        "rmse_orientation",
        "rmse_position_before_bias",
        "rmse_position_after_bias",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Group by geometry
    grouped = df.groupby("geometry_name")

    # Compute means
    summary = grouped.agg({
        "rmse_position": "mean",
        "rmse_linear_velocity": "mean",
        "rmse_angular_velocity": "mean",
        "rmse_orientation": "mean",
        "rmse_position_before_bias": "mean",
        "rmse_position_after_bias": "mean",
    }).reset_index()

    # Add number of runs per geometry
    summary["n_runs"] = grouped.size().values

    # Optional: reorder columns nicely
    summary = summary[
        [
            "geometry_name",
            "n_runs",
            "rmse_position",
            "rmse_linear_velocity",
            "rmse_angular_velocity",
            "rmse_orientation",
            "rmse_position_before_bias",
            "rmse_position_after_bias",
        ]
    ]

    # Save
    summary.to_csv(output_csv, index=False)

    print(f"Saved geometry-level summary to: {output_csv}")
    print(summary)

    return summary


if __name__ == "__main__":
    input_csv = r"compiled_rmse_summary.csv"
    output_csv = r"geometry_mean_rmse.csv"

    compute_geometry_means(input_csv, output_csv)