import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Fit utilities
# ============================================================

def compute_scores(y, yhat, n_params):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    n = len(y)
    residuals = y - yhat

    sse = np.sum(residuals**2)
    mse = sse / n
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    sst = np.sum((y - np.mean(y))**2)

    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    if n > n_params + 1:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)
    else:
        adj_r2 = np.nan

    eps = 1e-300
    aic = n * np.log((sse / n) + eps) + 2 * n_params
    bic = n * np.log((sse / n) + eps) + n_params * np.log(n)

    return {
        "n": n,
        "n_params": n_params,
        "sse": sse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "adj_r2": adj_r2,
        "aic": aic,
        "bic": bic,
    }


def fit_polynomial(x, y, degree):
    coeffs = np.polyfit(x, y, degree)

    def predict(x_new):
        return np.polyval(coeffs, x_new)

    yhat = predict(x)

    if degree == 0:
        equation = f"y = {coeffs[0]:.6g}"
    elif degree == 1:
        equation = f"y = {coeffs[0]:.6g} r + {coeffs[1]:.6g}"
    elif degree == 2:
        equation = f"y = {coeffs[0]:.6g} r^2 + {coeffs[1]:.6g} r + {coeffs[2]:.6g}"
    elif degree == 3:
        equation = (
            f"y = {coeffs[0]:.6g} r^3 + {coeffs[1]:.6g} r^2 "
            f"+ {coeffs[2]:.6g} r + {coeffs[3]:.6g}"
        )
    else:
        equation = f"Polynomial degree {degree}"

    return {
        "model": f"poly_deg_{degree}",
        "coeffs": coeffs,
        "predict": predict,
        "equation": equation,
        "yhat": yhat,
        "n_params": degree + 1,
    }


def fit_power_law(x, y):
    valid = (x > 0) & (y > 0)

    if np.sum(valid) < 3:
        return None

    logx = np.log(x[valid])
    logy = np.log(y[valid])

    b, loga = np.polyfit(logx, logy, 1)
    a = np.exp(loga)

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=float)
        return a * np.power(x_new, b)

    return {
        "model": "power_law",
        "coeffs": np.array([a, b]),
        "predict": predict,
        "equation": f"y = {a:.6g} r^{b:.6g}",
        "yhat": predict(x),
        "n_params": 2,
    }


def fit_exponential(x, y):
    valid = y > 0

    if np.sum(valid) < 3:
        return None

    logy = np.log(y[valid])

    b, loga = np.polyfit(x[valid], logy, 1)
    a = np.exp(loga)

    def predict(x_new):
        x_new = np.asarray(x_new, dtype=float)
        return a * np.exp(b * x_new)

    return {
        "model": "exponential",
        "coeffs": np.array([a, b]),
        "predict": predict,
        "equation": f"y = {a:.6g} exp({b:.6g} r)",
        "yhat": predict(x),
        "n_params": 2,
    }


def fit_models(x, y):
    fits = []

    for degree in [0, 1, 2, 3]:
        fits.append(fit_polynomial(x, y, degree))

    power = fit_power_law(x, y)
    if power is not None:
        fits.append(power)

    exponential = fit_exponential(x, y)
    if exponential is not None:
        fits.append(exponential)

    rows = []
    for fit in fits:
        rows.append({
            "model": fit["model"],
            "equation": fit["equation"],
            **compute_scores(y, fit["yhat"], fit["n_params"]),
        })

    scores_df = pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)

    return fits, scores_df


# ============================================================
# Data preparation
# ============================================================

def make_combined_xyz_dataset(
    df,
    x_column,
    y_columns,
    use_bin_midpoint=True,
):
    rows = []

    for _, row in df.iterrows():
        if use_bin_midpoint:
            x_val = 0.5 * (row["range_bin_min_m"] + row["range_bin_max_m"])
        else:
            x_val = row[x_column]

        for axis, col in zip(["x", "y", "z"], y_columns):
            rows.append({
                "range_m": x_val,
                "axis": axis,
                "sigma": row[col],
            })

    out = pd.DataFrame(rows)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    return out


# ============================================================
# Plotting
# ============================================================

def plot_combined_fit(data, fits, scores_df, output_png):
    x = data["range_m"].to_numpy(float)
    y = data["sigma"].to_numpy(float)

    x_dense = np.linspace(np.min(x), np.max(x), 500)

    plt.figure()

    for axis in ["x", "y", "z"]:
        sub = data[data["axis"] == axis]
        plt.scatter(
            sub["range_m"],
            sub["sigma"],
            label=f"{axis}-component data",
        )

    for _, score_row in scores_df.iterrows():
        model_name = score_row["model"]
        fit = next(f for f in fits if f["model"] == model_name)

        plt.plot(
            x_dense,
            fit["predict"](x_dense),
            label=f"{model_name}: BIC={score_row['bic']:.2f}, RMSE={score_row['rmse']:.4g}",
        )

    plt.xlabel("Range (m)")
    plt.ylabel("Component-wise position error std (m)")
    plt.title("Combined x/y/z Range-Dependent Position Error Fit")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()


def plot_best_fit_only(data, fits, scores_df, output_png):
    x = data["range_m"].to_numpy(float)
    x_dense = np.linspace(np.min(x), np.max(x), 500)

    best_model = scores_df.iloc[0]["model"]
    best_fit = next(f for f in fits if f["model"] == best_model)

    plt.figure()

    for axis in ["x", "y", "z"]:
        sub = data[data["axis"] == axis]
        plt.scatter(
            sub["range_m"],
            sub["sigma"],
            label=f"{axis}-component data",
        )

    plt.plot(
        x_dense,
        best_fit["predict"](x_dense),
        linewidth=2,
        label=f"Best: {best_model}",
    )

    plt.xlabel("Range (m)")
    plt.ylabel("Component-wise position error std (m)")
    plt.title("Best Range-Dependent Position Error Fit")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()


# ============================================================
# Main analysis
# ============================================================

def run_combined_xyz_fit(
    input_csv,
    output_folder,
    x_column="mean_true_range_m",
    y_columns=None,
    use_bin_midpoint=True,
):
    os.makedirs(output_folder, exist_ok=True)

    if y_columns is None:
        y_columns = [
            "std_position_error_x_m",
            "std_position_error_y_m",
            "std_position_error_z_m",
        ]

    df = pd.read_csv(input_csv)

    missing = [c for c in y_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing y columns: {missing}")

    if not use_bin_midpoint and x_column not in df.columns:
        raise ValueError(f"Missing x column: {x_column}")

    data = make_combined_xyz_dataset(
        df=df,
        x_column=x_column,
        y_columns=y_columns,
        use_bin_midpoint=use_bin_midpoint,
    )

    data_path = os.path.join(output_folder, "combined_xyz_fit_data.csv")
    data.to_csv(data_path, index=False)

    x = data["range_m"].to_numpy(float)
    y = data["sigma"].to_numpy(float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    fits, scores_df = fit_models(x, y)

    scores_path = os.path.join(output_folder, "combined_xyz_fit_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    plot_all_path = os.path.join(output_folder, "combined_xyz_all_fits.png")
    plot_best_path = os.path.join(output_folder, "combined_xyz_best_fit.png")

    plot_combined_fit(data, fits, scores_df, plot_all_path)
    plot_best_fit_only(data, fits, scores_df, plot_best_path)

    print("\nSaved:")
    print(data_path)
    print(scores_path)
    print(plot_all_path)
    print(plot_best_path)

    print("\nFit ranking by BIC:")
    print(scores_df[["model", "rmse", "r2", "adj_r2", "aic", "bic", "equation"]])


# ============================================================
# PyCharm run-button config
# ============================================================

if __name__ == "__main__":

    INPUT_CSV = "range_binned_uncertainties3.csv"
    OUTPUT_FOLDER = "fit_outputs_combined_xyz"

    # If True, uses midpoint of each bin:
    # 0-25 -> 12.5, 25-50 -> 37.5, etc.
    # If False, uses x_column, usually mean_true_range_m.
    USE_BIN_MIDPOINT = True

    X_COLUMN = "mean_true_range_m"

    # Y_COLUMNS = [
    #     "var_position_error_x_m",
    #     "var_position_error_y_m",
    #     "var_position_error_z_m",
    # ]
    #
    # Y_COLUMNS = [
    #     "var_angular_velocity_error_x_rad_s",
    #     "var_angular_velocity_error_y_rad_s",
    #     "var_angular_velocity_error_z_rad_s",
    # ]
    #
    Y_COLUMNS = [
        "var_quaternion_error_w",
        "var_quaternion_error_x",
        "var_quaternion_error_y",
        'var_quaternion_error_z',
    ]

    run_combined_xyz_fit(
        input_csv=INPUT_CSV,
        output_folder=OUTPUT_FOLDER,
        x_column=X_COLUMN,
        y_columns=Y_COLUMNS,
        use_bin_midpoint=USE_BIN_MIDPOINT,
    )