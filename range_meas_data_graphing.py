import os
import pandas as pd


def compile_results_csv(root_folder, output_csv):
    """
    Recursively search for CSV files whose filename contains
    'results_of_sim_debris_trimesh', extract selected columns,
    keep only rows where frame_good == True, and combine into one CSV.

    Parameters
    ----------
    root_folder : str
        Top-level folder to search.
    output_csv : str
        Path to the compiled output CSV.
    """

    target_substring = "results_of_sim_debris_trimesh"
    required_columns = ["ransac_error", "pca_error", "true_range", "file_name", "frame_good"]

    compiled_dfs = []

    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith(".csv") and target_substring in fname:
                full_path = os.path.join(dirpath, fname)

                try:
                    df = pd.read_csv(full_path)

                    missing = [col for col in required_columns if col not in df.columns]
                    if missing:
                        print(f"Skipping {full_path} because missing columns: {missing}")
                        continue

                    # Robust handling in case frame_good is stored as bool/string/0-1
                    frame_good_mask = df["frame_good"].astype(str).str.strip().str.lower().isin(
                        ["true", "1", "yes"]
                    )

                    filtered = df.loc[frame_good_mask, ["ransac_error", "pca_error", "true_range", "file_name"]]
                    compiled_dfs.append(filtered)

                    print(f"Processed: {full_path} | kept {len(filtered)} rows")

                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    if compiled_dfs:
        result = pd.concat(compiled_dfs, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["ransac_error", "pca_error", "true_range", "file_name"])

    result.to_csv(output_csv, index=False)
    print(f"\nSaved compiled CSV to: {output_csv}")
    print(f"Total rows: {len(result)}")


if __name__ == "__main__":
    root_folder = r"range_pca_ransac_data"
    output_csv = r"range_data_compiled_results.csv"

    compile_results_csv(root_folder, output_csv)