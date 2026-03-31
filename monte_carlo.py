import argparse
import os
import time
import logging
import random

import numpy as np
import pandas as pd
import yaml

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import analyze_parallel_withocclusion as analyze_parallel
# If your actual file/module is still named analyze_parallel_RC_pred_cluster.py,
# then keep that import instead.


random.seed(42)
np.random.seed(42)

DEFAULT_CONFIG_PATH = "configuration.yaml"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_monte_carlo(config):

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # get all file names
    pickle_files = os.listdir(config['pickle_directory_name'])

    if rank == 0:
        logger.info(f"Number of pickle files to analyse: {len(pickle_files)}")
        os.makedirs('full_results', exist_ok=True)
        os.makedirs('assignment_results', exist_ok=True)

    # Determine chunk size
    chunk_size = len(pickle_files) // size
    remainder = len(pickle_files) % size

    # Calculate start and end indices for this process
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    if rank == size - 1:
        end_idx += remainder  # Last process takes any remaining rows

    # Slice the files this rank will handle
    my_files = pickle_files[start_idx:end_idx]

    simulation_data = []
    comp_times = []

    for i, pickle_file in enumerate(my_files, start=1):
        logger.info(f"Rank {rank} processing {i}/{len(my_files)}: {pickle_file}")
        t0 = time.time()
        results = analyze_parallel.run(pickle_file, config, logger)
        t1 = time.time()
        comp_times.append(t1 - t0)
        simulation_data.append(results)

    logger.info(f"Rank {rank} done all files")

    comm.Barrier()

    # Gather lists from all ranks
    all_simulation_data = comm.gather(simulation_data, root=0)
    all_comp_times = comm.gather(comp_times, root=0)
    all_file_slices = comm.gather(my_files, root=0)

    if rank == 0:
        # Flatten in rank order so rows align
        flat_results = [row for rank_list in all_simulation_data for row in rank_list]
        flat_times = [ct for rank_list in all_comp_times for ct in rank_list]
        flat_files = [pf for rank_list in all_file_slices for pf in rank_list]

        if len(flat_results) == 0:
            logger.warning("No results were returned. Writing an empty results file.")
            results_as_df = pd.DataFrame()
        else:
            # Case A: each `results` is a dict
            if isinstance(flat_results[0], dict):
                results_as_df = pd.DataFrame(flat_results)

                if 'results_column_names' in config:
                    cols = [c for c in config['results_column_names'] if c in results_as_df.columns]
                    extras = [c for c in results_as_df.columns if c not in cols]
                    results_as_df = results_as_df[cols + extras]
            else:
                # Case B: each `results` is a list/tuple
                results_as_df = pd.DataFrame(
                    flat_results,
                    columns=config['results_column_names']
                )

        # Add timing and file columns
        if len(results_as_df) == len(flat_times):
            results_as_df['comp_time'] = flat_times
        else:
            logger.warning("Length mismatch between results and comp_times; skipping comp_time column.")

        if len(results_as_df) == len(flat_files):
            results_as_df['pickle_file'] = flat_files
        else:
            logger.warning("Length mismatch between results and pickle files; skipping pickle_file column.")

        # Save
        out_path = os.path.join(config['top_level_dir'], config['results_file_name'])
        os.makedirs(config['top_level_dir'], exist_ok=True)
        results_as_df.to_csv(out_path, sep=',', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    configs = load_config(args.config_path)
    run_monte_carlo(configs)