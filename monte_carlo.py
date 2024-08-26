import numpy as np
import yaml
import os
import analyze_parallel_RC_pred as analyze_parallel
import pandas as pd
# import mpi4py.rc
# mpi4py.rc.threads = False
from mpi4py import MPI
import logging
import random

random.seed(42)
np.random.seed(42)


def run_monte_carlo(config):

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configure logging only for the process with rank 0
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    

    # get all file names
    pickle_files = os.listdir(config['pickle_directory_name'])

    if rank == 0:
        logger.info(f"Number of pickle files to analyse: {len(pickle_files)}")
        os.makedirs('full_results', exist_ok=True)
        os.makedirs('assignment_results', exist_ok=True)
    else:
        pass

    # Determine chunk size
    chunk_size = len(pickle_files) // size
    remainder = len(pickle_files) % size

    # Calculate start and end indices for this process
    start = rank * chunk_size
    end = start + chunk_size
    if rank == size - 1:
        end += remainder  # Last process takes any remaining rows

    simulation_data = []

    for idx in range(start, end):

        if rank == 0:
            logger.info(f"Processing {idx} of {end - start} iterations")
        else:
            pass
        pickle_file = pickle_files[idx]
        results = analyze_parallel.run(pickle_file, config, logger)
        simulation_data.append(results)

    # Synchronize processes
    comm.Barrier()

    all_simulation_data = comm.gather(simulation_data, root=0)

    if rank == 0:
        # convert to dataframe
        results_as_df = pd.DataFrame(np.array(all_simulation_data).squeeze().reshape(len(pickle_files), len(config['results_column_names'])), columns=config['results_column_names'])
        results_as_df['pickle_file'] = pickle_files

        # save as csv
        results_as_df.to_csv(os.path.join(config['top_level_dir'], config['results_file_name']), sep=',', header=True,
                             index=False)
    else:
        pass

    return


with open('configuration.yaml', 'r') as f:
    configs = yaml.safe_load(f)

run_monte_carlo(configs)
