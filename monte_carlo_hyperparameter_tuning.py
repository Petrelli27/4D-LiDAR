import argparse
import copy
import itertools
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import analyze_parallel_withocclusion_hyperparameter_tuning as analyze_parallel


DEFAULT_CONFIG_PATH = "configuration.yaml"
DEFAULT_SELECTION_SEED = 42


random.seed(DEFAULT_SELECTION_SEED)
np.random.seed(DEFAULT_SELECTION_SEED)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _format_float_for_tag(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_combo_name(ransac_pca_threshold, orthonormal_thresh, eig_thresh, bias_recalibration_thresh):
    return (
        f"rpca_{_format_float_for_tag(ransac_pca_threshold)}"
        f"__ortho_{_format_float_for_tag(orthonormal_thresh)}"
        f"__eig_{_format_float_for_tag(eig_thresh)}"
        f"__brecal_{_format_float_for_tag(bias_recalibration_thresh)}"
    )


def _list_geometry_directories(root_dir):
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"geometry_root_directory does not exist or is not a directory: {root_dir}")

    geometry_dirs = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
    if not geometry_dirs:
        raise ValueError(f"No geometry subdirectories found in {root_dir}")
    return geometry_dirs


def _list_pickle_files(geometry_dir):
    return [p for p in sorted(geometry_dir.iterdir()) if p.is_file() and p.suffix == '.pickle']


def select_geometry_runs(config, logger):
    geometry_root = config['geometry_root_directory']
    runs_per_geometry = int(config['runs_per_geometry'])
    selection_seed = int(config.get('selection_seed', DEFAULT_SELECTION_SEED))

    if runs_per_geometry <= 0:
        raise ValueError("runs_per_geometry must be a positive integer")

    geometry_dirs = _list_geometry_directories(geometry_root)
    rng = random.Random(selection_seed)
    selected = []

    for geometry_dir in geometry_dirs:
        pickle_files = _list_pickle_files(geometry_dir)
        if not pickle_files:
            logger.warning("Skipping geometry '%s' because it contains no .pickle files", geometry_dir.name)
            continue
        if len(pickle_files) < runs_per_geometry:
            raise ValueError(
                f"Geometry '{geometry_dir.name}' only has {len(pickle_files)} pickle files, "
                f"but runs_per_geometry={runs_per_geometry}"
            )

        chosen = rng.sample(pickle_files, runs_per_geometry)
        chosen = sorted(chosen, key=lambda p: p.name)
        for run_idx, pickle_path in enumerate(chosen, start=1):
            selected.append({
                'geometry_name': geometry_dir.name,
                'pickle_path': str(pickle_path),
                'pickle_file': pickle_path.name,
                'run_number_for_combo': run_idx,
            })

    if not selected:
        raise ValueError("No pickle files were selected from any geometry directory")

    return selected


def build_hyperparameter_combos(config):
    rpca_values = config['ransac_pca_threshold']
    ortho_values = config['orthonormal_thresh']
    eig_values = config['eig_thresh']
    bias_recal_values = config['bias_recalibration_thresh']

    if not isinstance(rpca_values, (list, tuple)):
        rpca_values = [rpca_values]
    if not isinstance(ortho_values, (list, tuple)):
        ortho_values = [ortho_values]
    if not isinstance(eig_values, (list, tuple)):
        eig_values = [eig_values]
    if not isinstance(bias_recal_values, (list, tuple)):
        bias_recal_values = [bias_recal_values]

    combos = []
    for combo_index, (rpca, ortho, eig, bias_recal) in enumerate(
        itertools.product(rpca_values, ortho_values, eig_values, bias_recal_values),
        start=1,
    ):
        combos.append({
            'combo_index': combo_index,
            'ransac_pca_threshold': float(rpca),
            'orthonormal_thresh': float(ortho),
            'eig_thresh': float(eig),
            'bias_recalibration_thresh': float(bias_recal),
            'combo_name': build_combo_name(rpca, ortho, eig, bias_recal),
        })
    return combos


def build_tasks(config, logger):
    selected_runs = select_geometry_runs(config, logger)
    combos = build_hyperparameter_combos(config)
    tasks = []

    for run_info in selected_runs:
        for combo in combos:
            task_config = copy.deepcopy(config)
            task_config['ransac_pca_threshold'] = combo['ransac_pca_threshold']
            task_config['orthonormal_thresh'] = combo['orthonormal_thresh']
            task_config['eig_thresh'] = combo['eig_thresh']
            task_config['bias_recalibration_thresh'] = combo['bias_recalibration_thresh']

            task = {
                **run_info,
                **combo,
                'config': task_config,
            }
            tasks.append(task)

    return tasks, selected_runs, combos


def run_monte_carlo(config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    all_tasks, selected_runs, combos = build_tasks(config, logger)

    if rank == 0:
        logger.info("Selected %d runs across %d geometry directories", len(selected_runs), len({r['geometry_name'] for r in selected_runs}))
        logger.info("Evaluating %d hyperparameter combinations", len(combos))
        logger.info("Total MPI tasks to execute: %d", len(all_tasks))
        os.makedirs('full_results', exist_ok=True)
        os.makedirs('assignment_results', exist_ok=True)

    num_tasks = len(all_tasks)
    chunk_size = num_tasks // size
    remainder = num_tasks % size

    if rank < remainder:
        start_idx = rank * (chunk_size + 1)
        end_idx = start_idx + (chunk_size + 1)
    else:
        start_idx = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size
        end_idx = start_idx + chunk_size

    my_tasks = all_tasks[start_idx:end_idx]

    simulation_data = []
    comp_times = []
    task_metadata = []

    for i, task in enumerate(my_tasks, start=1):
        logger.info(
            "Rank %d processing %d/%d: geometry=%s run=%s combo=%s",
            rank,
            i,
            len(my_tasks),
            task['geometry_name'],
            task['pickle_file'],
            task['combo_name'],
        )
        t0 = time.time()
        results = analyze_parallel.run(task, task['config'], logger)
        t1 = time.time()
        comp_times.append(t1 - t0)
        simulation_data.append(results)
        task_metadata.append({
            'geometry_name': task['geometry_name'],
            'pickle_file': task['pickle_file'],
            'pickle_path': task['pickle_path'],
            'combo_name': task['combo_name'],
            'combo_index': task['combo_index'],
            'run_number_for_combo': task['run_number_for_combo'],
            'ransac_pca_threshold': task['ransac_pca_threshold'],
            'orthonormal_thresh': task['orthonormal_thresh'],
            'eig_thresh': task['eig_thresh'],
            'bias_recalibration_thresh': task['bias_recalibration_thresh'],
        })

    logger.info("Rank %d done all tasks", rank)

    comm.Barrier()

    all_simulation_data = comm.gather(simulation_data, root=0)
    all_comp_times = comm.gather(comp_times, root=0)
    all_task_metadata = comm.gather(task_metadata, root=0)

    if rank == 0:
        flat_results = [row for rank_list in all_simulation_data for row in rank_list]
        flat_times = [ct for rank_list in all_comp_times for ct in rank_list]
        flat_task_metadata = [meta for rank_list in all_task_metadata for meta in rank_list]

        if len(flat_results) == 0:
            logger.warning("No results were returned. Writing an empty results file.")
            results_as_df = pd.DataFrame()
        else:
            if isinstance(flat_results[0], dict):
                results_as_df = pd.DataFrame(flat_results)
            else:
                results_as_df = pd.DataFrame(flat_results, columns=config['results_column_names'])

        if len(results_as_df) == len(flat_task_metadata):
            metadata_df = pd.DataFrame(flat_task_metadata)
            results_as_df = pd.concat([results_as_df.reset_index(drop=True), metadata_df.reset_index(drop=True)], axis=1)
        else:
            logger.warning("Length mismatch between results and task metadata; skipping metadata merge.")

        if len(results_as_df) == len(flat_times):
            results_as_df['comp_time'] = flat_times
        else:
            logger.warning("Length mismatch between results and comp_times; skipping comp_time column.")

        if 'results_column_names' in config:
            cols = [c for c in config['results_column_names'] if c in results_as_df.columns]
            extras = [c for c in results_as_df.columns if c not in cols]
            results_as_df = results_as_df[cols + extras]

        out_path = os.path.join(config['top_level_dir'], config['results_file_name'])
        os.makedirs(config['top_level_dir'], exist_ok=True)
        results_as_df.to_csv(out_path, sep=',', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    configs = load_config(args.config_path)
    run_monte_carlo(configs)
