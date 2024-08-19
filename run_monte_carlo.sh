#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:0
#SBATCH --job-name monte_carlo_true_metric 
#SBATCH --output=mc_mpi_output_true_metric_%j.txt
#SBATCH --mail-type=FAIL

module load CCEnv StdEnv/2023
module load mpi4py
source ~/.virtualenvs/lidarenv/bin/activate
mpirun python monte_carlo.py
