#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:00:0
#SBATCH --job-name monte_carlo_1000_qinit_is_true 
#SBATCH --output=mc_mpi_output_1000_qinit_is_true_%j.txt
#SBATCH --mail-type=FAIL

module load CCEnv StdEnv/2023
module load mpi4py
source ~/.virtualenvs/lidarenv/bin/activate
mpirun python monte_carlo.py
