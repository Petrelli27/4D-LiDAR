#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=23:00:0
#SBATCH --job-name generate_sim_data_1000 
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

#module load NiaEnv/2022a
#module load python/3.11.5
#module load intel/2022u2
#module load openmpi/4.1.4+ucx-1.11.2
module load CCEnv StdEnv/2023
module load mpi4py
export MPLCONFIGDIR=$SCRATCH/matplotlib
source ~/.virtualenvs/lidarenv/bin/activate
mpirun python simulate_many_mpi4py_monte.py
