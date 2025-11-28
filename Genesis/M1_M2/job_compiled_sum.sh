#!/bin/bash
#SBATCH --job-name=...
#SBATCH --partition=compute
#SBATCH --account=...
#SBATCH --nodes=10   
#SBATCH --ntasks-per-node=128  
#SBATCH --mem=450GB
#SBATCH --time=08:00:00
#SBATCH --output=log_comp-%j.out
#SBATCH --error=log_comp-%j.err

# to create conda env: conda env create -f scatter_env.yml
source activate scatter_env
python setup_scatter.py build_ext --inplace

srun --mpi=pmi2 python3 ./M1_M2_sum.py
