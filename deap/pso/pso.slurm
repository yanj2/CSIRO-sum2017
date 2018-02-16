#!/bin/bash 
#SBATCH --time=01:00:00
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --mem=10GB 

source ../../../virtual/bin/activate

scontrol show hostnames $SLURM_NODELIST > host.list

time python -m scoop --hostfile host.list pso_parallel.py 
