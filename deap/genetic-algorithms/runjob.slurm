#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10g
#SBATCH --time=20:00:00

python -m scoop onemax_parallel.py 
