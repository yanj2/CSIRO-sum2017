#!/bin/bash 
#SBATCH --time=01:00:00
#SBATCH --ntasks=3
#SBATCH --nodes=4
#SBATCH --mem=10GB 

time -m scoop python pso_parallel.py 
