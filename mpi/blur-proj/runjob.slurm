#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=12
#SBATCH --nodes=1
#SBATCH --mem=10g

module load openmpi/1.10.6
time mpirun -np 12 --bynode  python parallel-filter.py
echo Nodes=1 TaskesPerNode=12 Processes=12 
#requesting 4 nodes, 2 MPI processes on each node and 2 threads for each process 
