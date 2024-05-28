#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 5:00

mpirun -np 2 ./main 32
mpirun -np 4 ./main 32
mpirun -np 6 ./main 32
mpirun -np 8 ./main 32
mpirun -np 2 ./main 1000
mpirun -np 4 ./main 1000
mpirun -np 6 ./main 1000
mpirun -np 8 ./main 1000
mpirun -np 2 ./main 10000
mpirun -np 4 ./main 10000
mpirun -np 6 ./main 10000
mpirun -np 8 ./main 10000
mpirun -np 2 ./main 100000
mpirun -np 4 ./main 100000
mpirun -np 6 ./main 100000
mpirun -np 8 ./main 100000
