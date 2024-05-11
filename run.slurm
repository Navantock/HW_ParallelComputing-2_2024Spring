#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -t 5:00

./main input1.mat output1_serial.txt output1_parallel.txt 2 10
python check.py input1.mat output1_serial.txt
python check.py input1.mat output1_parallel.txt
./main input2.mat output2_serial.txt output2_parallel.txt 2 10
python check.py input2.mat output2_serial.txt
python check.py input2.mat output2_parallel.txt
