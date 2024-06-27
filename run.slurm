#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -t 5:00

echo "Serial Begin"
./serial_ccl pixels_1.txt ./output/labels_1_serial.txt 10
./serial_ccl pixels_2.txt ./output/labels_2_serial.txt 10
./serial_ccl pixels_3.txt ./output/labels_3_serial.txt 10
./serial_ccl pixels_4.txt ./output/labels_4_serial.txt 10
echo "Serial Done"
echo "Parallel Done"
./cuda_ccl pixels_1.txt ./output/labels_1_cuda.txt > /dev/null
./cuda_ccl pixels_1.txt ./output/labels_1_cuda.txt 10
./cuda_ccl pixels_2.txt ./output/labels_2_cuda.txt 10
./cuda_ccl pixels_3.txt ./output/labels_3_cuda.txt 10
./cuda_ccl pixels_4.txt ./output/labels_4_cuda.txt 10
echo "Parallel Done"
echo "Check Results Correctness"
python check.py ./output/labels_1_serial.txt ./output/labels_1_cuda.txt
python check.py ./output/labels_2_serial.txt ./output/labels_2_cuda.txt
python check.py ./output/labels_3_serial.txt ./output/labels_3_cuda.txt
python check.py ./output/labels_4_serial.txt ./output/labels_4_cuda.txt


