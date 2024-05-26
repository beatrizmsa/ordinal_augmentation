#!/bin/bash
#
# alternatively, use srun
# srun --partition=gpu_min8GB --qos=gpu_min8GB_ext python3 train.py none
#
# gpu_min8GB, gpu_min11GB, gpu_min12GB, gpu_min24GB, gpu_min32GB, gpu_min80GB
#SBATCH --partition=gpu_min8gb       # Reserved partition
#SBATCH --qos=gpu_min8gb_ext             # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train             # Job name
#slurm.%j.out
#SBATCH -o slurm-%j.txt           # File containing STDOUT output
#SBATCH -e slurm-%j.txt           # File containing STDERR output

# about half an hour per method

DATASET="Adience"
METHODS="none mixup ordinal_adjacent_mixup ordinal_exponential_mixup nested jaime"
for METHOD in $METHODS; do
    echo "python train.py $DATASET $METHOD"
    python -u train.py $DATASET $METHOD
done
# rm slurm.txt; sbatch ./train.sh