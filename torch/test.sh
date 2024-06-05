#!/bin/bash
#
# alternatively, use srun
# srun --partition=gpu_min8GB --qos=gpu_min8GB_ext python3 train.py none
#
# gpu_min8GB, gpu_min11GB, gpu_min12GB, gpu_min24GB, gpu_min32GB, gpu_min80GB
#SBATCH --partition=gpu_min8gb       # Reserved partition
#SBATCH --qos=gpu_min8gb_ext             # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=test             # Job name
#slurm.%j.out
#SBATCH -o slurm-%j.txt           # File containing STDOUT output
#SBATCH -e slurm-%j.txt           # File containing STDERR output

# about half an hour per method

DATASET="Adience"
METHODS=" model-Adience-none.pth model-Adience-mixup.pth model-Adience-ordinal_adjacent_mixup.pth model-Adience-ordinal_exponential_mixup.pth model-Adience-nested.pth model-Adience-jaime.pth"
for METHOD in $METHODS; do
    echo "python test.py $METHOD $DATASET "
    python -u test.py  $METHOD $DATASET 
done
# rm slurm.txt; sbatch ./train.sh