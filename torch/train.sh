#!/bin/bash
#
# gpu_min8GB, gpu_min11GB, gpu_min12GB, gpu_min24GB, gpu_min32GB, gpu_min80GB
#SBATCH --partition=gpu_min8GB       # Reserved partition
#SBATCH --qos=gpu_min8GB_ext             # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=train             # Job name
#slurm.%j.out
#SBATCH -o slurm.txt           # File containing STDOUT output
#SBATCH -e slurm.txt           # File containing STDERR output

echo "train with none augmentation"
python train.py none
