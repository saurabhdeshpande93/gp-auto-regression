#!/bin/bash -l
#SBATCH -J train_auto
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 14

cd ..
source /etc/profile
time python train.py --model autoencoder
