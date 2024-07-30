#!/bin/bash -l
#SBATCH -N 1
#SBATCH -J train_gp
#SBATCH --ntasks-per-node=14
#SBATCH --time=0-08:00:00
#SBATCH -p batch
#SBATCH --qos=normal

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

cd ..
source /etc/profile
time python train.py --model gp
