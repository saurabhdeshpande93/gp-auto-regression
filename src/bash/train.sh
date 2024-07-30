#!/bin/bash -l
#SBATCH -N 1
#SBATCH -J auto_gp_framework
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:10:00
#SBATCH -p batch
#SBATCH --qos=normal

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"


autoencoder_job_id=$(sbatch --parsable run_auto.sh)
sbatch --dependency=afterok:$autoencoder_job_id run_gp.sh
