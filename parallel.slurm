#!/bin/bash
#SBATCH --job-name=RYZH_DDP
#SBATCH --output=RYZH_DDP%j.out
#SBATCH --error=RYZH_DDP%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --account=dxe@a100
#SBATCH --hint=nomultithread


module purge
conda deactivate

nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"


module load arch/a100
module load pytorch-gpu/py3/2.3.0

set -x

torchrun --nproc_per_node=4 train.py