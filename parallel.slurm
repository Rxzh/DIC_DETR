#!/bin/bash
#SBATCH --job-name=RYZH_JEPA_MINI_DDP  # Changed job name
#SBATCH --output=RYZH_JEPA_MINI_DDP%j.out # Changed output file
#SBATCH --error=RYZH_JEPA_MINI_DDP%j.err  # Changed error file
#SBATCH --constraint=a100
#SBATCH --nodes=1                 # We are still using 1 machine (node)
#SBATCH --ntasks=1                # We will run 1 "main" task, which is torchrun
#SBATCH --time=04:00:00
#SBATCH --account=dxe@a100
#SBATCH --hint=nomultithread

N_GPUS=4 

#SBATCH --gres=gpu:$N_GPUS

# Request enough CPUs for all GPU processes
CPUS_PER_GPU=4
#SBATCH --cpus-per-task=$(( N_GPUS * CPUS_PER_GPU ))

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

srun torchrun --nproc_per_node=$N_GPUS train.py