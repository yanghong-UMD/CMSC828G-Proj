#!/bin/bash
#SBATCH -A cmsc828-class
#SBATCH --job-name=PyTorchGPU_16   # Name of the job
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00            # Set max runtime (e.g., 24 hours)
#SBATCH --ntasks=1                 # Number of tasks

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
source /home/akim1240/proj_env_gpu/bin/activate

# Command to execute your code
srun python PyTorchGPU.py --batch_size 16
