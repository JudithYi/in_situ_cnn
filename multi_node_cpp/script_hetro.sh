#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J life_science
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --cpus-per-task=18 --nodes=1 --ntasks-per-node=4 --constraint="gpu" --gres=gpu:a100:4 --time=03:00:00 --nvmps -o ./async0.out.%j 
#SBATCH hetjob
#
#SBATCH --cpus-per-task=18 --nodes=1 --ntasks-per-node=4 --constraint="gpu" --gres=gpu:a100:4 --time=03:00:00 --nvmps -o ./async1.out.%j 

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.com


module purge
module load anaconda/3/2021.05
module load gcc/11
module load openmpi/4
module load cuda/11.4
module load keras/2.6.0
module load keras-preprocessing/1.1.2
module load tensorflow-estimator/2.6.0
module load tensorboard/2.6.0
module load tensorflow/gpu-cuda-11.4/2.6.0
module load horovod-tensorflow-2.6.0/gpu-cuda-11.4/0.22.0
module load opencv/cpu/4.5.2
module load mpi4py/3.0.3
module list

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores

srun --het-group=0 ./preprocess_cpp 8 8 8 8 8 8 : --het-group=1 python train.py
echo "job finished"
