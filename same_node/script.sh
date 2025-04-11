#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J life_science
#
# Number of MPI Tasks, e.g. 8:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --nvmps
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=02:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
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
# module load vtune
# export PYTHONPATH=/u/yju/.local/lib/python3.8/site-packages/:$PYTHONPATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
# Run the program:

srun python in_situ_sync.py
#srun hpcmd_suspend aps --stat-level=4 -r aps_result --  python pre_train_horovod.py

echo "job finished"
