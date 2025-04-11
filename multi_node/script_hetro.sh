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
#SBATCH --cpus-per-task=18 --nodes=4 --ntasks-per-node=4 --time=03:00:00 -o ./async0.out.%j
#SBATCH hetjob
#
#SBATCH --cpus-per-task=18 --nodes=1 --ntasks-per-node=4 --constraint="gpu" --gres=gpu:a100:4 --time=03:00:00 -o ./async1.out.%j 
#SBATCH --nvmps

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de.de


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
# export PYTHONPATH=/u/qchai/adios2_cpu_py3.8/lib/python3.8/site-packages/adios2/:$PYTHONPATH
#srun bash -c 'echo "Rank: $PMI_RANK   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"'
#srun bash -c 'echo "Rank: $PMI_RANK   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"'
#srun ./run_gpu.sh cp_comp_parallel.x -i cp.in 
srun --het-group=0 python preprocess.py : --het-group=1 python train.py
echo "job finished"
