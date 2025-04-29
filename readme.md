# How to Run the CNN with In-Situ SRRF

## 1. Install the required modules

Make sure the following software and libraries are installed:

- Anaconda
- GCC
- OpenMPI
- CUDA
- TensorFlow
- Horovod
- OpenCV
- mpi4py
- ADIOS2

## 2. Modify file paths

Update the relevant file paths in the scripts and configuration files to match your local environment.

## 3. Compile the C++ version (if needed)

If you are using the C++ version:

- Navigate to the `multi_node_cpp` directory.
- Compile the executable using the provided `make.sh` script:

```bash
cd multi_node_cpp
./make.sh