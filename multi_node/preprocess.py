import numpy as np
import sys
import os
import math
import time
import preprocess_srrf_mpi_test
import adios2
from mpi4py import MPI


####################### Writer ###########################
print("start python script", flush=True)

global_comm = MPI.COMM_WORLD
global_rank = global_comm.Get_rank()
global_size = global_comm.Get_size()
colour=0
comm=global_comm.Split(colour,global_rank)
rank = comm.Get_rank()
size = comm.Get_size()
print("MPI initialized on rank: ", rank, flush = True)

adios = adios2.ADIOS("config/config.xml", comm, True)
inIO = adios.DeclareIO("writer")
writer = inIO.Open("globalArray", adios2.Mode.Write, comm)
print("Adios initialized on rank: ", rank, flush = True)


DATA_DIR = "/u/path/to/sequence/"
files = os.listdir(DATA_DIR)

NUM_PRE_IMG = 12
NUM_TRAIN_IMG = 16

NTHREAD = 16 # set identical to NUM_TRAIN_IMG to fully utilize the resouce
WIDTH = 224
HEIGHT = 224
CHANNEL = 1

GLOBAL_BATCH_SIZE = 64
GLOBAL_TRAIN_SIZE = 19200
NUM_ITER = GLOBAL_TRAIN_SIZE // GLOBAL_BATCH_SIZE


if rank == 0:
    print("start preprocessing", flush = True)

img_shape = (WIDTH,HEIGHT,CHANNEL)
img_size = np.prod(img_shape)

global_img_count = size * NUM_TRAIN_IMG
local_img_count = NTHREAD
write_start = rank * local_img_count
write_end = write_start + local_img_count

img_array = np.zeros([global_img_count, WIDTH, HEIGHT, CHANNEL])
label_array = np.zeros(global_img_count, dtype=int)
img_var = inIO.DefineVariable("image", img_array, [global_img_count, WIDTH, HEIGHT, CHANNEL], [write_start, 0, 0, 0], [local_img_count, WIDTH, HEIGHT, CHANNEL], adios2.ConstantDims)
label_var = inIO.DefineVariable("label",label_array, [global_img_count], [write_start], [local_img_count], adios2.ConstantDims)

width_in = 128
height_in = 128

EPOCHS = 2
for i in range(EPOCHS):
    for j in range(NUM_ITER):
        processed_img = preprocess_srrf_mpi_test.process_SRRF_mpi(rank, size, NTHREAD, NUM_TRAIN_IMG, NUM_PRE_IMG, width_in, height_in, files)
        #processed_img = np.ones([NUM_TRAIN_IMG, 224, 224, 3])
        labels = np.random.randint(2, size = local_img_count) # TODO: currently randomly generate labels
        writer.BeginStep()
        writer.Put(img_var, processed_img)
        writer.Put(label_var, labels)
        writer.EndStep()
        if rank == 0:
            print(j, "th iter of writing done", flush = True)

writer.Close()
if rank == 0:
    print("finish python script")
