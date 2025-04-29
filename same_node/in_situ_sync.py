#!/usr/bin/env python
# coding: utf-8

import time
import argparse
#import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import cv2
import horovod.tensorflow as hvd
import math
#from mpi4py import MPI
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import preprocess_srrf_mpi_test
import my_model

os.environ["HOROVOD_STALL_CHECK_TIME_SECONDS"] = "240"
######################## distributed using horovod ############################
'''
world_comm = MPI.COMM_WORLD
world_rank = world_comm.Get_rank()
world_comm_size = world_comm.Get_size()
subcomm = world_comm.Split(color=0, key=world_rank)
rank = subcomm.Get_rank()
size = subcomm.Get_size()
print("rank: ", rank, "subcomm size : ", size, flush = True)
'''
# Define the image size for preprocessing
IMG_SIZE = (224, 224)

# Define the batch size and number of epochs
BATCH_SIZE = 64
NUM_EPOCHS = 2
TRAIN_SIZE = 19200
NUM_ITER = math.floor(TRAIN_SIZE / BATCH_SIZE)

checkpoint_dir = '.ipynb_checkpoints/profiling/' 
path = 'tfrecords_4scattered/'
autotune = tf.data.experimental.AUTOTUNE

DATA_DIR = "/u/qchai/sequence/sequence/"
files = os.listdir(DATA_DIR)

parser = argparse.ArgumentParser()
# parser.add_argument('--save', default=1, type=int)
parser.add_argument('--save', default=0, type=int)
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int)
parser.add_argument('--meta_version', type=str)
parser.add_argument('--version', type=str)
parser.add_argument('--epochs', default=NUM_EPOCHS, type=int)
parser.add_argument('--schedule_rate', type=float)
parser.add_argument('--weights', type=str)
parser.add_argument('--nums', type=int)
args = parser.parse_args()

hvd.init()
#hvd.init(comm=subcomm)
size = hvd.size()
rank = hvd.rank()

gpus = tf.config.experimental.list_physical_devices('GPU')
print("hvd.size=",hvd.size(),"hvd.rank=", hvd.rank(), flush = True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


class checkpoint_cb(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, rank):
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank

    def on_epoch_end(self, epoch, logs=None):
        if self.rank == 0:
            print('saving model')
            checkpoint.save(self.checkpoint_dir)
            #zapisac w ten sposob co w horovodrun


class TimestampCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data):
        super(TimestampCallback, self).__init__()
        self.train_data = train_data

    def on_train_batch_begin(self, batch, logs=None):
        # Record the start time of the batch
        self.start_time = time.time()
        #batch_size = self.train_data.element_spec[0].shape[0]
        #print(f"Rank {hvd.rank()}: Element Spec: {self.train_data.element_spec}")
        #print(f"Rank {hvd.rank()}: Input Shape: {self.train_data.element_spec[0].shape}")
        #print(f"Rank {hvd.rank()}: Batch {batch}, Batch Size: {batch_size}")
    def on_train_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"Rank {hvd.rank()}: Batch {batch}, Time Elapsed: {elapsed_time:.3f} seconds")


nthread = 16
NUM_TRAIN_IMG = 16
NUM_PRE_IMG_PER_TRAIN_IMG = 50
width_in = 128
height_in = 128
CHANNEL = 1
NUM_AUGMENTATION = 9

# Define the function to preprocess each image
def preprocess():
    labels =  [random.randint(0, 1) for _ in range(NUM_TRAIN_IMG)]
    images = preprocess_srrf_mpi_test.process_SRRF_mpi(rank, size, nthread, NUM_TRAIN_IMG, NUM_PRE_IMG_PER_TRAIN_IMG, width_in, height_in, files)
    images = np.repeat(images, 3, axis=-1)
    
    return images, labels

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                             fill_mode='constant', cval=0)

def generator_function():
    cnt = 0
    while True:

#        print("rank = ", rank, ", ", cnt, " th start of preprocess", "images dtype: ",flush = True)
        images, labels = preprocess()
#        print("rank = ", rank, ", ", cnt, " th return of preprocess", "images dtype: ",flush = True)
        cnt = cnt+1

        for i in range(len(images)):
            
            # Convert the numpy array to tensorflow tensors
            image_tensor = tf.convert_to_tensor(images[i])
            label_tensor = tf.convert_to_tensor(labels[i])
            #yield image_tensor, label_tensor

            for j in range(NUM_AUGMENTATION+1):
                augmented_image = datagen.random_transform(images[i])    
                augmented_image_tensor = tf.convert_to_tensor(augmented_image)
                #print("rank = ", rank, ", ", j, " th augmentated image", flush = True)
                if j == 0:
                    yield image_tensor, label_tensor
                # Yield the augmented image and its corresponding label
                else:
                    augmented_image = datagen.random_transform(images[i])    
                    augmented_image_tensor = tf.convert_to_tensor(augmented_image)
                    yield augmented_image_tensor, label_tensor


train_dataset = tf.data.Dataset.from_generator(generator_function, output_types=(tf.float64, tf.int32))
#train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
#train_dataset = train_dataset.repeat(NUM_EPOCHS)
train_dataset = train_dataset.batch(BATCH_SIZE // hvd.size())
train_dataset = train_dataset.prefetch(buffer_size = autotune) # preferch 1 batch after .batch()
#train_dataset = train_dataset.shard(hvd.size(), hvd.rank())

initial_learning_rate = 0.01 * hvd.size()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

model = my_model.make_model()
loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer = hvd.DistributedOptimizer(optimizer)
model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['binary_accuracy'], 
        experimental_run_tf_function=False
    )
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

callbacks=[early_stopping_cb, 
               hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0), 
               hvd.keras.callbacks.MetricAverageCallback(),
               TimestampCallback(train_dataset),]
if args.save == True:
    callbacks.append(checkpoint_cb(checkpoint_dir, hvd.rank()))

tic = time.perf_counter()


model.fit(train_dataset,
          steps_per_epoch= NUM_ITER // hvd.size(),
          callbacks=callbacks,
          epochs=args.epochs,
          verbose=1 if hvd.rank() == 0 else 0)


toc = time.perf_counter()
print(f"Training Script executed in {toc - tic:0.4f} seconds")

