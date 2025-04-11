import numpy as np
import sys
import os
import time
import math
import random
import adios2
import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mpi4py import MPI
import my_model

os.environ["HOROVOD_STALL_CHECK_TIME_SECONDS"] = "240"
####################### Reader ###########################
global_comm = MPI.COMM_WORLD
global_rank = global_comm.Get_rank()
global_size = global_comm.Get_size()
colour=1
sub_comm=global_comm.Split(colour,global_rank)
rank = sub_comm.Get_rank()
size = sub_comm.Get_size()

hvd.init(comm=sub_comm)

adios = adios2.ADIOS("config/config.xml", sub_comm, True)
inIO = adios.DeclareIO("reader")
reader = inIO.Open("globalArray", adios2.Mode.Read, sub_comm)

# training parameters

BATCH_SIZE = 64
SUB_BATCH_SIZE = math.floor(BATCH_SIZE / size)
EPOCHS = 2
TRAIN_SIZE = 19200
NUM_ITER = math.floor(TRAIN_SIZE / BATCH_SIZE)

gpus = tf.config.experimental.list_physical_devices('GPU')
print("hvd.size=",hvd.size(),"hvd.rank=", hvd.rank(), flush = True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


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

model = my_model.make_model()

checkpoint_dir = '.ipynb_checkpoints/profiling/' 
autotune = tf.data.experimental.AUTOTUNE
initial_learning_rate = 0.01 * hvd.size()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)
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


WIDTH = 224
HEIGHT = 224
CHANNEL = 1
nthread = 16
write_ranks = 16
NUM_TRAIN_IMG = nthread * write_ranks // size 
image_read_start = hvd.rank() * NUM_TRAIN_IMG
image_read_end = image_read_start + NUM_TRAIN_IMG


def read_data_from_adios():
    status = reader.BeginStep()
    if status == adios2.StepStatus.OK:
        img_in = inIO.InquireVariable("image")
        label_in = inIO.InquireVariable("label")
        img_in.SetSelection([[image_read_start, 0, 0,0], [NUM_TRAIN_IMG, WIDTH, HEIGHT, CHANNEL]])
        label_in.SetSelection([[image_read_start],[NUM_TRAIN_IMG]])

        # prepare for receiving data
        images = np.zeros([NUM_TRAIN_IMG, WIDTH, HEIGHT, CHANNEL])
        labels = np.zeros(NUM_TRAIN_IMG)

        reader.Get(img_in, images)
        reader.Get(label_in, labels)

        reader.EndStep()
        
        # convert to 3 channels
        images = np.repeat(images, 3, axis=-1)
        
        return images, labels

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                             fill_mode='constant', cval=0)
NUM_AUGMENTATION = 9
def generator_function():
    for i in range(NUM_ITER):
        # Read a batch of images and labels from ADIOS

        # print("rank = ", rank,"this is the ", i, "th call of generator", flush = True)
        images, labels = read_data_from_adios()
        # print("rank = ", rank, ", ", i, "th done", flush = True)
        
        print("rank ", rank, " read_from_adios: ", len(images), "images", flush = True)
        for j in range(len(images)):
            
            image_tensor = tf.convert_to_tensor(images[j])
            label_tensor = tf.convert_to_tensor(labels[j])

            for k in range(NUM_AUGMENTATION + 1):
                if k == 0:
                    yield image_tensor, label_tensor
                else:
                    augmented_image = datagen.random_transform(images[j])
                    augmented_image_tensor = tf.convert_to_tensor(augmented_image)
                    yield augmented_image_tensor, label_tensor


# Create a dataset from ADIOS
train_dataset = tf.data.Dataset.from_generator(
    generator_function, output_types=(tf.float64, tf.int32)
)

#train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
train_dataset = train_dataset.batch(SUB_BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


callbacks=[early_stopping_cb, 
               hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0), 
               hvd.keras.callbacks.MetricAverageCallback(),
               TimestampCallback(train_dataset),]

tic = time.perf_counter()

model.fit(train_dataset,
          steps_per_epoch=NUM_ITER,
          callbacks=callbacks,
          epochs=EPOCHS,
          verbose=1 if hvd.rank() == 0 else 0)


toc = time.perf_counter()

print(f"Training Script executed in {toc - tic:0.4f} seconds")

reader.Close()
