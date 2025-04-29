import tensorflow as tf

def make_model():
    width, height = (224, 224)
    model = tf.keras.models.Sequential()
    channels = 64
    model.add(tf.keras.layers.Input(shape = (width, height, 3)))
    model.add(tf.keras.layers.Conv2D(channels, (7, 7), 
                                     padding = 'same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, kernel_regularizer = tf.keras.regularizers.l2(0.01)))
    model.add(tf.keras.layers.Dense(1))
    return model