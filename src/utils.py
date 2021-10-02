import os
import numpy as np
import pickle
import time
import tensorflow as tf





def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # training set / data
    x_train = x_train.astype('float32') / 255
    # validation set / data 
    x_test = x_test.astype('float32') / 255
    # target / class name
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test 



def process_data(x_train, y_train, x_test, y_test, batch_size=128, buffer_size=1024):

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    return train_dataset, val_dataset