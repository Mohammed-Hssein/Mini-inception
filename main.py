#main script to launch training

import os, pickle, time
import argparse
import numpy as np
import tensorflow as tf
from src.classes import *
from src.utils import *



def mk_tmp():
    """
    """
    dirname="tmp"
    path = os.path.join(os.getcwd(), dirname)
    os.makedirs(path, exist_ok=True)
    return

def mk_tensorboard():
    """
    """
    dirname="tb-logs"
    path_train = os.path.join(os.getcwd(),os.path.join("tmp", os.path.join(dirname, "train")))
    path_val = os.path.join(os.getcwd(),os.path.join("tmp", os.path.join(dirname, "val")))
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_val, exist_ok=True)
    return

def mk_model():
    """
    make model directory
    """
    dirname="model"
    path_model = os.path.join(os.getcwd(),os.path.join("tmp", dirname))
    os.makedirs(path_model, exist_ok=True)
    return path_model

def save_model(model):
    """
    """
    path = mk_model()
    model.save(path, save_format="tf")
    print("MODEL SAVED SUCCESSFULLY !")
    return

def save_training_info(train_loss, val_loss):
    """
    """
    import pickle
    path = 'tmp/artefacts'
    os.makedirs(path, exist_ok=True)
    train_list = os.path.join(path, "training.pkl")
    with open(train_list, "wb") as f:
        pickle.dump(train_loss, f)
    
    val_list = os.path.join(path, 'validation.pkl')
    with open(val_list, "wb") as f:
        pickle.dump(val_loss, f)
    
    return

def prepare_tensorboard_writers():
    """
    """
    path_train = os.path.join(os.getcwd(),os.path.join("tmp", os.path.join("tb-logs", "train")))
    path_test = os.path.join(os.getcwd(),os.path.join("tmp", os.path.join("tb-logs", "val")))
    train_writer = tf.summary.create_file_writer(path_train)
    test_writer = tf.summary.create_file_writer(path_test)
    return train_writer, test_writer

#Defining some variables
#Instantiate an optimizer to train the model.
optimizer = tf.keras.optimizers.Adam()

#Instantiate a loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

#Prepare the metrics
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

#tensorboard writer 
#train_writer = tf.summary.create_file_writer('logs/train/')
#test_writer  = tf.summary.create_file_writer('logs/test/')


def main(model_object, training_data, validation_data, writers, num_epochs, freq_train, freq_val):
    """
    main loop
    """
    num_batch_train = len(list(training_data))
    num_batch_val = len(list(validation_data))
    training_values = list()
    training_values_batch = list()
    validation_values = list()
    validation_values_batch = list()
    time_per_epoch = list()
    train_writer = writers[0]
    val_writer = writers[-1]

    for epoch in range(num_epochs):
        start_time = time.time()
        #iterate on batches of data
        for batch_train_step, (x_batch_train, y_batch_train) in enumerate(training_data):
            #forward pass
            with tf.GradientTape() as tape :
                logits = model_object(x_batch_train, training=True)
                train_loss_value = loss_fn(y_batch_train, logits)
        
            training_values_batch.append(train_loss_value)
            #printing options
            ratio_batch = batch_train_step/num_batch_train
            ratio_epoch = (epoch+1)/num_epochs
            template = "epoch:{} - batch loop:{} - train_loss:{}"
            if batch_train_step%freq_train == 0:
                print(template.format(
                        str(epoch+1) + "(" + str(round(100*ratio_epoch, 4)) + "%)",
                        str(batch_train_step) + "(" + str(round(100*ratio_batch, 4)) + "%)",
                        str(round(train_loss_value.numpy(), 4))
                ))
            else: 
                continue
            #compute gradients
            grads = tape.gradient(train_loss_value, model_object.trainable_weights)
            #update weights
            optimizer.apply_gradients(zip(grads, model_object.trainable_weights))

            #Write to tensorboard ....
            with train_writer.as_default():
                batch_train_step = tf.convert_to_tensor(batch_train_step, dtype=tf.int64)
                tf.summary.scalar('loss', train_loss_value, step=batch_train_step)
                tf.summary.scalar('accuracy', train_acc_metric.result(), step=batch_train_step)
    
        training_values.append(sum(training_values_batch)/len(training_values_batch))
        #validation loss
        for batch_val_step, (x_batch_val, y_batch_val) in enumerate(validation_data):
            val_logits = model_object(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            validation_values_batch.append(val_loss_value)

            #printing options
            ratio_batch = batch_val_step/num_batch_val
            ratio_epoch = (epoch+1)/num_epochs
            template = "epoch:{} - batch loop:{} - validation_loss:{}"
            if batch_val_step%freq_val == 0:
                print(template.format(
                        str(epoch+1) + "(" + str(round(100*ratio_epoch, 4)) + "%)",
                        str(batch_val_step) + "(" + str(round(100*ratio_batch, 4)) + "%)",
                        str(round(val_loss_value.numpy(), 4))
                ))
            else: 
                continue
            #Write to tensorboard        
            with val_writer.as_default():
                batch_val_step = tf.convert_to_tensor(batch_val_step, dtype=tf.int64)
                tf.summary.scalar('val loss', val_loss_value, step=batch_val_step)
                tf.summary.scalar('val accuracy', val_acc_metric.result(), step=batch_val_step)

        validation_values.append(sum(validation_values_batch)/len(validation_values_batch))


        end_time = time.time()
        time_per_epoch.append(round((end_time - start_time)/60, 3))
    
    #function to save the model
    save_model(model=model_object)
    #function to plot the losses and save them
    try:
        print("Saving losses history ...")
        save_training_info(train_loss=training_values, val_loss=validation_values)
    except :
        print("Failed !")
    return


if __name__ == '__main__':
    #parse arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, help="epochs to use to train the network")
    parser.add_argument('--batch-size', dest='batch_size', type=int, help='batch size to use')
    parser.add_argument('--buffer-size', dest='buffer_size', type=int, default=1024, help="buffer size")
    parser.add_argument('--freq-display-train',
                    dest='freq_display_train',
                    type=int, 
                    default=20,
                    help="Frequency of printing the training loss for monitoring purpose")
    parser.add_argument('--freq-display-val',
                    dest="freq_display_val",
                    type=int,
                    default=40,
                    help="Frequency of printing the validation loss")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    buffer_size=args.buffer_size
    freq_train = args.freq_display_train
    freq_val = args.freq_display_val

    print("Started ! \n")
    print("making tmp folder")
    mk_tmp()
    print("making tensorboard folder ... ")
    mk_tensorboard()
    print("defining model ... ")
    inception_model = MiniInception()

    print("Starting collecting data ... ")
    x_train, y_train, x_test, y_test = load_data()
    train_dataset, val_dataset = process_data(x_train=x_train,
                                            y_train=y_train,
                                            x_test=x_test,
                                            y_test=y_test,
                                            batch_size=batch_size,
                                            buffer_size=buffer_size)

    print("Preparing tensorboard writers .... ")
    train_writer, test_writer = prepare_tensorboard_writers()
    print("Start training .... ")
    main(model_object=inception_model, 
        training_data=train_dataset, 
        validation_data=val_dataset, 
        num_epochs=epochs,
        writers=(train_writer, test_writer), 
        freq_train=freq_train, 
        freq_val=freq_val)

    print("done !") 