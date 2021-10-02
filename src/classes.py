#File for the model

import os
import numpy as np
import pickle
import tensorflow as tf


class ConvModule(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, padding='same'):
        super(ConvModule, self).__init__()
        # conv layer
        self.conv = tf.keras.layers.Conv2D(kernel_num, 
                        kernel_size=kernel_size, 
                        strides=strides, padding=padding)

        # batch norm layer
        self.bn   = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        return x


class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size1x1, kernel_size3x3):
        super(InceptionModule, self).__init__()
        
        # two conv modules: they will take same input tensor 
        self.conv1 = ConvModule(kernel_size1x1, kernel_size=(1,1), strides=(1,1))
        self.conv2 = ConvModule(kernel_size3x3, kernel_size=(3,3), strides=(1,1))
        self.cat   = tf.keras.layers.Concatenate()


    def call(self, input_tensor, training=False):
        x_1x1 = self.conv1(input_tensor)
        x_3x3 = self.conv2(input_tensor)
        x = self.cat([x_1x1, x_3x3])
        return x


class DownsampleModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(DownsampleModule, self).__init__()

        # conv layer
        self.conv3 = ConvModule(kernel_size, kernel_size=(3,3), 
                         strides=(2,2), padding="valid") 

        # pooling layer 
        self.pool  = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), 
                         strides=(2,2))
        self.cat   = tf.keras.layers.Concatenate()


    def call(self, input_tensor, training=False):

        # forward pass 
        conv_x = self.conv3(input_tensor, training=training)
        pool_x = self.pool(input_tensor)

        # merged
        return self.cat([conv_x, pool_x])



class MiniInception(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MiniInception, self).__init__()

        # the first conv module
        self.conv_block = ConvModule(96, (3,3), (1,1))

        # 2 inception module and 1 downsample module
        self.inception_block1  = InceptionModule(32, 32)
        self.inception_block2  = InceptionModule(32, 48)
        self.downsample_block1 = DownsampleModule(80)
  
        # 4 inception module and 1 downsample module
        self.inception_block3  = InceptionModule(112, 48)
        self.inception_block4  = InceptionModule(96, 64)
        self.inception_block5  = InceptionModule(80, 80)
        self.inception_block6  = InceptionModule(48, 96)
        self.downsample_block2 = DownsampleModule(96)

        # 2 inception module 
        self.inception_block7 = InceptionModule(176, 160)
        self.inception_block8 = InceptionModule(176, 160)

        # average pooling
        self.avg_pool = tf.keras.layers.AveragePooling2D((7,7))

        # model tail
        self.flat      = tf.keras.layers.Flatten()
        self.classfier = tf.keras.layers.Dense(num_classes, activation='softmax')


    def call(self, input_tensor, training=False, **kwargs):
        
        # forward pass 
        x = self.conv_block(input_tensor)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.downsample_block1(x)

        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.downsample_block2(x)

        x = self.inception_block7(x)
        x = self.inception_block8(x)
        x = self.avg_pool(x)

        x = self.flat(x)
        return self.classfier(x)

    def build_graph(self, raw_shape):
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))