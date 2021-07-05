import os
import json
import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense, UpSampling2D, \
                         Conv2DTranspose, Concatenate, BatchNormalization, \
                         ZeroPadding2D, LeakyReLU, LocallyConnected2D
from keras.models import Model

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

def architectures(architecture, input_shape, output_shape):

    if architecture == 'MNN':

        inputs = Input(shape = input_shape)
      
        # Scale 2x2
        l1_1 = Conv2D(filters = 50, kernel_size = (2, 2),
                       activation = 'relu', padding = 'same')(inputs)
        l2_1 = Conv2D(filters = 25, kernel_size = (2, 2),
                       activation = 'relu', padding = 'same')(l1_1)
        l3_1 = Conv2D(filters = 10, kernel_size = (2, 2),
                       activation = 'relu', padding = 'same')(l2_1)
      
        # Scale 3x3
        l1_2 = Conv2D(filters = 50, kernel_size = (3, 3),
                       activation = 'relu', padding = 'same')(inputs)
        l2_2 = Conv2D(filters = 25, kernel_size = (3, 3),
                       activation = 'relu', padding = 'same')(l1_2)
        l3_2 = Conv2D(filters = 10, kernel_size = (3, 3),
                       activation = 'relu', padding = 'same')(l2_2)
      
        # Scale 5x5
        l1_3 = Conv2D(filters = 50, kernel_size = (5, 5),
                       activation = 'relu', padding = 'same')(inputs)
        l2_3 = Conv2D(filters = 25, kernel_size = (5, 5),
                       activation = 'relu', padding = 'same')(l1_3)
        l3_3 = Conv2D(filters = 10, kernel_size = (5, 5),
                       activation = 'relu', padding = 'same')(l2_3)
      
        # Scale 7x7
        l1_4 = Conv2D(filters = 50, kernel_size = (7, 7),
                       activation = 'relu', padding = 'same')(inputs)
        l2_4 = Conv2D(filters = 25, kernel_size = (7, 7),
                       activation = 'relu', padding = 'same')(l1_4)
        l3_4 = Conv2D(filters = 10, kernel_size = (7, 7),
                       activation = 'relu', padding = 'same')(l2_4)
      
        # Scale 9x9
        l1_5 =Conv2D(filters = 50, kernel_size = (9, 9),
                      activation = 'relu', padding = 'same')(inputs)
        l2_5 = Conv2D(filters = 25, kernel_size = (9, 9),
                       activation = 'relu', padding = 'same')(l1_5)
        l3_5 = Conv2D(filters = 10, kernel_size = (9, 9),
                       activation = 'relu', padding = 'same')(l2_5)
      
        conctLayer = Concatenate(axis = -1)([l3_1, l3_2, l3_3, l3_4, l3_5])
        
        l3 = UpSampling2D()(conctLayer)
        l4 = Conv2D(filters = 50, kernel_size = (5, 5), strides = 1,
                     activation = 'relu', padding = 'same')(l3)
        l5 = Conv2D(filters = 75, kernel_size = (5, 5), strides = 1,
                     activation = 'relu', padding = 'same')(l4)
        l6 = Conv2D(filters = 100, kernel_size = (5, 5), strides = 1,
                     activation = 'relu', padding = 'same')(l5)
        l7 = UpSampling2D()(l6)
        l8 = Conv2D(filters = 132, kernel_size = (8, 8), strides = 1,
                     activation = 'relu', padding = 'same')(l7)

        l9_1 = Conv2D(filters = 1, kernel_size = (4, 4), strides = 1,
                       activation = 'sigmoid', padding = 'valid')(l8)

        l9_2 = Conv2D(filters = 1, kernel_size = (4, 4), strides = 1,
                       activation = 'linear', padding = 'valid')(l8)

        l9_3 = Conv2D(filters = 1, kernel_size = (4, 4), strides = 1,
                       activation = 'linear', padding = 'valid')(l8)

        l9 = Concatenate()([l9_1, l9_2, l9_3])
      
        model = Model(inputs = inputs, outputs = l9)

        return model

    else:

        print('Select a valid model')
        return None

