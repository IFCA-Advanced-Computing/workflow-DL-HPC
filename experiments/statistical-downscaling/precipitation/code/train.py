import os
import json
import numpy as np
from time import time
from keras.layers import Input, Conv2D, Flatten, Dense, UpSampling2D, \
                         Conv2DTranspose, Concatenate, BatchNormalization, \
                         ZeroPadding2D, LeakyReLU, LocallyConnected2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras import optimizers
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import code.models as models
import horovod.keras as hvd

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

def gpuConfig_TF2(horovod = False):
    '''
    If a RTX GPU is available set some parameters needes to avoid CUDNN issues.
    '''

    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    if (horovod):
        tf.config.experimental.set_visible_devices(gpu_devices[hvd.local_rank()], 'GPU')

def custom_loss(true, pred): 
    '''
    Custom loss function used to reproduce the distribution of precipitations
    By minimizing this function we are minimizing the negative log-likelihood
    of a Bernoulli-Gamma distribution
    (Tensorflow 2.X.X)
    '''

    ocurrence = pred[:,:,:,0] # p parameter
    shape_parameter = K.exp(pred[:,:,:,1]) # shape parameter
    scale_parameter = K.exp(pred[:,:,:,2]) # beta parameter

    bool_rain = K.cast(K.greater(true, 0), 'float32')
    epsilon = 0.000001 # avoid nan values

    noRainCase = (1 - bool_rain) * K.log(1 - ocurrence + epsilon) # no rain
    rainCase = bool_rain * (K.log(ocurrence + epsilon) + # rain
                           (shape_parameter - 1) * K.log(true + epsilon) -
                           shape_parameter * K.log(scale_parameter + epsilon) -
                           tf.math.lgamma(shape_parameter + epsilon) -
                           true / (scale_parameter + epsilon))

    return - K.mean(noRainCase + rainCase) # loss output

# Time log callback
class timeLogging(Callback):

    def __init__(self, batch_size, logs = {}):
        self.batch_times = []
        self.epoch_times = []
        self.train_times = []
        self.images_sec = []

        self.batch_size = batch_size
        self.batches_per_epoch = 0


    def on_batch_begin(self, batch, logs = {}):
        self.start_time_batch = time()

    def on_batch_end(self, batch, logs = {}):
        self.batch_times.append(time() - self.start_time_batch)
        self.batches_per_epoch += 1

    def on_epoch_begin(self, epoch, logs = {}):
        self.start_time_epoch = time()

    def on_epoch_end(self, epoch, logs = {}):
        self.epoch_times.append(time() - self.start_time_epoch)
        self.images_sec.append(self.batch_size * self.batches_per_epoch /
                               self.epoch_times[-1])
        self.batches_per_epoch = 0

    def on_train_begin(self, logs = {}):
        self.start_time_train = time()

    def on_train_end(self, logs = {}):
        self.train_times.append(time() - self.start_time_train)

def train_cnn_horovod(x, y, model_name):
    '''
    Train the specified model on multi GPU environments
    '''

    # Inititalize Horovod
    hvd.init()

    # Make GPUs visible
    gpuConfig_TF2(horovod = True)

    # Distribute batch size between workers
    # Batch size per worker
    batch_size = 32 # batch_size * GPUs

    nbatch = (x.shape[0] // (batch_size * hvd.size())) + 1
    print('# Batches: ' + str(nbatch))

    timeLog = timeLogging(batch_size = batch_size)

    callbacks =  [hvd.callbacks.MetricAverageCallback(),
                  timeLog,
                  hvd.callbacks.BroadcastGlobalVariablesCallback(0)] # initial values from rank 0

    loss_function = custom_loss

    # Model training
    print('Num of GPUs: ' + str(hvd.size()))
    opt = optimizers.Adam(lr = 0.0001 * hvd.size())

    opt = hvd.DistributedOptimizer(opt)

    # Load model
    model = models.architectures(architecture = model_name,
                                 input_shape = x.shape[1:], output_shape = y.shape[1])
    print(model.summary())

    # Train model
    model.compile(loss = loss_function, optimizer = opt)
    hist = model.fit(x, y, epochs = 200, batch_size = batch_size,
                     steps_per_epoch = nbatch,
                     callbacks = callbacks,
                     verbose = 1)

    img_sec_mean = np.mean(timeLog.images_sec[1:])
    img_sec_conf = 1.96 *  np.std(timeLog.images_sec[1:])

    print('')
    print('*****************************************************')
    print('Mean batch time (sec): ' +
          str(round(np.mean(timeLog.batch_times), 4)))
    print('Mean epoch time (sec): ' +
          str(round(np.mean(timeLog.epoch_times), 4)))
    print('Number of epochs: ' + str(len(timeLog.epoch_times)))
    print('Training time (sec): ' +
          str(round(timeLog.train_times[0], 4)))
    print('')
    print('Final loss on training set: ' +
          str(round(np.min(hist.history['loss']), 3)))
    print('')
    print('Images per second (1 GPU): ' +
          str(round(img_sec_mean, 3)) + ' +- ' +
          str(round(img_sec_conf, 2)))
    print('Images per second (' + str(hvd.size()) + ' GPUs): ' +
          str(round(img_sec_mean * hvd.size(), 3)) + ' +- ' +
          str(round(img_sec_conf * hvd.size(), 2)))
    print('*****************************************************')
    print('')

    print(hist.history['loss'])

    return model
