import os
import json
import numpy as np
import code.models as models
import code.train as train
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

# Paths
DATA_PATH = './data/'
MODELS_PATH = './models/'

# Convert from rda to numpy
base = importr('base')
base.load(DATA_PATH + 'elements_list.rda')

x = np.array(robjects.r['elements_list'].rx2('x_train'))
y = np.array(robjects.r['elements_list'].rx2('y_train'))

train.train_cnn_horovod(x = x, y = y,
                        model_name = str(robjects.r['elements_list'].rx2('model_name')[0]))
