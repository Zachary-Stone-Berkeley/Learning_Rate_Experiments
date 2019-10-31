import sys
sys.path.append("./Experiments/")
sys.path.append("./Utils/")
import numpy as np
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, SpatialDropout2D, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from ast import literal_eval
#from weightwatcher import WeightWatcher

from ensemble import ensemble_experiment

# CIFAR Model from RMT_Util in ImplicitSelfRegularization-master
def MyModel(batchnorm=False, 
            params=[(96, 5, 1, 3, 3), (256, 5, 1, 3, 3), 384, 192], 
            activation='relu', 
            initializer='glorot_normal', 
            alpha=0., 
            n_samples=10, 
            dropout=False):
  
  model = Sequential()

  if initializer == "zeros":
    initializer = keras.initializers.Zeros()
  elif initializer == "ones":
    initializer = keras.initializers.Ones()
  elif initializer == "uniform":
    initializer = keras.initializers.RandomUniform()
  elif initializer == "normal":
    initializer = keras.initializers.RandomNormal()
  elif initializer == "orthogonal":
    initializer = keras.initializers.Orthogonal()
  elif initializer == "custom_alpha":
    initializer = custom_init(alpha)
  elif initializer == "custom_uniform":
    initializer = my_uniform(n_samples)

  model.add(Conv2D(filters=params[0][0], kernel_size=(params[0][1], params[0][1]), strides=(params[0][2], params[0][2]), input_shape=(32, 32, 3), kernel_initializer=
           initializer, bias_initializer=Constant(0.1), padding=
           'same', activation=activation))
  model.add(MaxPooling2D((params[0][3], params[0][3]), (params[0][4], params[0][4]), padding='same'))
  model.add(BatchNormalization())
  if dropout:
    model.add(Dropout(0.2))
  
  model.add(Conv2D(filters=params[1][0], kernel_size=(params[1][1], params[1][1]), strides=(params[1][2], params[1][2]), kernel_initializer=initializer,
           bias_initializer=Constant(0.1), padding='same',
           activation=activation)) 
  model.add(MaxPooling2D((params[1][3], params[1][3]), (params[1][4], params[1][4]), padding='same'))
  model.add(BatchNormalization())
  if dropout:
    model.add(Dropout(0.2))
  
  model.add(Flatten())
  model.add(Dense(params[2], kernel_initializer=initializer,
          bias_initializer=Constant(0.1), activation=activation))
  if batchnorm:
    model.add(BatchNormalization())
  if dropout:
    model.add(Dropout(0.2))
  
  model.add(Dense(params[3], kernel_initializer=initializer,
          bias_initializer=Constant(0.1), activation=activation))
  if batchnorm:
    model.add(BatchNormalization())

  model.add(Dense(10, kernel_initializer=initializer,
          bias_initializer=Constant(0.1), activation='softmax'))

  for layer in model.layers:
    print(layer, layer.input_shape, layer.output_shape)

  return model

if __name__ == '__main__':
  '''
  python train.py --ensemble_size 1 --num_epochs 100 --batch_sizes 64     --learning_rate 0.16 --starting_i 0 --device_id 0     --params '(96, 5, 1, 3, 3)' '(256, 5, 1, 3, 3)' 384 192
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument("--ensemble_size", type=int, default=100, help="integer size of ensemble")
  parser.add_argument("--save_checkpoints", type=bool, default=True, help="Bool to save checkpoints or not")
  parser.add_argument("--save_each_epoch", type=bool, default=False, help="Do you wish to save each epoch or only the last.")
  parser.add_argument("--plot_ESDs", type=bool, default=False, help="Bool to plot ESDs or not")
  parser.add_argument("--num_epochs", type=int, default=6, help="number of training epochs")
  parser.add_argument("--batch_sizes", '--arg', nargs='+', default=0, type=int)
  parser.add_argument("--starting_i", type=int, default=0, help="starting iteration number")
  parser.add_argument("--device_id", type=str, default="0", help="gpu device id to use")
  parser.add_argument("--learning_rate", type=float, default=0.16, help="learning rate")
  parser.add_argument("--params", nargs='+', type=literal_eval)
  args = parser.parse_args()
  # 0.01.6 and 0.01.1 need to be redone... fml
  with tf.device('/device:GPU:' + args.device_id):
    if len(args.batch_sizes) == 1:
      for lr in [0.00128, 0.00256, 0.0064, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]:
        args.learning_rate = lr
        ensemble_experiment(args, MyModel)
    else:
      ensemble_experiment(args, MyModel)

    '''
    python train.py --ensemble_size 5 --num_epochs 100 --batch_sizes 64 \
    --learning_rate 0.0064 --starting_i 0 --device_id 0 \
    --params '(96, 5, 1, 3, 3)' '(256, 5, 1, 3, 3)' 384 192

    python train.py --ensemble_size 5 --num_epochs 100 --batch_sizes 64 \
    --learning_rate 0.0064 --starting_i 5 --device_id 1 \
    --params '(96, 5, 1, 3, 3)' '(256, 5, 1, 3, 3)' 384 192
    '''
