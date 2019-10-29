import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import powerlaw

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
import keras.backend as K

from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
from tqdm import tqdm_notebook as tqdm
from ast import literal_eval

#from weightwatcher import WeightWatcher
from train import MyModel

def logloghist(eigenvalues, bins=100, alpha=0.25, color='blue', density=True, stacked=True):
  logbins = np.logspace(np.log10(np.min(eigenvalues) - 0.1), np.log10(np.max(eigenvalues) + 0.1), bins)
  plt.hist(eigenvalues, bins=logbins, alpha=alpha, color=color, density=density, stacked=stacked)
  plt.xscale('log')
  plt.yscale('log', nonposy='clip')

def get_keras_matrix(model, layer=2):
  return model.layers[layer].get_weights()[0]

def load_keras_matrix(model, weightfile, layer=2):
  model.load_weights(weightfile)
  return get_keras_matrix(model, layer)

def calc_sigma(Q, evs):
  lmax = np.max(evs)
  inv_sqQ = 1.0/np.sqrt(Q)
  sigma_2 = lmax/np.square(1+inv_sqQ)
  sigma = np.sqrt(sigma_2)
  return sigma

def calc_lplus(Q, sigma):
  quotient = 1/np.sqrt(Q)
  lmax = np.square(sigma)*np.square(1 + quotient)
  return lmax

def get_eigenvalues(model, weightfile, layer=7):
  "Read the keras weightfile, get weights for the layer, and compute the eigenvalues(-1)"
  W = load_keras_matrix(model, weightfile, layer)
  u, sv, vh = svd(W)
  eigenvalues = sv*sv
  return eigenvalues

def marchenko_pastur_pdf(x_min, x_max, Q, sigma=1):
  y=1/Q
  x=np.arange(x_min,x_max,0.001)

  b=np.power(sigma*(1 + np.sqrt(1/Q)),2) # Largest eigenvalue
  a=np.power(sigma*(1 - np.sqrt(1/Q)),2) # Smallest eigenvalue
  return x, (1/(2*np.pi*sigma*sigma*x*y))*np.sqrt((b-x)*(x-a))

def plot_ESD_and_fit(eigenvalues=None, model=None,  weightfile=None, 
                     layer=2,  Q=1.0, num_spikes=0, sigma=None, 
                     alpha=0.25, color='blue', skip=False, verbose=True,
                     loglog=False):
  """Plot histogram of eigenvalues, for Q, and fit Marchenk Pastur.  
  If no sigma, calculates from maximum eigenvalue (minus spikes)
  Can read keras weights from model if specified.  Does not read PyTorch"""
  if eigenvalues is None:
    eigenvalues = get_eigenvalues(model, weightfile, layer)
  
  if loglog:
    logloghist(eigenvalues, bins=100, alpha=alpha, color=color, density=True, stacked=True)
  else:
    print("histo")
    plt.hist(eigenvalues, bins=100, alpha=alpha, color=color, density=True, stacked=True)

  if skip:
    return
  if not (num_spikes):
    num_spikes = 0
      
  # sort, descending order, minus a few max eigenvalues
  evals = np.sort(eigenvalues)[::-1][num_spikes:]

  if (sigma is None):
    sigma=calc_sigma(Q,evals)
      
  percent_mass = 100.0*(num_spikes)/len(evals)

  ev = np.array(evals)
  x_min, x_max =  0, np.max(evals)
  print(x_min, x_max)

  x, mp = marchenko_pastur_pdf(x_min, x_max, Q, sigma)
  if verbose:
    print("% spikes outside bulk {0:.2f}".format(percent_mass))
    print("% sigma {0:.4f}".format(sigma))

  plt.plot(x,mp, linewidth=1, color = 'r', label="MP fit")
  plt.title(" W{} ESD, MP Sigma={}".format(layer,sigma))
  return sigma

def get_shuffled_eigenvalues(W, layer=7, num=100):
  "get eigenvalues for this model, but shuffled, num times"
  
  print("get_shuffled_eigenvalues")
  N, M = W.shape[0], W.shape[1]

  if (N<M):
    N, M = W.shape[1], W.shape[0] 
 
  eigenvalues = []
  for idx in range(num):
    W_shuf = W.flatten()
    np.random.shuffle(W_shuf)
    W_shuf = W_shuf.reshape([N,M])

    u, sv, vh = svd(W_shuf)

    eigenvalues.extend(sv*sv)
      
  evals = (np.array(eigenvalues).flatten())
  return evals

def mp_soft_rank(evals, num_spikes):
  evals = np.array(evals)
  lambda_max = np.max(evals)
  if num_spikes> 0:
    evals = np.sort(evals)[::-1][num_spikes:]
    lambda_plus = np.max(evals)
  else:
    lambda_plus = lambda_max
      
  return lambda_plus/lambda_max


if __name__ == '__main__':

  '''
  python analysis.py --ensemble_size 10 --learning_rate 0.16 --batch_sizes 64 --params '(96, 5, 1, 3, 3)' '(256, 5, 1, 3, 3)' 384 192
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument("--ensemble_size", type=int, default=1, help="integer size of ensemble")
  parser.add_argument("--batch_sizes", '--arg', nargs='+', default=0, type=literal_eval)
  parser.add_argument("--shuffle", type=bool, default=False)
  parser.add_argument("--skip", type=bool, default=True)
  parser.add_argument("--fit_alpha", type=bool, default=False)
  parser.add_argument("--params", nargs='+', type=literal_eval)
  parser.add_argument("--path_to_weights", type=str, default="./Weights/")
  parser.add_argument("--epoch", type=int, default=-1)
  parser.add_argument("--loglog", type=bool, default=False)
  parser.add_argument("--device_id", type=str, default="1", help="gpu device id to use")
  parser.add_argument("--name", type=str, default="Mini")
  parser.add_argument("--learning_rate", type=float, default=0.01)
  args = parser.parse_args()

  #CUDA_VISIBLE_DEVICES = 1
  #os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
  
  with tf.device('/device:GPU:' + args.device_id):
    if args.epoch == -1:
      ENSEMBLE_WEIGHTFILE = args.path_to_weights + "{}/cifar.{}.{}.h5"
    else:
      #ENSEMBLE_WEIGHTFILE = args.path_to_weights + args.dataset + ".{}.{}." + "{:04d}.h5".format(args.epoch)
      #ENSEMBLE_WEIGHTFILE = args.path_to_weights + f"{args.name}-" "AlexNet.{}.{}.h5"
      pass
    LAYER_ID = 7
    BATCH_SIZES = args.batch_sizes
    ALPHA = 0.25

    model = MyModel(params=args.params)
    print(model.layers[LAYER_ID], model.layers[LAYER_ID].output_shape)

    Q = np.prod([val for val in model.layers[LAYER_ID].input_shape if val != None]) / np.prod([val for val in model.layers[LAYER_ID].output_shape if val != None])
    print(Q)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    
    def ensemble_evals(batch_size, layer=LAYER_ID, sort=True):
      """Read weightfiles for ensemble of batchsizes, compute eigenvalues for all ensembles"""
      evals = []

      if args.ensemble_size > 0:
        for idx in range(0, args.ensemble_size):
          if len(args.batch_sizes) > 1:
            weightfile = ENSEMBLE_WEIGHTFILE.format("batch_size", batch_size, idx)
          else:
            weightfile = ENSEMBLE_WEIGHTFILE.format("learning_rate", args.learning_rate, idx)
          evs = get_eigenvalues(model, weightfile, layer=layer)
          evals.extend(evs)
      else:
        if len(args.batch_sizes) > 1:
          weightfile = ENSEMBLE_WEIGHTFILE.format("batch_size", batch_size, 0)
        else:
          weightfile = ENSEMBLE_WEIGHTFILE.format("learning_rate", args.learning_rate, 0)
        evs = get_eigenvalues(model, weightfile, layer=layer)
        evals.extend(evs)
      
      if sort:
        evals =  np.sort(evals)[::-1]
      print("num evals", len(evals))
      return evals  
    
    mp_soft_ranks = []
    for batch_size in BATCH_SIZES:

      eigenvalues = ensemble_evals(batch_size)



      
      if args.shuffle:
        if len(args.batch_sizes) > 1:
          weightfile = ENSEMBLE_WEIGHTFILE.format("batch_size", batch_size, 0)
        else:
          weightfile = ENSEMBLE_WEIGHTFILE.format("learning_rate", args.learning_rate, 0)
        W = load_keras_matrix(model, weightfile, LAYER_ID)
        shuffled_evals = get_shuffled_eigenvalues(W, layer=LAYER_ID, num=100)
        sigma = plot_ESD_and_fit(eigenvalues=shuffled_evals, Q=Q, 
                                          layer=LAYER_ID, num_spikes=0, skip=False)
        plt.show()
        lplus = calc_lplus(Q, sigma)
        print(lplus/np.max(eigenvalues))
        num_spikes = len([val for val in eigenvalues if val > lplus])
      else:
        sigma = None
        num_spikes=0

      if args.fit_alpha:
        fit = powerlaw.Fit(eigenvalues, xmax=np.max(eigenvalues), verbose=False)  
        alpha = fit.alpha
        print(alpha)

      plt.clf()
      plot_ESD_and_fit(eigenvalues, Q=Q, num_spikes=num_spikes, alpha=ALPHA, skip=True, sigma=sigma, loglog=args.loglog)
      plt.title("")
      plt.xlabel(r"Eigenvalues $\lambda$ of $\mathbf{X}=\mathbf{W}}^{T}\mathbf{W}$")
      plt.ylabel("Spectral Density")
      plt.legend()
      plt.tight_layout()
      plt.savefig("./Images/{}.ESD.png".format(str(batch_size)))
      plt.show()

