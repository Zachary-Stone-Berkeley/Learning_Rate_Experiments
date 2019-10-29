import sys
sys.path.append('../../294-58/Models/')
sys.path.append('../../294-58/Utils/')
sys.path.append('../../294-58/Data Pipeline/')
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

from weightwatcher import WeightWatcher
from model import MyModel

from test import ConstrainedMNIST
import CustomGANs

def logloghist(eigenvalues, bins=100, alpha=0.25, color='blue', density=True, stacked=True):
  logbins = np.logspace(np.log10(np.min(eigenvalues) - 0.1), np.log10(np.max(eigenvalues) + 0.1), bins)
  plt.hist(eigenvalues, bins=logbins, alpha=alpha, color=color, density=density, stacked=stacked)
  plt.xscale('log')
  plt.yscale('log', nonposy='clip')

def get_conv2D_Wmats(Wtensor):
  """Extract W slices from a 4 index conv2D tensor of shape: (N,M,i,j) or (M,N,i,j).  Return ij (N x M) matrices"""
  Wmats = []
  s = Wtensor.shape
  N, M, imax, jmax = s[0],s[1],s[2],s[3]
  #print("tensor shape", N,M,imax,jmax)
  for i in range(imax):
    for j in range(jmax):
      W = Wtensor[:,:,i,j]
      if N < M:
        W = W.T
      Wmats.append(W)
  return Wmats

def get_tf_matrix(model, var_name):
  var = [var for var in tf.trainable_variables() if var.name[:-2] == var_name][0]
  print(var.shape)
  return model.sess.run(var)

def load_tf_matrix(model, weightfile, layer=2):
  model.load_weights(weightfile)
  return get_tf_matrix(model, layer)

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
  W = load_tf_matrix(model, weightfile, layer)
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
                     loglog=True):
  """Plot histogram of eigenvalues, for Q, and fit Marchenk Pastur.  
  If no sigma, calculates from maximum eigenvalue (minus spikes)
  Can read keras weights from model if specified.  Does not read PyTorch"""
  if eigenvalues is None:
    eigenvalues = get_eigenvalues(model, weightfile, layer)
  
  if loglog:
    logloghist(eigenvalues, bins=100, alpha=alpha, color=color, density=True, stacked=True)
  else:
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

  def identity(x):
    return literal_eval(x)

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default="mnist", help="mnist or cifar")
  parser.add_argument("--ensemble_size", type=int, default=1, help="integer size of ensemble")
  parser.add_argument("--shuffle", type=bool, default=False)
  parser.add_argument("--skip", type=bool, default=True)
  parser.add_argument("--fit_alpha", type=bool, default=False)
  parser.add_argument("--params", nargs='+', type=identity)
  parser.add_argument("--path_to_weights", type=str, default="../../../Desktop/weights/")
  parser.add_argument("--epoch", type=int, default=-1)
  parser.add_argument("--loglog", type=bool, default=False)
  parser.add_argument("--device_id", type=str, default="1", help="gpu device id to use")

  parser.add_argument("--noise_dim", type=int, default=49, help="the dimension of the latent code z for the generator")
  parser.add_argument("--noise_type", type=str, default="normal", help="the latent noise type, one of \"normal\", \"uniform\", or \"truncated\"")
  parser.add_argument("--loss_type", type=str, default="hinge", help="the loss type, one of \"wasserstein\", \"hinge\", or \"KL\"")
  parser.add_argument("--learning_rate", type=float, default=0.00015, help="learning rate")
  parser.add_argument("--batch_size", type=int, default=64, help="batch size for training and testing")
  parser.add_argument("--num_steps", type=int, default=6001, help="number of training steps")
  args = parser.parse_args()
    
  if args.ensemble_size == 1:
    ENSEMBLE_WEIGHTFILE = "../../../Desktop/mnist_gan/my_model_{}.ckpt"
  else:
    ENSEMBLE_WEIGHTFILE = None
  #LAYER_ID = "mnist_gan/generator/linear_01/kernel"
  LAYER_ID = "mnist_gan/discriminator/logits/kernel"
  ALPHA = 0.25

  sess = tf.Session()
  model = CustomGANs.MnistGAN(data_set="mnist", 
                         batch_size=args.batch_size, 
                         noise_dim=args.noise_dim, 
                         loss_type=args.loss_type,
                         noise_type=args.noise_type,
                         learning_rate=args.learning_rate,
                         sess=sess,
                         name="mnist_gan")

  for var in tf.trainable_variables():
    print(var.name)
  
  #print(model.layers[LAYER_ID], model.layers[LAYER_ID].output_shape)
  #Q = np.prod([val for val in model.layers[LAYER_ID].input_shape if val != None]) / np.prod([val for val in model.layers[LAYER_ID].output_shape if val != None])
  #print(Q)
  Q = 1

  
  def ensemble_evals(epoch, layer=LAYER_ID, sort=True):
    """Read weightfiles for ensemble of batchsizes, compute eigenvalues for all ensembles"""
    evals = []
    if args.ensemble_size == 1:
      weightfile = ENSEMBLE_WEIGHTFILE.format(epoch)
      evs = get_eigenvalues(model, weightfile, layer=layer)
      evals.extend(evs)
    else:
      for idx in range(0, args.ensemble_size):
        weightfile = ENSEMBLE_WEIGHTFILE.format(epoch, idx)
        evs = get_eigenvalues(model, weightfile, layer=layer)
        evals.extend(evs)
    
    if sort:
      evals =  np.sort(evals)[::-1]
    return evals  
  
  mp_soft_ranks = []
  for epoch in [1100]:#range(0, args.num_steps, 100):
    print(epoch)

    eigenvalues = ensemble_evals(epoch)
    
    if args.shuffle:
      weightfile = ENSEMBLE_WEIGHTFILE.format(epoch, 1)
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
    print(num_spikes)
    plot_ESD_and_fit(eigenvalues, Q=Q, num_spikes=num_spikes, alpha=ALPHA, skip=True, sigma=sigma, loglog=args.loglog)
    plt.title("")
    plt.xlabel(r"Eigenvalues $\lambda$ of $\mathbf{X}=\mathbf{W}}^{T}\mathbf{W}$")
    plt.ylabel("Spectral Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./img/epoch_{}.ESD.png".format(str(epoch)))
    plt.show()

