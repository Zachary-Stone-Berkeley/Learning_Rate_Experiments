# On Learning Rate and the Spectral Distributions of Neural Network Weight Matrices

A recent and very interesting paper by Mike Mahoney and Charles Martin looked at the distributions of singular values of weight matrices in state of the art (SotA) deep neural network models. In the extensive paper, five primary stages of training are proprosed, corrosponding to increasingly long-term correlations in the model's weight matrices. SotA models were found to possess heavy-tailed spectral weight distributions which is the fifth stage of training. It was further demonstrated that all five stages of training can be induced in the mini-AlexNet model, trained on Cifar-10, by adjusting the batch size.



## Prerequisites

python (3.6.1)  
tf-nightly-gpu (1.13.0.dev20190221)  
numpy (1.12.1)  
scikit-learn (0.18.1)  

Note: Newer versions of scikit-learn will raise an error as it is no longer possible to pass a 1D array to one-hot encoder.

## Running the Code

To train an ensemble, run

```
python train.py --args
```

where possible args are

```
--ensemble_size: Int. The size of the ensemble.
--num_epochs: Int. The number of epochs to train for.
--batch_sizes: List. An esemble will be trained for each provided batch size.
--learning_rate: Float. The learning rate for training.
--params: List. The model's parameters. Experiments were done with '(96, 5, 1, 3, 3)' '(256, 5, 1, 3, 3)' 384 192.
```

To analyze the weight matrices, run

```
python analysis.py --args
```

where possible args are those above.

## Credits

The arch_ops and losses files are modified from https://github.com/google/compare_gan.  
The functions in `anaylsis.py` for computing the spectral distributions are from https://github.com/CalculatedContent/WeightWatcher.
