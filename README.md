# On Learning Rate and the Spectral Distributions of Neural Network Weight Matrices

A recent and very interesting paper (https://arxiv.org/abs/1810.01075) by Michael Mahoney and Charles Martin looked at the distribution of singular values in the weight matrices in state of the art (SotA) deep neural network models. In the extensive paper, five primary stages of training are proposed, corresponding to increasingly long-term correlations in the model's weight matrices. SotA models were found to possess heavy-tailed spectral weight distributions which is the fifth stage of training. It was further demonstrated that all five stages of training can be induced in the mini-AlexNet model, trained on Cifar-10, by adjusting the batch size. In particular, lower batch sizes produced more heavy-tailed empirical spectral distributions, and better generalization.
  
However, as a practical improvement, lowering the batch size has limited use. With a batch size of 64, the mini-AlexNet model takes roughly six seconds to complete an epoch on a GTX 1080 GPU. With a batch size of 2, which produced the most heavy-tailed empirical spectral distribution, it would take several minutes to complete an epoch. Thus, one might wonder what else affects the spectral weight distributions in deep neural networks, and if the effect of decreasing batch size can be observed by changing some other parameter without increasing training time.
  
The most natural hyper-parameter to vary instead of batch size is learning rate. However, we expect the effect to be inverse. Lowering the batch size will increase the noise in the weight updates. For learning rate, it is an increase that adds noise to the weight updates. The code in this repo can be used to train a mini-AlexNet model with varying batch size or learning rate, and then, to analyze the resulting weight matrices.
  
Below are the empirical spectral distributions of the first fully connected layer in the mini-AlexNet model, after training for 100 epochs, with batch size 64 and respective learning rates 0.00128, 0.01, 0.02, 0.08, and 0.16 which correspond to randomlike, bleeding out, bulk+spike, bulk-decay, and heavy-tailed distributions.

![alt text](Images/Stage%201.%20Randomlike/64.00128.png)
![alt text](Images/Stage%202.%20Bleeding%20Out/64.01.png)
![alt text](Images/Stage%203.%20Bulk%20%2B%20Spikes/64.02.png)
![alt text](Images/Stage%204.%20Bulk%20Decay/64.08.png)
![alt text](Images/Stage%205.%20Heavy%20Tailed/64.16.png)

As observed by Mahoney and Martin, it is the models with more heavy-tailed empirical spectral distributions that generalize better. The mini-AlexNet model with learning rate 0.00128, 0.01, and 0.16 achieves best test accuracy 71%, 74.5%, 78.0% (training accuracies are 98%, 100%, 100%). However, compared to lowering the batch size, increasing the learning rate does not significantly increase training time.

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
