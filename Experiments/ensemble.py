import tensorflow as tf
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from custom_callbacks import WeightAnalyzer, TestCallback, SaverCallback

def ensemble_experiment(args, callable_model):

  (train_batch, train_label), (test_batch, test_label) = tf.keras.datasets.cifar10.load_data()
  train_label, test_label = keras.utils.to_categorical(train_label, num_classes=10), keras.utils.to_categorical(test_label, num_classes=10)

  for batch_size in args.batch_sizes:
    for i in range(args.starting_i, args.starting_i + args.ensemble_size):
      model = callable_model(params=args.params)

      callbacks = []
      if args.plot_ESDs:
        weight_analyzer = WeightAnalyzer(model)
        callbacks.append(weight_analyzer)
      if args.save_checkpoints:
        
        if args.save_each_epoch:
          if len(args.batch_sizes) > 1:
            filepath = './Weights/batch_size/cifar.{}.{}.{{epoch:04d}}.h5'.format(str(batch_size), str(i))
          else:
            filepath = './Weights/learning_rate/cifar.{}.{}.{{epoch:04d}}.h5'.format(str(args.learning_rate), str(i))
          checkpointer = ModelCheckpoint(filepath=filepath, 
                                         verbose=1, 
                                         save_best_only=False,
                                         save_weights_only=True)
        else:
          if len(args.batch_sizes) > 1:
            filepath = './Weights/batch_size/cifar.{}.{}.h5'.format(str(batch_size), str(i))
          else:
            filepath = './Weights/learning_rate/cifar.{}.{}.h5'.format(str(args.learning_rate), str(i))
          checkpointer = ModelCheckpoint(filepath=filepath, 
                                         verbose=1, 
                                         save_best_only=False,
                                         save_weights_only=True)
        callbacks.append(checkpointer)
        
      testcb = TestCallback(model, test_batch, test_label)
      callbacks.append(testcb)
      model.compile(optimizer=keras.optimizers.SGD(lr=args.learning_rate, momentum=0.9), loss='mse', metrics=['accuracy'])
      model.fit(x=train_batch, y=train_label, epochs=args.num_epochs, batch_size=batch_size, callbacks=callbacks)