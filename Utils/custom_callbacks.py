import keras
import numpy as np

class WeightAnalyzer(keras.callbacks.Callback):

  def __init__(self, model):
    self.ww = WeightWatcher(model)
    self.model = model
    super(WeightAnalyzer, self).__init__()
  
  def on_epoch_end(self, epoch, logs=None):
    for layer in self.model.layers:
      if isinstance(layer, keras.layers.core.Dense):
        weights = layer.get_weights()
        self.ww.analyze_weights(weights=[weights[0]], compute_alphas=True, plot=True)

# from github.com/keras-team/keras/issues/2548 user joelthchao
class TestCallback(keras.callbacks.Callback):

  def __init__(self, model, test_inputs, test_labels, start_epoch=100, verbose=True, storage=None, max_length=10):
    self.model = model
    self.inputs = test_inputs
    self.labels = test_labels
    self.verbose = verbose
    self.storage = storage
    self.curr_epoch = 0
    self.start_epoch = start_epoch
    self.max_length = max_length
    super(TestCallback, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    self.curr_epoch += 1
    loss, acc = self.model.evaluate(self.inputs, self.labels, verbose=0)
    if self.verbose and self.curr_epoch % 1 == 0:
      print("\nTesting loss: {}, acc: {}\n".format(loss, acc))
    if self.storage != None and self.curr_epoch > self.start_epoch:
      if len(self.storage) >= self.max_length:
        self.storage.pop(0)
      self.storage.append(acc)

class TrueAccCallback(keras.callbacks.Callback):

  def __init__(self, model, inputs, labels):
    self.model = model
    self.inputs = inputs
    self.labels = labels
    super(TrueAccCallback, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    loss, acc = self.model.evaluate(self.inputs, self.labels, verbose=0)
    print(f"Accuracy: {acc}")


class ForgetfullnessCallback(keras.callbacks.Callback):

  def __init__(self, model, inputs, labels, iters_per_update, storage=None, max_length=50000):
    self.model = model
    self.inputs = inputs
    self.n_instances = len(inputs)

    self.onehot_labels = labels
    #self.total_gained = 0

    self.labels = np.argmax(labels, axis=1)
    self.learned = [0 for _ in self.inputs]
    self.storage = storage
    self.iters_per_update = iters_per_update
    self.max_length = max_length
    self.curr_epoch = 0

    super(ForgetfullnessCallback, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    self.curr_epoch += 1

  def on_batch_end(self, batch, logs=None):
    if batch % self.iters_per_update == 0 and batch == 0 and self.curr_epoch == 0:

      forgotten = 0
      learned = 0

      preds = np.argmax(self.model.predict(self.inputs), axis=1)
      correct = np.equal(preds, self.labels)
      forgetfullnes_vector = correct - self.learned

      for entry in forgetfullnes_vector:
        if entry == 1:
          learned += 1
        elif entry == -1:
          forgotten += 1
      
      loss, acc = self.model.evaluate(self.inputs, self.onehot_labels, verbose=0)
      #print(f"\nEpoch {self.curr_epoch} | Batch {batch}:\n Learned {learned} datapoints and forgot {forgotten} datapoints. Accuracy is {acc}. Total gained is {self.total_gained}.")
      
      left_to_learn = sum([1 for val in self.learned if val == 0])
      already_learned = sum([1 for val in self.learned if val == 1])
      gained = learned - forgotten#/left_to_learn if left_to_learn > 0 else 0 # you know the minor issue
   
      print("\n\n", learned, forgotten, gained, acc, "\n\n")
      #learned = learned/left_to_learn if left_to_learn > 0 else 0
      #forgotten = forgotten/already_learned if already_learned > 0 else 0
      
      #learned /= self.n_instances
      #forgotten /= self.n_instances
      
      if self.storage != None:
        if len(self.storage[0]) >= self.max_length:
          self.storage[0].pop(0)
          self.storage[1].pop(0)
          self.storage[2].pop(0)
        self.storage[0].append(learned)
        self.storage[1].append(forgotten)
        self.storage[2].append(gained)
        #print(learned, forgotten)

      self.learned = [1 if val == True else 0 for val in correct.tolist()]

class SaverCallback(keras.callbacks.Callback):

  def __init__(self, model, iters_per_save, model_name, save_req=lambda x: True):
    self.model = model
    self.iters_per_save = iters_per_save
    self.model_name = model_name
    self.curr_epoch = 0
    self.save_req = save_req
    super(SaverCallback, self).__init__()

  def on_epoch_end(self, epoch, logs=None):
    self.curr_epoch += 1

  def on_batch_end(self, batch, logs=None):
    if batch % self.iters_per_save == 0 and self.save_req(batch) and self.iters_per_save != -1:
      self.model.save(f"./saves/{self.model_name}.{self.curr_epoch}.{batch}.h5")
