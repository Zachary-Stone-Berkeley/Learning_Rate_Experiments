import numpy as np
import matplotlib.pyplot as plt

# Code by Parag Mital (github.com/pkmital/CADL)
def make_montage(images):

  if isinstance(images, list):
    images = np.array(images)
  img_h = images.shape[1]
  img_w = images.shape[2]
  n_plots = int(np.ceil(np.sqrt(images.shape[0])))
  if len(images.shape) == 4 and images.shape[3] == 3:
    m = np.ones(
      (images.shape[1] * n_plots + n_plots + 1,
       images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
  elif len(images.shape) == 4 and images.shape[3] == 1:
    m = np.ones(
      (images.shape[1] * n_plots + n_plots + 1,
       images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
  elif len(images.shape) == 3:
    m = np.ones(
      (images.shape[1] * n_plots + n_plots + 1,
       images.shape[2] * n_plots + n_plots + 1)) * 0.5
  else:
    raise ValueError('Could not parse image shape of {}'.format(images.shape))
  for i in range(n_plots):
    for j in range(n_plots):
      this_filter = i * n_plots + j
      if this_filter < images.shape[0]:
        this_img = images[this_filter]
        m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
          1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
  return m

def plot_images(images):
  imgs = np.array(images)
  imgs = [img[:,:,0] for img in imgs]
  imgs = make_montage(imgs)
  plt.axis('off')
  plt.imshow(imgs, cmap='gray')
  plt.show()