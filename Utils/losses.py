import tensorflow as tf

def check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits):
  """Checks the shapes and ranks of logits and prediction tensors.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].

  Raises:
    ValueError: if the ranks or shapes are mismatched.
  """
  def _check_pair(a, b):
    if a[1:] != b[1:]:
      raise ValueError("Shape mismatch: %s vs %s." % (a, b))
    if len(a) != 2 or len(b) != 2:
      raise ValueError("Rank: expected 2, got %s and %s" % (len(a), len(b)))

  if (d_real is not None) and (d_fake is not None):
    _check_pair(d_real.shape.as_list(), d_fake.shape.as_list())
  if (d_real_logits is not None) and (d_fake_logits is not None):
    _check_pair(d_real_logits.shape.as_list(), d_fake_logits.shape.as_list())
  if (d_real is not None) and (d_real_logits is not None):
    _check_pair(d_real.shape.as_list(), d_real_logits.shape.as_list())


def non_saturating(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Non-saturating loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("non_saturating_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_real_logits, labels=tf.ones_like(d_real_logits),
        name="cross_entropy_d_real"))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits),
        name="cross_entropy_d_fake"))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
        name="cross_entropy_g"))
    return d_loss, d_loss_real, d_loss_fake, g_loss

def wasserstein(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for Wasserstein loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("wasserstein_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = -tf.reduce_mean(d_real_logits)
    d_loss_fake = tf.reduce_mean(d_fake_logits)
    d_loss = d_loss_real + d_loss_fake
    g_loss = -d_loss_fake
    return d_loss, d_loss_real, d_loss_fake, g_loss

def least_squares(d_real, d_fake, d_real_logits=None, d_fake_logits=None):
  """Returns the discriminator and generator loss for the least-squares loss.

  Args:
    d_real: prediction for real points, values in [0, 1], shape [batch_size, 1].
    d_fake: prediction for fake points, values in [0, 1], shape [batch_size, 1].
    d_real_logits: ignored.
    d_fake_logits: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("least_square_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.square(d_real - 1.0))
    d_loss_fake = tf.reduce_mean(tf.square(d_fake))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    g_loss = 0.5 * tf.reduce_mean(tf.square(d_fake - 1.0))
    return d_loss, d_loss_real, d_loss_fake, g_loss

def hinge(d_real_logits, d_fake_logits, d_real=None, d_fake=None):
  """Returns the discriminator and generator loss for the hinge loss.

  Args:
    d_real_logits: logits for real points, shape [batch_size, 1].
    d_fake_logits: logits for fake points, shape [batch_size, 1].
    d_real: ignored.
    d_fake: ignored.

  Returns:
    A tuple consisting of the discriminator loss, discriminator's loss on the
    real samples and fake samples, and the generator's loss.
  """
  with tf.name_scope("hinge_loss"):
    check_dimensions(d_real, d_fake, d_real_logits, d_fake_logits)
    d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - d_real_logits))
    d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + d_fake_logits))
    d_loss = d_loss_real + d_loss_fake
    g_loss = - tf.reduce_mean(d_fake_logits)
    return d_loss, d_loss_real, d_loss_fake, g_loss

def wgangp_penalty(discriminator, batch_size, x, x_fake, y=None, is_training=True):
  """Returns the WGAN gradient penalty.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.
    x: samples from the true distribution, shape [bs, h, w, channels].
    x_fake: samples from the fake distribution, shape [bs, h, w, channels].
    y: Encoded class embedding for the samples. None for unsupervised models.
    is_training: boolean, are we in train or eval model.

  Returns:
    A tensor with the computed penalty.
  """
  with tf.name_scope("wgangp_penalty"):
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], name="alpha") #x.shape[0].value
    #alpha = tf.random.stateless_uniform(shape=[x.shape[0].value, 1, 1, 1], name="alpha")
    interpolates = x + alpha * (x_fake - x)
    logits = discriminator(interpolates)
    gradients = tf.gradients(logits, [interpolates])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty

def l2_penalty(discriminator):
  """Returns the L2 penalty for each matrix/vector excluding biases.

  Assumes a specific tensor naming followed throughout the compare_gan library.
  We penalize all fully connected, conv2d, and deconv2d layers.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.

  Returns:
     A tensor with the computed penalty.
  """
  with tf.name_scope("l2_penalty"):
    d_weights = [v for v in discriminator.trainable_variables
                 if v.name.endswith("/kernel:0")]
    return tf.reduce_mean(
        [tf.nn.l2_loss(i) for i in d_weights], name="l2_penalty")

def dragan_penalty(discriminator, x, y, is_training):
  """Returns the DRAGAN gradient penalty.

  Args:
    discriminator: Instance of `AbstractDiscriminator`.
    x: Samples from the true distribution, shape [bs, h, w, channels].
    y: Encoded class embedding for the samples. None for unsupervised models.
    is_training: boolean, are we in train or eval model.

  Returns:
    A tensor with the computed penalty.
  """
  with tf.name_scope("dragan_penalty"):
    _, var = tf.nn.moments(x, axes=list(range(len(x.get_shape()))))
    std = tf.sqrt(var)
    x_noisy = x + std * (ops.random_uniform(x.shape) - 0.5)
    x_noisy = tf.clip_by_value(x_noisy, 0.0, 1.0)
    logits = discriminator(x_noisy, y=y, is_training=is_training, reuse=True)[1]
    gradients = tf.gradients(logits, [x_noisy])[0]
    slopes = tf.sqrt(0.0001 + tf.reduce_sum(
        tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))
    return gradient_penalty