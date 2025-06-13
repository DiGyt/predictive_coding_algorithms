import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


DEFAULT_MEANS = tf.convert_to_tensor([
    [0., 1., 1.],
    [0., -1., 2.],
    [0., 2., 3.],
    [0., -2., 4.],
    [0., 3., 5.],
    [0., -3., -1.],
    [0., 4., -2.],
    [0., -4., -3.],
    [0., 5., -4.],
    [0., -5., -5.],
], dtype=tf.float32) * 0.2


#####################
# Dataset loaders
#####################

def get_MNIST():
    """Loads the MNIST dataset"""
    train_dataset = tfds.load("mnist", split='train[:90%]', as_supervised=True)
    val_dataset = tfds.load("mnist", split='train[90%:]', as_supervised=True)
    return train_dataset, val_dataset

def get_BAIR():
    """Loads the BAIR dataset into a useable shape"""
    train_dataset = tfds.load("bair_robot_pushing_small", split="train")
    val_dataset = tfds.load("bair_robot_pushing_small", split="test")
    # basic preproc
    train_dataset = train_dataset.map(bair_preproc_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(bair_preproc_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset, val_dataset


#####################
# Full preproc pipelines
#####################


def roving_oddball_preproc(x, y, train_split, y_return):
  """Preprocesses data according to a roving oddball paradigm."""
  x, y = img_preproc(x, y)
  x, y = roving_preproc(x, y, n_steps=16, y_return=y_return)
  #x, y = augment_preproc(x, y, shuffle_split=0., flip_once=False, augment=True, y_return="y", p_noise=0., cumsum=False)
  x = augment_image(x, p_rot = 0.5, p_shift = 0.5, p_noise = 0., cumsum=False, max_rotation = 0.4, max_shift = 3)
  x, y = add_noise(x, y)
  return x, y


def omission_preproc(x, y, train_split, y_return):
  """Preprocesses data according to a random omission paradigm."""
  x, y = img_preproc(x, y)
  x, y = random_omissions(x, y, omissions=0.3)
  x, y = roving_preproc(x, y, n_steps=16, y_return=y_return)
  x, y = add_noise(x, y)
  return x, y


def statistical_learning_preproc(x, y, train_split, y_return):
  """Preprocesses data according to a statistical learning paradigm."""
  x, y = img_preproc(x, y)
  x, y = boring_movie(x, y)
  x, y = sim_rule_preproc(x, y, means=DEFAULT_MEANS, stddevs=tf.zeros_like(DEFAULT_MEANS), y_return=y_return, shuffle_split=train_split)
  x, y = add_noise(x, y)
  return x, y


def BAIR_learning_preproc(x, y, train_split, y_return):
  """Preprocesses BAIR video data."""
  x, y = augment_preproc(x, y, shuffle_split=train_split, flip_once=False, augment=False, y_return=y_return)
  #x, y = add_noise(x, y)
  return x, y


#####################
# Dataset loaders
#####################


def img_preproc(x, y, dtype=tf.float32):
  """Cast input image to a certain tf dtype and normalize them between 0 and 1."""
  return tf.cast(x, dtype) / 255., y

def img_color_to_grayscale(x, y, keepdims=True, luminance=False):
    """Converts a color image to greyscale by averaging the color channels."""
    if luminance:
        weights = tf.constant([0.299, 0.587, 0.114], dtype=x.dtype)
        x = tf.tensordot(x, weights, axes=[-1, 0])
        x = tf.expand_dims(x, axis=-1) if keepdims else x
    else:
        x = tf.math.reduce_mean(x, axis=-1, keepdims=keepdims)
    return x, y

def bair_preproc_pipeline(x, seq_len=16, img_size=(28, 28)):
    """Apply basic video preprocessing to the bair dataset to bring it into useable format."""
    # select fields to read from
    y = x["action"]
    x = x['image_aux1']
    
    # randomly start an index and take the next seq_len steps
    start_idx = tf.random.uniform(shape=(), minval=0, maxval=len(x) - seq_len + 1, dtype=tf.int32)
    x = x[start_idx : start_idx + seq_len]
    y = y[start_idx : start_idx + seq_len]
    
    # do other preprocessing steps
    x, _ = resize_video(x, None, size=img_size, has_batch_dim=False)
    x, _ = img_preproc(x, None)
    x, _ = img_color_to_grayscale(x, None, keepdims=True, luminance=True)
    x = tf.ensure_shape(x, (seq_len, img_size[0], img_size[1], 1))
    return x, y

# create the data mapping
def resize_video(tensor, y, size=(28, 28), method=tf.image.ResizeMethod.BILINEAR, has_batch_dim=True):
    """reshape image before or after batching"""
    shape = tf.shape(tensor)
    if has_batch_dim:
        # Reshape to merge two batch dimensions, resize, and reshape back
        return tf.reshape(tf.image.resize(tf.reshape(tensor, [-1, shape[2], shape[3], shape[4]]), size, method=method),[shape[0], shape[1], size[0], size[1], shape[4]]), y
    else:
        return tf.image.resize(tensor, size, method=method), y


def overlay_preproc(x, y, shuffle=True, y_return="p_neg"):
  """
  This function implements the overlay shown in the Forward-Forward paper.
  It will overlay an MNIST image with a pixel on the upper left, indicating the
  number that this supposed to be shown in the image.

  Parameters
  ----------
  x, y:
    Represent the standard image, target inputs from the dataset.
  shuffle: bool
    If True, half of the images will be randomly shuffled into incoherent pairs
    of pixels and numbers. These can be used as negative data.
  y_return: str in ("p_neg", "x", "y")
    This parameter decides what should be returned as the y output from the
    function. "p_neg" will return the probability of each sample being negative
    (i.e. 0 for positive data and 1 for negative data). "y" will just return the
    original supervised targets, which can be used for standard supervised
    training. "x" will just double the x input and return it as x as well as y.
    This can help for an autoregressive setting where we want to predict future
    values of x as a target.

  Returns
  -------
  x:
    The x input data overlayed with according number pixels (and shuffled).
  y:
    Original x, y, or p_neg, depending on y_return.

  """

  # randomly shuffle half of the batch
  if shuffle:
    # get the index for splitting the batch
    split = len(y) // 2

    # the positive split is the first half of the target data
    positive_split = y[:split]

    # the negative split is the latter part, but randomly shuffled.
    # This will assign False target numbers to the images
    negative_split = tf.reshape(tf.random.shuffle(tf.reshape(y[split:], [-1])), tf.shape(y[split:]))  # the reshape trick helps us shuffling stacked array for RNN

    # concatenate the positive and negative split again
    y = tf.concat([positive_split, negative_split], axis=0)

  # Now we overlay the targets on the input image
  # create a one hot of the target encoding. shape: (batch_size, num_classes)
  one_hot_y = tf.cast(tf.one_hot(y, depth=10), tf.float32)
  # overlay the first column of the image with the one hot encoding: (batch_size, image_height)
  first_col = tf.concat([one_hot_y[..., :, None, None], x[..., 10:, :1, :]], axis=-3)
  # overlay the entire image with the first column of the image
  x = tf.concat([first_col, x[..., :, 1:, :]], axis=-2)

  # decide on the y output to return
  if y_return == "x":
    y = x
  elif y_return == "p_neg":
    if shuffle:
      # y now represents the "probability" of a sample being negative
      y = tf.concat([tf.zeros_like(y[:split]), tf.ones_like(y[split:])], axis=0)
  return x, y

def boring_movie(x, y, n_steps=16):
  """Stacks a Tensor to create a 'time series' of repeating images."""
  x = tf.stack([x for _ in range(n_steps)], axis=1)  # make the "boring" movie of subsequently following images
  y = tf.stack([y for _ in range(n_steps)], axis=1)
  return x, y


def blank_out_images(x, y, blank_out_values=(0)):
    """
    Blanks out images in the tensor x based on values in the target array y.
    Creates omissions by replacing a certain integer in the dataset with all zero values.

    Parameters:
    - x: Input image tensor.
    - y: Target array containing integers corresponding to the content of images.
    - blank_out_values: List of integers to blank out.

    Returns:
    - Tensor with blanked out images.
    """
    # Create a mask for values to be blanked out
    mask = tf.reduce_any(tf.equal(tf.expand_dims(y, axis=-1), blank_out_values), axis=-1)
    #mask = tf.equal(tf.expand_dims(y, axis=-1), blank_out_values)

    # Multiply the mask with x to blank out images
    x_blanked = x * tf.cast(~mask[:, None, None, None], dtype=x.dtype)
    return x_blanked, y


def poststim_mask(x, y, onset=2, random=True):
  """Replaces the first few images of a sequence with a random/zero prestimulus mask."""
  # cutoff the latter part of the stimulus sequence
  stim_x = x[:, :onset, ...]

  # replace the first part with either uniform noise or zeros
  if random:
    poststim_x = tf.random.uniform(tf.shape(x[:, onset:, ...]), 0, 1, dtype=stim_x.dtype)
  else:
    poststim_x = tf.zeros(tf.shape(x[:, onset:, ...]), dtype=stim_x.dtype)

  # concatenate them again
  x = tf.concat([stim_x, poststim_x], axis=1)
  return x, y


def random_omissions(x, y, omissions=0.5):
  """Randomly omits a fraction of the batch."""
  batch_size = len(x)
  split = int(float(batch_size) * omissions)
  split_remainder = batch_size - split

  # create a mask and apply it
  mask = tf.concat([tf.zeros(split), tf.ones(split_remainder)], axis=0)[:, None, None, None]
  x = x * tf.cast(mask, x.dtype)
  return x, y


def prestim_mask(x, y, onset=2, random=True):
  """Replaces the first few images of a sequence with a random/zero prestimulus mask."""
  # cutoff the latter part of the stimulus sequence
  stim_x = x[:, onset:, ...]

  # replace the first part with either uniform noise or zeros
  if random:
    prestim_x = tf.random.uniform(tf.shape(x[:, :onset, ...]), 0, 1, dtype=stim_x.dtype)
  else:
    prestim_x = tf.zeros(tf.shape(x[:, :onset, ...]), dtype=stim_x.dtype)

  # concatenate them again
  x = tf.concat([prestim_x, stim_x], axis=1)
  return x, y


def augment_image(x, p_rot = 0.3, p_shift = 0.3, p_noise = 1., cumsum=True,
                  max_rotation = 0.2, max_shift = 2, noise_std = 0.1):
  """
  This function will randomly augment a (time-distributed) batch of images. We
  do this py applying random rotations, random shifts, as well as adding random
  noise to an image.

  Parameters
  ----------
  p_rot, p_shift, p_noise: float
    The probabilities with which to apply rotations, shifts and noise to any
    given image in the batch. Setting these probabilities to less than one will
    mean that the according augmentation is not applied to every image in the
    batch, but only to some.
  max_rotation, max_shift: float
    The maximum rotation (in radians) and maximum shift (in pixels) to be
    applied to an image at any given time.
  noise_std: float
    The standard deviation of the gaussian noise added to any specific image in
    the batch.
  cumsum: bool
    cumsum=False will augment each image seperately starting from the original
    image. cumsum=True will make sure that each augmentation is added to the
    previous augmentation over the time axis, such that the images are
    cummulatively more augmented over time.

  Returns
  -------
  x:
    The augmented images

  """
#def augment_image(x, p_rot = 0., p_shift = 0., p_noise = 0.5, cumsum=True,
#                  max_rotation = 0.2, max_shift = 2, noise_std = 0.1):
  orig_shape = tf.shape(x)
  x = tf.reshape(x, [-1, orig_shape[2], orig_shape[3], orig_shape[4]])
  tmp_shape = tf.shape(x)


  if cumsum:
    # create random rotations, apply cumsum and reshape the same way as x
    rand_rot = tf.random.uniform(orig_shape[:2], -max_rotation, max_rotation)
    rand_rot = tf.reshape(tf.cumsum(rand_rot, axis=1), [tmp_shape[0]])

    # create random shifts, apply cumsum and reshape the same way as x
    rand_shift = tf.random.uniform((orig_shape[0], orig_shape[1], 2), -max_shift, max_shift)
    rand_shift = tf.reshape(tf.cumsum(rand_shift, axis=1), [tmp_shape[0], 2])

    # create random noise, apply cumsum and reshape the same way as x
    rand_noise = tf.random.normal(shape=orig_shape, mean=0, stddev=noise_std)
    rand_noise = tf.reshape(tf.cumsum(rand_noise, axis=1), tmp_shape)


  else:
    # apply the noise directly to the reshaped array
    rand_rot = tf.random.uniform(tmp_shape[:1], -max_rotation, max_rotation)
    rand_shift = tf.random.uniform((tmp_shape[0], 2), -max_shift, max_shift)
    rand_noise = tf.random.normal(shape=tmp_shape, mean=0, stddev=noise_std)

  # create random numbers to apply augmentations only to images < p_augment
  randoms = tf.random.uniform([3, tmp_shape[0]], 0, 1)
  mask_rot = tf.cast(randoms[0] < p_rot, tf.float32)
  mask_shift = tf.cast(randoms[1] < p_shift, tf.float32)
  mask_noise = tf.cast(randoms[2] < p_noise, tf.float32)  # TODO: FIX this in a way that noise is not deleted for a new frame, merely no new noise is added for that frame in cumsum

  # apply image transformations
  x = tfa.image.rotate(x, rand_rot * mask_rot)
  x = tfa.image.translate(x, rand_shift * mask_shift[:, None])
  x = tf.clip_by_value(x + rand_noise * mask_noise[:, None, None, None], 0, 1)

  # go back to orig shape
  x = tf.reshape(x, orig_shape)
  return x


def augment_preproc(x, y=None, shuffle_split=0.5, flip_once=False, augment=True,
                         y_return="p_neg", **augment_kwargs):
  """This function will preprocess an image batch based on augmentation. The
  batch can be separated into a shuffled and an unshuffled split, allowing the
  creation of positive (augmented, unshuffled) and negative (augmented, shuffled)
  data.

  Parameters
  ----------
  x, y:
    The standard image (x), target (y) inputs from the dataset. Targets are
    not strictly required as we augment images in a self-supervised way.
  shuffle_split: float
    The fraction of images in the batch that will be shuffled. Passing 1. will
    shuffle all the images in the batch, passing 0. will shuffle no images.
    Passing 0.5 will shuffle half the images in the batch.
  augment: bool
    Whether or not to apply `augment_image` to augment the images in the batch.
  flip_once: bool
    If True, the shuffled images will only be randomly switched once in the
    middle of the time sequence. If False, the shuffled images will change for
    every step in the time series.
  y_return: str in ("p_neg", "x", "y")
    This parameter decides what should be returned as the y output from the
    function. "p_neg" will return the probability of each sample being negative
    (i.e. 0 for positive data and 1 for negative data). "y" will just return the
    original supervised targets, which can be used for standard supervised
    training. "x" will just double the x input and return it as x as well as y.
    This can help for an autoregressive setting where we want to predict future
    values of x as a target.
  **augment_kwargs:
    All further keyword arguments will be passed on to the `augment_image`
    function.

  Returns
  -------
  x:
    The augmented (and partially shuffled) image data.
  y:
    x, y, or p_neg, depending on what is requested in y_return.
  """

  # get the sample number at which to separate the shuffled part from the rest
  split = int(float(len(x)) * (1. - shuffle_split))

  # separate the non shuffled part
  positive_split = x[:split]

  # shuffle the negative split either by image or once in the middle
  pos_shape = tf.shape(x[:split])
  neg_shape = tf.shape(x[split:])
  if flip_once:
    shuffle_shape = [neg_shape[0] * 2, neg_shape[1] // 2, neg_shape[2], neg_shape[3], neg_shape[4]]
  else:
    shuffle_shape = [-1, neg_shape[2], neg_shape[3], neg_shape[4]]

  # reshape, shuffle, then reshape back (for computational efficiency)
  negative_split = tf.reshape(tf.random.shuffle(tf.reshape(x[split:], shuffle_shape)), neg_shape)

  # concatenate the normal and the shuffled part again
  x = tf.concat([positive_split, negative_split], axis=0)

  # apply augmentation to ALL of the images
  if augment:
    x = augment_image(x, **augment_kwargs)

  # decide on the y output to return
  if y_return == "x":
    y = x
  elif y_return == "p_neg":
      # y now represents the "probability" of a sample being negative
      y = tf.concat([tf.zeros([pos_shape[0], pos_shape[1]]),
                     tf.ones([neg_shape[0], neg_shape[1]])], axis=0)  # TODO: test whether this way works for 1 and 0 shuffle split

  return x, y


def zscore_probability_with_deviation(z_scores):
    """
    Calculates two-sided tail probabilities from Z-scores using the standard normal distribution.
    
    Args:
        z_scores (tf.Tensor): Tensor of Z-scores.
    
    Returns:
        tf.Tensor: Tensor of two-sided tail probabilities corresponding to each Z-score.
    """
    # Calculate the cumulative distribution function (CDF) of the standard normal distribution
    cdf = 0.5 * (1.0 + tf.math.erf(z_scores / tf.sqrt(2.0)))

    # Calculate the two-sided probability
    two_sided_prob = 2.0 * tf.minimum(cdf, 1.0 - cdf)
    return two_sided_prob


def sim_rule_preproc(x, y, means=DEFAULT_MEANS, stddevs=tf.zeros_like(DEFAULT_MEANS), shuffle_split=0.5, y_return="p_neg"):
    """
    Applies a simulation-based preprocessing pipeline to image sequences, including random rotations and translations
    based on object-specific movement rules, and optionally returns various target labels.
    
    Row 0 of the random movement vector controls rotation, while rows 1 and 2 control x and y translation, respectively.
    
    Args:
        x (tf.Tensor): Input image sequence tensor of shape [batch, time, height, width, channels].
        y (tf.Tensor): Object class labels corresponding to each sequence in the batch.
        means (tf.Tensor): Tensor of shape [num_classes, 3] representing the mean for each movement rule.
        stddevs (tf.Tensor): Tensor of shape [num_classes, 3] representing the stddev for each movement rule.
        shuffle_split (float, optional): Proportion of data to shuffle as negative samples. Defaults to 0.5.
        y_return (str, optional): Type of label to return. Options: 'x', 'p_neg', 'z_prod', or original y. Defaults to 'p_neg'.
    
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Transformed image sequences and corresponding target labels.
    """
    # get the original array shape
    shape = tf.shape(x)
    split = int(float(len(x)) * (1. - shuffle_split))
    
    # get random numbers from the normal distribution
    rands = tf.random.normal(shape=[shape[0], shape[1], 3], mean=0, stddev=1.)
    z_prod = 1. - tf.math.reduce_prod(zscore_probability_with_deviation(rands), axis=-1)  # reflects the probability of a sample being negative (i.e. the degree of being ood)
    
    # shuffle if neccessary
    neg_shape = tf.shape(x[split:])
    positive_split = x[:split]
    negative_split = tf.reshape(tf.random.shuffle(tf.reshape(x[split:], [-1, neg_shape[2], neg_shape[3], neg_shape[4]])), neg_shape)
    x = tf.concat([positive_split, negative_split], axis=0)
    
    # gather the means according to the object in y
    means = tf.gather(means, y)#[:, None, :]
    stddevs = tf.gather(stddevs, y)#[:, None, :]
    
    # scale one-step movement by the according object rules
    rands = rands * stddevs + means
    
    # as its a temporal sequence, we need to take the cumsum, now it becomes a random walk
    rands = tf.math.cumsum(rands, axis=1)
    
    # reshape them for the transform
    rands_flat = tf.reshape(rands, [shape[0] * shape[1], 3])
    x = tf.reshape(x, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
    
    # Randomly rotate images
    x = tfa.image.rotate(x, rands_flat[:, 0])
    
    # Randomly shift images
    #x = tfa.image.translate(x, rands_flat[:, 1:])  # the latter two rows are used for the x/y movement
    x = tfa.image.translate(x, tf.math.sin(rands_flat[:, 1:]) * 6)  # rao version
    
    # shape x back
    x = tf.reshape(x, shape)
    
    # decide on the y output to return
    if y_return == "x":
        y = x
    elif y_return == "p_neg":
        # y now represents the "probability" of a sample being negative
        y = tf.concat([tf.zeros([split, neg_shape[1]]),
                    tf.ones([neg_shape[0], neg_shape[1]])], axis=0)
    elif y_return == "z_prod":
        y = z_prod
    else:
        y = y  # supervised target
    return x, y


def revert_rule_preproc(x, y, means=DEFAULT_MEANS, stddevs=tf.zeros_like(DEFAULT_MEANS), reverse_split=0.5, y_return="z_prod"):
    """
    Applies reversed movement simulation with rotation and translation; row 0 = rotation, rows 1-2 = x/y shift.
    
    Args:
        x (tf.Tensor): Input image sequence tensor of shape [batch, time, height, width, channels].
        y (tf.Tensor): Object class labels corresponding to each sequence.
        means (tf.Tensor): Mean movement vectors per class.
        stddevs (tf.Tensor): Standard deviation vectors per class.
        reverse_split (float): Proportion of samples with reversed movement direction.
        y_return (str): Output label type: 'x', 'p_neg', 'z_prod', or original y.
    
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Transformed image sequences and corresponding labels.
    """
    
    # get the original array shape
    shape = tf.shape(x)
    split = int(float(len(x)) * (1. - reverse_split))
    
    # get random numbers from the normal distribution
    rands = tf.random.normal(shape=[shape[0], shape[1], 3], mean=0, stddev=1.)
    z_prod = 1. - tf.math.reduce_prod(zscore_probability_with_deviation(rands), axis=-1)  # reflects the probability of a sample being negative (i.e. the degree of being ood)
    
    # gather the means according to the object in y
    means = tf.gather(means, y)#[:, None, :]
    stddevs = tf.gather(stddevs, y)#[:, None, :]
    
    # scale one-step movement by the according object rules
    rands = rands * stddevs + means
    
    # reverse_split % of the rands move in the other direction
    rands = tf.concat([rands[:split], -rands[split:]], axis=0)
    
    # as its a temporal sequence, we need to take the cumsum, now it becomes a random walk
    rands = tf.math.cumsum(rands, axis=1)
    
    # reshape them for the transform
    rands_flat = tf.reshape(rands, [shape[0] * shape[1], 3])
    x = tf.reshape(x, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
    
    # Randomly rotate images
    x = tfa.image.rotate(x, rands_flat[:, 0])
    
    # Randomly shift images
    x = tfa.image.translate(x, tf.math.sin(rands_flat[:, 1:]) * 6)  # the latter two rows are used for the x/y movement
    
    # shape x back
    x = tf.reshape(x, shape)
    
    # decide on the y output to return
    if y_return == "x":
        y = x
    elif y_return == "p_neg":
        # y now represents the "probability" of a sample being negative
        y = tf.concat([tf.zeros([split, shape[1]]), tf.ones([len(x) - split, shape[1]]) ], axis=0)
    elif y_return == "z_prod":
        y = z_prod
    else:
        y = y  # supervised target
    return x, y


def add_noise(x, y=None, noise_std=.1, cumsum=False):
  """Add random gaussian noise to images. If cumsum, sum noise over time (2nd) axis."""
  # define random noise
  x_shape =tf.shape(x)
  rand_noise = tf.random.normal(shape=x_shape, mean=0, stddev=noise_std)

  # cummulate if wanted
  if cumsum:
    rand_noise = tf.cumsum(rand_noise, axis=1)

  # add to x and return
  x = tf.clip_by_value(x + rand_noise, 0, 1)
  return x, y


def random_repeat_array(n_samples, n_steps, n_min=6, n_max=9):
    """Generates a random array of indices repeated with variable lengths to match a target total length."""


    # create a sum limit
    sum_limit = n_samples * n_steps
    approx_length = sum_limit // ((n_min + n_max - 1) // 2) + n_samples // n_steps  # the + n_samples // n_steps is just overshoot to make sure we rather have too many samples than too few

    rand = tf.random.uniform(shape=[approx_length], minval=n_min, maxval=n_max, dtype=tf.int32)
    fill_diff = sum_limit - tf.reduce_sum(rand)
    #tf.print(fill_diff)
    if fill_diff > 0:
      rand = tf.concat([rand, [fill_diff]], axis=0)
    #tf.print(tf.reduce_sum(rand), sum_limit)

    # Generate random indices for repetition
    indices = tf.random.uniform(shape=[tf.shape(rand)[0]], minval=0, maxval=n_samples, dtype=tf.int32)

    # Repeat each element according to random indices
    repeated_array = tf.repeat(indices, rand)[:sum_limit]
    #tf.print(tf.reduce_sum(rand), sum_limit)
    return repeated_array



def roving_preproc(x, y=None, n_steps=16, y_return="p_neg", **kwargs):
    """Preprocesses x into sequences of length n_steps via random repetition and shuffling; optionally generates corresponding labels."""
    x_shape = tf.shape(x)
    
    # repeat and shuffle
    indices = random_repeat_array(x_shape[0], n_steps, **kwargs)
    #tf.print(repeats)
    #tf.print(tf.reduce_mean(repeats[:-6]))
    #tf.print(tf.reduce_sum(repeats / len(x)))
    x = tf.gather(x, indices)
    x = tf.reshape(x, tf.concat([[x_shape[0]], [n_steps], x_shape[1:]], axis=0))
    
    if y_return == "x":
        y = x
    elif y_return == "p_neg":
        # create change/no change mapping
        indices_res = tf.reshape(indices, [x_shape[0], n_steps])
        index_repeat = tf.cast(indices_res[:, 1:] != indices_res[:, :-1], dtype=tf.float32)
        y = tf.concat([tf.ones([x_shape[0], 1]), index_repeat], axis=1)
    else:
        #y = tf.stack([y for _ in range(n_steps)], axis=1)  # stack previous y
        y = tf.reshape(tf.gather(y, indices), [len(y), n_steps])
    return x, y
