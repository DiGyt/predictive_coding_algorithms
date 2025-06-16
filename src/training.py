import copy

import tensorflow as tf
import keras.backend as K
import tensorflow_probability as tfp


####################
# Training Layers
####################

class FrozenSimpleRNN(tf.keras.layers.SimpleRNN):
    def build(self, input_shape):
        super(FrozenSimpleRNN, self).build(input_shape)

        # Set the recurrent kernel as untrainable during initialization
        self.cell.recurrent_kernel = self.add_weight(shape=(self.units, self.units),
                                                    name="recurrent_kernel",
                                                    initializer="zeros",  # eliminate all recurrent connections.
                                                    regularizer=self.recurrent_regularizer,
                                                    constraint=self.recurrent_constraint,
                                                    trainable=False)  # Freezes the recurrent kernel.
        self.built = True

####################
# Training tools
####################

def leaky_relu(x):
    """Leaky ReLU with fixed leakage of 0.5."""
    return tf.nn.leaky_relu(x, 0.5)
    #return tf.where(x >= 0, x, x * 0.05)

def inv_leaky_relu(x):
    """Inverse leaky ReLU with fixed leakage of 0.5."""
    return tf.where(x >= 0, x, x / 0.5)
    #return tf.where(x >= 0, x, x / 0.05)

def expoly(x, exp=1.00001):
    """A veeeeery slightly bent almost linear function."""
    return x * exp**x

def inv_expoly(x, exp=1.00001):
    """The inverse of that function."""
    return tfp.math.lambertw(x * tf.math.log(exp)) / tf.math.log(exp)


def get_inv_activation(activation):
    """Returns the inverse of an activation if an appropriate inverse exists."""
    if activation == tf.keras.activations.linear:
        inv_activation = tf.keras.activations.get("linear")
    elif activation == tf.keras.activations.relu:
        inv_activation = tf.keras.activations.get("linear") # this will heuristically map all values that have been lower than 0 before to 0.
    elif activation == tf.keras.activations.softplus:
        # see tfp.math.softplus_inverse
        inv_activation = lambda x:tf.math.log(tf.math.exp(x) - 1.)
    elif activation == tf.keras.activations.get("tanh"):
        inv_activation = tf.math.atanh
    elif activation == tf.keras.activations.get("sigmoid"):
        inv_activation = lambda x: -tf.math.log(1./x-1.)
    elif activation in (leaky_relu, tf.nn.leaky_relu):  # assumes an alpha of 0.5
        inv_activation = inv_leaky_relu
    elif activation == expoly:  # assumes an exp of 1 + 1e-5#
        inv_activation = inv_expoly
    else:
        raise ValueError("Activation Function not invertible")
    return inv_activation


def layer_predict_backtrack(h_t, h_t_prev, layer):
    """Recreate priors and predictions for a single layer using the output and the layer weights. Includes the functionality of backtrack_prior."""

    if isinstance(layer, tf.keras.layers.SimpleRNN):

        # get activation
        inv_activation = get_inv_activation(layer.cell.activation)
        
        # extract the weights for this layer
        input_kernel, recurrent_kernel, bias = layer.cell.kernel, layer.cell.recurrent_kernel, layer.cell.bias  # layer.weights      
        
        # backtrack function
        #drive = inv_activation(tf.maximum(h_t, 1e-12))  # the tf.maximum is clipping the input to prevent inf values
        drive = inv_activation(h_t)
        drive = tf.clip_by_value(drive, -1e3, 1e3)
        unbiased_drive = drive - bias[None, :]
        unbiased_drive = tf.clip_by_value(unbiased_drive, -1e3, 1e3)
        input_drive = unbiased_drive - h_t_prev @ recurrent_kernel
        input_drive = tf.clip_by_value(input_drive, -1e3, 1e3)
        inv_kernel = tf.linalg.pinv(input_kernel)  # omit multiple pinv calculations
        input_prev = input_drive @ inv_kernel
        input_prev = tf.where(tf.math.is_nan(input_prev), tf.zeros_like(input_prev), input_prev)
        input_prev = tf.clip_by_value(input_prev, -1e3, 1e3)
        
    elif isinstance(layer, tf.keras.layers.Reshape):
        # the prev input is just shaping back the output to the input
        input_prev = tf.reshape(h_t, (-1,) + layer.input_shape[1:])
        #tf.debugging.check_numerics(input_prev, "12. input_prev")

    #elif isinstance(layer, Cast) or isinstance(layer, Norm):
    elif isinstance(layer, tf.keras.layers.Activation):
        input_prev = get_inv_activation(layer.activation)(h_t)
    else:
        input_prev = h_t
    #else:
        # raise ValueError("No Reverse Operation defined for this layer type")
    return input_prev


def model_reconstruct_prediction(model, state):
  """Reconstructs the input data from a model and it's output data."""
  states = []
  for layer in model.layers[::-1]:
    state = layer_predict_backtrack(state, tf.concat([tf.zeros_like(state)[:, :1], state[:, :-1]], axis=1), layer)
    state = tf.where(tf.math.is_nan(state), tf.zeros_like(state), state)
    states.append(state)
    #state = tf.nn.batch_normalization(state, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-10)
  return states[::-1]


def model_reconstruct_prediction_nonrec(model, state):
  """Reconstructs the input data from a model and its output data."""
  states = []
  for layer in model.layers[::-1]:
    state = layer_predict_backtrack(state, tf.zeros_like(state), layer)
    state = tf.where(tf.math.is_nan(state), tf.zeros_like(state), state)
    states.append(state)
    #state = tf.nn.batch_normalization(state, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-10)
  return states[::-1]


def model_predict_backtrack(model, input_data, to_input_shape=False):
  """Computes intermediate activations and reconstructs layer-wise priors by backtracking through an SimpleRNN model.

  Args:
    model (tf.keras.Model): A Keras model, typically containing RNN layers.
    input_data (tf.Tensor): Input tensor to the model.
    to_input_shape (bool): If True, projects each layer's prior back to the input space using the layer's transformation.

  Returns:
    Tuple[List[tf.Tensor], List[tf.Tensor]]: A list of intermediate activations and a list of reconstructed priors,
    ordered from input to output layer.
  """

  # layer activations
  intermediate_model = tf.keras.models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers])
  activations = intermediate_model(input_data)

  # store predictions here
  priors = []

  # iterate backwards through the RNN
  for layer, output in zip(model.layers[::-1], activations[::-1]):

    # backtrack the priors if wanted (as the output of the previous layer)
    if to_input_shape:

      priors = [layer_predict_backtrack(prior, output, layer) for prior in priors] # we project the prior of the current layer back through the input

    if isinstance(layer, tf.keras.layers.SimpleRNN):

      # extract the weights for this layer
      input_kernel, recurrent_kernel, bias = layer.cell.kernel, layer.cell.recurrent_kernel, layer.cell.bias  # layer.weights

      # calculate priors
      prior_drive = tf.concat([tf.zeros_like(output)[:, :1], output[:, :-1]], axis=1) @ recurrent_kernel
      prior = prior_drive @ tf.linalg.pinv(input_kernel)

    elif isinstance(layer, tf.keras.layers.Reshape):
      # the prev input is just shaping back the output to the input
      prior = tf.reshape(prior, (-1,) + layer.input_shape[1:])

    #elif isinstance(layer, Cast) or isinstance(layer, Norm):
    else:
      pass  # prior is just previous prior. No changes.

    # add the prior to the list
    priors.insert(0, prior)


  return activations, priors



####################
# Training functions
####################
@tf.function
def model_fit_layers(model, train_data, validation_data=None, progressive=True, **kwargs):
    """Progressive=False uses the original y as target. Progressive=True uses the previous layer output as input"""

    # get optimizer config
    #optimizer_config = model.optimizer.get_config()
    if isinstance(model.loss, FFLossWithThreshold):
      progressive=False
    else:
      progressive=True

    # loop through all layers
    for i, layer in enumerate(model.layers):

      for x, y in train_data.take(1):
        layer.build(x.shape)


      if layer.trainable and isinstance(layer, (tf.keras.layers.SimpleRNN, FrozenSimpleRNN)):

        # compile current layer
        if len(layer.trainable_variables) > 0:

          layer_model = tf.keras.models.Sequential([layer])
          # for some losses we need to adapt the function to match the layer instead of the entire model
          if isinstance(model.loss, PredPriorLoss):
            layer_loss = PredPriorLoss(layer_model)
          elif isinstance(model.loss, FFLossWithThreshold):
            layer_loss = FFLossWithThreshold(10., average=False)
          else:
            layer_loss = model.loss
          layer_model.compile(optimizer=copy.copy(model.optimizer),
                              loss=layer_loss,
                              metrics=model.metrics)

          # fit the layer
          layer_model.fit(train_data, validation_data=validation_data, **kwargs)

      # functions to map the input through the layer
      @tf.function
      def map_progressive(x, y):
          """encodes input AND TARGET through the according layer."""
          x_transformed = layer(x)
          return x_transformed, x_transformed
    
      @tf.function
      def map_non_progressive(x, y):
          """encodes ONLY INPUT through the according layer."""
          x_transformed = layer(x)
          return x_transformed, y
          
      if progressive:
        train_data = train_data.map(map_progressive)  # the target is also encoded in the same way as the input data
      else:
        train_data = train_data.map(map_non_progressive)  # the target data remains the same, only the input data is encoded

      if validation_data is not None:
        if progressive:
          validation_data = validation_data.map(map_progressive)
        else:
          validation_data = validation_data.map(map_non_progressive)


def model_fit(model, train_data, validation_data=None, progressive=False, **kwargs):
  """Do a standard model fit."""
  model.fit(train_data, validation_data=validation_data, **kwargs)


####################
# Loss functions
####################
class SupervisedProjection(tf.keras.losses.Loss):
    """
    A custom loss that fits a linear regression from the predicted latent kernel
    to the true outputs, then projects using that linear map and computes MSE.
    Intended for use with models like SimpleRNN where the latent state should
    be predictive of both input reconstruction and class separation.
    """

    def __init__(self, size_y_pred=None, size_x_pred=None, name="supervised_gaussian_mixture", **kwargs):
        super(SupervisedProjection, self).__init__(name=name, **kwargs)
        self.size_y_pred = int(size_y_pred) if size_y_pred is not None else None
        self.size_x_pred = int(size_x_pred) if size_x_pred is not None else None

    def call(self, y_true, y_pred):
        """
        Project y_pred to best linear fit to y_true and compute MSE between
        the projection and y_true.

        Args:
            y_true: Tensor of shape [batch_size, ..., output_dim]
            y_pred: Tensor of shape [batch_size, ..., latent_dim]

        Returns:
            Tensor scalar: Mean squared error between projected y_pred and y_true
        """
        # Flatten batch and time dims for linear regression: [batch*time, feature]
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

        # Solve for linear regression weights W: y_true ≈ y_pred @ W
        # W = (X^T X)^-1 X^T y
        X = y_pred_flat  # Shape: [N, D]
        Y = y_true_flat  # Shape: [N, O]

        XT = tf.transpose(X)  # [D, N]
        XTX = XT @ X           # [D, D]
        XTY = XT @ Y           # [D, O]

        # Add small identity term to XTX for numerical stability (ridge regularization)
        ridge = 1e-6 * tf.eye(tf.shape(XTX)[0])
        W = tf.linalg.solve(XTX + ridge, XTY)  # [D, O]

        # Project the prediction to the best-fit linear approximation
        Y_proj = X @ W  # [N, O]

        # Compute mean squared error between projection and true labels
        return tf.reduce_mean(tf.square(Y - Y_proj))

    def get_config(self):
        config = super(SupervisedProjection, self).get_config()
        config.update({'size_x_pred': self.size_x_pred, 'size_y_pred': self.size_y_pred})
        return config



class PredPriorLoss(tf.keras.losses.Loss):
    """Optimizes the latent kernel to predict the input. Only works for SimpleRNN"""
    def __init__(self, model, **kwargs):
        super(PredPriorLoss, self).__init__(**kwargs)
        self.model = model

    def call(self, y_true, y_pred):
        # flatten the target images
        shape = tf.shape(y_true)

        ## last block
        y_pred = y_pred @ self.model.layers[-1].cell.recurrent_kernel
        y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
        prediction_at_input = model_reconstruct_prediction(self.model, y_pred)[0]
        prediction_at_input = tf.where(tf.math.is_nan(prediction_at_input), tf.zeros_like(prediction_at_input), prediction_at_input)
        pred_loss = tf.keras.losses.mean_squared_error(-y_true[:, 1:], prediction_at_input[:, :-1])

        # Implement your loss computation using layer_weights and layer_bias
        return pred_loss

    def get_config(self):
        # Include the necessary information to recreate the object
        base_config = super(PredPriorLoss, self).get_config()
        config = {'model': self.model}
        return dict(list(base_config.items()) + list(config.items()))



class FFLossWithThreshold():
  """Forward Forward-based contrastive loss according to Hinton (2022)."""

  def __init__(self, threshold, average=False, **kwargs):
    """This is a wrapper function (imagine it like a loss object), that returns the actual loss function with a certain predefined threshold."""
    #super(FFLossWithThreshold, self).__init__(**kwargs)
    self.threshold = threshold
    self.average = average

  def __call__(self, y_true, y_pred):
    """This returns the actual FF loss following the standard Keras loss structure.
    Note that y_true here means the probability of a sample being negative,
    while y_pred is the output returned by the model."""

    # cast y_true and y_pred into the correct data types
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)

    # square the model output (as described in FF), then sum them across all output neurons
    g = K.pow(y_pred, 2.)
    g = K.sum(g, axis=-1)

    # subtract the threshold
    g = g - self.threshold

    # now we convert the sums of activations into a 0-1 space to model the probability of a sample being negative
    p_negative = tf.nn.sigmoid(g)

    # and calculate the Binary Crossentropy between prediction of negativity, based on the layer's firing intensity
    # and the actual probability for the sample being negative
    # This is slightly deviating from the description in the FF paper, but it's basically the same.
    loss = tf.keras.losses.BinaryCrossentropy()(y_true, p_negative)

    # if necessary, we average the loss
    if self.average:
      loss = tf.reduce_mean(loss)
    return loss

  def get_config(self):
      # Include the necessary information to recreate the object
      #base_config = super(FFLossWithThreshold, self).get_config()
      config = {'threshold': self.threshold, 'average': self.average}
      #return dict(list(base_config.items()) + list(config.items()))
      return config
