import tensorflow as tf
import numpy as np
from scipy import stats

from .training import get_inv_activation, FrozenSimpleRNN


def get_input_drive(layer, outputs):
  """gets the input drive of a layer"""

  inv_activation = get_inv_activation(layer.cell.activation)

  # extract the weights for this layer
  input_kernel, recurrent_kernel, bias = layer.cell.kernel, layer.cell.recurrent_kernel, layer.cell.bias  # layer.weights


  # backtrack function
  drive = inv_activation(outputs)
  unbiased_drive = drive - bias[None, :]
  input_drive = unbiased_drive - tf.concat([outputs[:, :-1], tf.zeros_like(outputs[:, :1])], axis=1) @ recurrent_kernel
  return input_drive


def fisher_r_to_z(r):
    """Do Fisher Z transformation"""
    #return 0.5 * np.log((1 + r) / (1 - r))
    return np.arctanh(r)


def fisher_z_confidence_interval(correlation, n, alpha=0.05):
    """Uses Fisher's z to calculate ULCI and LLCI"""
    fisher_z = fisher_r_to_z(correlation)
    se = 1 / np.sqrt(n - 3)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = fisher_z - z_critical * se
    ci_upper = fisher_z + z_critical * se
    return np.tanh(ci_lower), np.tanh(ci_upper)


def compare_correlations(r1, n1, r2, n2):
    """Compare two r values, using fisher z transformation and the sample Size. """
    z1 = fisher_r_to_z(r1)
    z2 = fisher_r_to_z(r2)
    se_diff = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z = (z1 - z2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value


def get_erp(model, x, select_layer=None):
  """Accumulates all the activations of a network"""
  act = []
  out = x
  for layer in model.layers:
    if isinstance(layer, (tf.keras.layers.SimpleRNN, FrozenSimpleRNN)):
      outs = layer(out)
      act.append(tf.reduce_mean(tf.math.pow(outs, 2), axis=(2)))
    out = layer(out)
  if select_layer == None:
    erp = np.stack([a for a in act], axis=-1).mean(axis=-1)
  else:
    erp = act[select_layer]
  return erp


def erp_significance(erp1, erp2, n_comparisons=1):
  """Calculate when different time steps deviate significantly for 2 erps."""

  # Perform paired-sample t-test
  t_values, p_values = stats.ttest_rel(erp1, erp2, axis=0)

  # Perform Bonferroni correction
  p_values_corrected = p_values * n_comparisons

  return t_values, p_values, p_values_corrected


def accuracy_standard_error(accuracy, n):
    """Calculate the SE from an accuracy"""
    return np.sqrt((accuracy * (1 - accuracy)) / n)


def accuracy_t_test(accuracy_A, accuracy_B, sample_size_A, sample_size_B):
    # Calculate standard errors
    SE_A = accuracy_standard_error(accuracy_A, sample_size_A)
    SE_B = accuracy_standard_error(accuracy_B, sample_size_B)

    # Calculate pooled standard error (assuming equal variances)
    pooled_SE = np.sqrt(SE_A**2 + SE_B**2)

    # Calculate t-statistic
    t_statistic = (accuracy_A - accuracy_B) / pooled_SE

    # Degrees of freedom
    df = sample_size_A + sample_size_B - 2  # degrees of freedom for a two-sample t-test

    # Calculate p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))  # two-tailed test

    return t_statistic, p_value


def accuracy_confidence_interval(accuracy, n, alpha=0.05):
    """Uses Prediction SE to calculate ULCI and LLCI"""
    se = accuracy_standard_error(accuracy, n)
    z_critical = stats.norm.ppf(1 - alpha / 2)
    ci_lower = accuracy - z_critical * se
    ci_upper = accuracy + z_critical * se
    return ci_lower, ci_upper