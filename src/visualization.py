import warnings
import logging

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

PLT_STYLE = "seaborn-v0_8-poster" # "xkcd"

def plot_input_series(x, condition="", sample_idx=0, show_indices=[0, 2, 4, 6]):
    """Plot a series of inputs in a row. Only `show_indices` indices of the series are shown."""
    
    # ignore wanrnings & set random seed
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore')
    
    # set style for the function
    style = plt.style.context(PLT_STYLE)
    with style:
    
        # plot all the images in a row
        fig, axes = plt.subplots(1, len(show_indices), figsize=(15, 4.1), sharey=True)
        
        for idx, item in enumerate(show_indices):
            axes[idx].imshow(x[sample_idx][item], cmap="gray")
            axes[idx].get_xaxis().set_visible(False)
            axes[idx].set_title(f"Step {item}")
        
        # set title
        plt.suptitle(condition)
        plt.show()


def plot_series(x, x_pred, skip_layers=[]):
    """Plots a sequence of true and predicted images across time steps and layers, skipping specified prediction layers."""
    if isinstance(x_pred, list):
        x_list = [x] + x_pred
    else:
        x_list = [x, x_pred]
    
    # remove the layers we don't want to see
    x_list = [x for idx, x in enumerate(x_list) if (idx - 1) not in skip_layers]  # remove the layers from x_pred that are in the list
    
    # Create the subplots
    ncols = len(x) - (len(x_list) - 1)
    fig, axes = plt.subplots(nrows=len(x_list), ncols=ncols, figsize=(ncols*2, len(x_list) * 2), sharex=True, sharey=True)
    
    # Iterate over the top row
    for row_i, row in enumerate(axes):
        for i, ax in enumerate(row):
            offset = 1 if row_i == 0 else 0
            # Get the corresponding image
            image = x_list[row_i][i + offset]
            
            # Display the image using imshow
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([], [])
            ax.set_yticks([], [])

            # adapt labels
            if row_i== 0 and i == 0:
                ax.set_ylabel('True image at t+1')
            else:
                if i == 0:
                  ax.set_ylabel(f'Layer {row_i - 1} priors at t+0')

            # set step index at the bottom row
            if row_i == len(axes) - 1: 
                ax.set_xlabel(f'Step {i}')
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Show the figure
    plt.show()


def plot_erp(x, ci=None, ax=plt):
  """Plots an ERP. input shape must be (n_samples, n_steps). Ci should be the alpha level"""

  erp = np.mean(x, axis=0)
  ax.plot(erp) # - np.mean(results["Image Match"].mean()))
  if ci != None:
    critical_z = stats.norm.ppf(1 - ci / 2.)
    ax.fill_between(np.arange(x.shape[1]), erp - critical_z * stats.sem(x, axis=0), erp + critical_z * stats.sem(x, axis=0), alpha=0.3)