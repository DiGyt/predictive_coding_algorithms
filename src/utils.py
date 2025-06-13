import os
import os.path as op

import numpy as np

def construct_cond_id(exp_paradigm, training_algorithm, localized, recurrence, control_gain, enforce_sparsity):
    """Constructs a condition ID string based on experiment parameters."""
    return "-".join(["exp_" + exp_paradigm.replace(" ", "_"),
                      "algo_" + training_algorithm.replace(" ", "_"),
                      "loc_" + str(localized),
                      "rec_" + str(recurrence),
                      "gain_" + str(control_gain),
                      "sparse_" + str(enforce_sparsity),
                      ])


def save_results(model, exp_paradigm, training_algorithm, localized, recurrence, control_gain, enforce_sparsity,
                 save_dir="/home/guetlid95/projects/pc_algos_study_1/results/replication_results"):
    """save all important results in a structured way."""
    # create an ID string for this condition
    cond_id = construct_cond_id(exp_paradigm, training_algorithm, localized, recurrence, control_gain, enforce_sparsity)
    
    # create directories for all results
    exp_dir = op.join(save_dir, cond_id)
    model_dir = op.join(exp_dir, "model")
    
    # create dirs if not existing
    for cur_dir in [save_dir, exp_dir]:
        if not op.exists(cur_dir):
          os.mkdir(cur_dir)
    
    # save all relevant objects
    model.save(model_dir)


def load_as_list(result_dir):
  """Load results as a list"""
  return [np.load(op.join(result_dir, fname), allow_pickle=True) for fname in sorted(os.listdir(result_dir)) if fname != ".ipynb_checkpoints"]
