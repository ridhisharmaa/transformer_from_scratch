

import numpy as np

def layer_norm(X):

  normalized = []

  for row in X:

    mean = np.mean(row)

    std = np.std(row)

    norm_row = (row - mean) / (std + 1e-8)

    normalized.append(norm_row)

  return np.array(normalized)