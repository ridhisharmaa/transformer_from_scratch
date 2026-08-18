

import numpy as np

def initialize_layer_norm(embed_dim):

  gamma = np.ones(embed_dim) #Because initially we don't want to change the normalized output.

  beta = np.zeros(embed_dim) #Because initially we don't want any shift.

  return gamma, beta

def layer_norm(X, gamma, beta):

  normalized_output = []

  for row in X:

    mean = np.mean(row)

    std = np.std(row)

    normalized_row = (row - mean) / (std + 1e-8)

    output= gamma * normalized_row + beta

    normalized_output.append(output)

  return np.array(normalized_output)