import numpy as np


def output_softmax(logits):

  probabilities = []

  for row in logits:

    row = row - np.max(row)

    exp_row = np.exp(row)

    softmax_row = exp_row / np.sum(exp_row)

    probabilities.append(softmax_row)

  return np.array(probabilities)