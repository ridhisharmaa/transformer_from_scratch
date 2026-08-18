import numpy as np
from initializers import *


def initialize_linear(embed_dim, vocab_size):

  W = xavier(embed_dim, vocab_size)

  b = np.zeros(vocab_size)

  return W, b


def linear(X, W, b):

  output = X @ W + b

  return output