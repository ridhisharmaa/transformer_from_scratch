import numpy as np


def initialize_linear(embed_dim, vocab_size):

  W = np.random.rand(embed_dim, vocab_size)

  b = np.random.rand(vocab_size)

  return W, b


def linear(X, W, b):

  output = X @ W + b

  return output