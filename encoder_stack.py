import numpy as np
from encoder import *

def encoder_stack(X, embed_dim, hidden_dim, num_heads, num_layers):

  for i in range(num_layers):

    X=encoder_block(
      X,
      embed_dim,
      hidden_dim,
      num_heads
    )

  return X