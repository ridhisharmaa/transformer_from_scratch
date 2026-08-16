import numpy as np

# Xavier / Glorot initialization.
# Picks the weight size based on how many numbers get summed in a matmul,
# so the signal doesn't explode or vanish as it flows through layers.
#   std = sqrt( 2 / (fan_in + fan_out) )
#   fan_in  = number of inputs  (rows of W)
#   fan_out = number of outputs (cols of W)

def xavier(fan_in, fan_out):

  std = np.sqrt(2.0 / (fan_in + fan_out))

  return np.random.randn(fan_in, fan_out) * std
