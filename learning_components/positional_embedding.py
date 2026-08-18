import numpy as np

def positional_encoding(seq_len, embed_dim):

  position = np.zeros((seq_len, embed_dim))

  for pos in range(seq_len):

    for i in range(embed_dim):

      if i % 2 == 0:

        position[pos, i] = np.sin(
            pos / (10000 ** (i / embed_dim))
        )

      else:

        position[pos, i] = np.cos(
            pos / (10000 ** ((i - 1) / embed_dim))
        )

  return position