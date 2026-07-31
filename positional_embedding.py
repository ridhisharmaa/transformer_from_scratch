import numpy as np

def pos_encod(seq, embed_dim):

  position=np.zeros((seq, embed_dim))

  for pos in range(seq):

    for i in range(embed_dim):

      if(pos%2==0):
        position[pos][i]=np.sin(pos/(10000**(i/embed_dim)))

      else:
        position[pos][i]=np.cos(pos/(10000**((i-1)/embed_dim)))

  return position

