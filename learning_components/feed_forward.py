import numpy as np
from initializers import *

# w1 -> embed dim x hidden dim
# b1 -> 1 x hidden dim
# w2 -> hidden dim x embed dim
# b2 -> 1 x embed dim

def initialize_ffn(embed_dim, hidden_dim):

  w1=xavier(embed_dim,hidden_dim)
  b1=np.zeros(hidden_dim)
  w2=xavier(hidden_dim,embed_dim)
  b2=np.zeros(embed_dim)

  return w1,b1,w2,b2

def relU(X):

  return np.maximum(0,X)

def feed_ffd(X,w1,b1,w2,b2):

  hidden=X@w1 +b1

  hidden=relU(hidden)

  output=hidden @ w2 + b2

  return output
