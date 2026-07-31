import numpy as np

def initialize_weights(embed_dim):

    Wq = np.random.rand(embed_dim, embed_dim)
    Wk = np.random.rand(embed_dim, embed_dim)
    Wv = np.random.rand(embed_dim, embed_dim)

    return Wq, Wk, Wv

def compute_qkv(X, Wq, Wk, Wv):

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    return Q, K, V


def compute_score(Q,K):

  score = Q @ K.T

  return score


def scale_score(score,dim):
  score = score / np.sqrt(dim)

  return score
   