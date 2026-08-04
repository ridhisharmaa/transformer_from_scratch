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


def scale_score(score,head_dim):
  score = score / np.sqrt(head_dim)

  return score

def softmax(score):

  softmax_scores = []

  for row in score:

    # Numerical stability
    row = row - np.max(row)

    exp_row = np.exp(row)

    row_scores = exp_row / np.sum(exp_row)

    softmax_scores.append(row_scores)

  return np.array(softmax_scores)

    
def attention_score(score,V):

  attention=score @ V

  return attention

  
   