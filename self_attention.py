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

  

  softmax_scores=[]

  for row in score:

    row_sum=0
     

    for value in row:
      row_sum = row_sum+np.exp(y)

    row_scores=[]

    for value in row:
      row_scores.append(np.exp(value)/row_sum)

    softmax_scores.append(row_scores)

  return np.array(softmax_scores)

    
def attention_score(score,V):

  attention=score @ V

  return attention

  
   