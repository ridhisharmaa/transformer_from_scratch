import numpy as np
from self_attention import *

def initialize_multihead_weights(embed_dim, num_heads):

  head_dim = embed_dim // num_heads

  Wq = []
  Wk = []
  Wv = []

  for _ in range(num_heads):

    Wq.append(np.random.rand(embed_dim, head_dim))
    Wk.append(np.random.rand(embed_dim, head_dim))
    Wv.append(np.random.rand(embed_dim, head_dim))

  Wo = np.random.rand(embed_dim, embed_dim)

  return Wq, Wk, Wv, Wo

def multihead_attention(X,Wq,Wk,Wv,Wo):
  outputs = []
  head_dim = Wq[0].shape[1]
  num_heads = len(Wq)


  for head in range(num_heads): 

    Q, K, V = compute_qkv(X, Wq[head], Wk[head], Wv[head])

    score = compute_score(Q, K)

    score = scale_score(score, head_dim)

    score = softmax(score)

    output = attention_score(score, V)
    
    outputs.append(output)


  result=np.concatenate(outputs, axis=1)

  return result @ Wo

  