import numpy as np
from self_attention import *

def initialize_multihead_weights(embed_dim, num_heads):

  head_dim = embed_dim // num_heads

  Wq = []
  Wk = []
  Wv = []

  for i in range(num_heads):

    Wq.append(np.random.rand(embed_dim, head_dim))
    Wk.append(np.random.rand(embed_dim, head_dim))
    Wv.append(np.random.rand(embed_dim, head_dim))

  Wo = np.random.rand(embed_dim, embed_dim)

  return Wq, Wk, Wv, Wo

def multihead_att(head_dim,X,Wq,Wk,Wv,Wo):
  outputs=[]

  for i in range(len(Wq)): 
    Q = X @ Wq[i]
    K = X @ Wk[i]
    V = X @ Wv[i]

    score=Q@K.T
    score=score/np.sqrt(head_dim)
    score=softmax(score)
    output=attention_score(score,V)
    outputs.append(output)


  result=np.concatenate(outputs,axis=1)

  return result @ Wo

  