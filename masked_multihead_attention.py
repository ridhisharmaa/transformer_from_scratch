import numpy as np

from self_attention import *


def create_mask(seq_len):

  mask = np.triu(
      np.ones((seq_len, seq_len)),
      k=1
  )

  return mask * (-1e9)


def masked_multihead_attention(X, Wq, Wk, Wv, Wo):

  outputs = []

  head_dim = Wq[0].shape[1]

  num_heads = len(Wq)
  
  mask = create_mask(num_heads)

  for head in range(num_heads):

    Q, K, V = compute_qkv(X, Wq[head], Wk[head], Wv[head])
    
    score = compute_score(Q, K)

    score = scale_score(score, head_dim)

    score = score + mask      

    score = softmax(score)

    output = attention_score(score, V)

    outputs.append(output)

  result = np.concatenate(outputs, axis=1)

  return result @ Wo