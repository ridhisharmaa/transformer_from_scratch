import numpy as np

from self_attention import *

def cross_attention(
    encoder_output,
    decoder_output,
    Wq,
    Wk,
    Wv,
    Wo
):
  outputs = []

  head_dim = Wq[0].shape[1]

  for head in range(len(Wq)):

    Q = decoder_output @ Wq[head]

    K = encoder_output @ Wk[head]

    V = encoder_output @ Wv[head]

    score = compute_score(Q,K)

    score = scale_score(score, head_dim)

    score = softmax(score)

    output = attention_score(score, V)

    outputs.append(output)

  result = np.concatenate(
    outputs,
    axis=1
  )

  return result @ Wo