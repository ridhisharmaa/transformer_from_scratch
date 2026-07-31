from multihead_attention import *
from skip_connection import *
from layer_norm import *
from feed_forward import *

def encoder_block(X, embed_dim, hidden_dim, num_heads):

  Wq,Wk,Wv,Wo = initialize_multihead_weights(embed_dim,num_heads)

  attention = multihead_att(
      X,
      Wq,
      Wk,
      Wv,
      Wo
  )

  skip1 = skip_connection(X, attention)

  norm1 = layer_norm(skip1)

  W1,b1,W2,b2 = initialize_ffn(embed_dim, hidden_dim)

  ffd=feed_ffd(norm1,W1,b1,W2,b2)

  skip2 = (norm1, ffd)
  norm2= layer_norm(skip2)

  return norm2

