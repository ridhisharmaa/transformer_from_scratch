import numpy as np

from residual import *
from layer_norm import *
from feed_forward import *
from cross_attention import *
from masked_multihead_attention import *
from multihead_attention import *


def initialize_decoder(embed_dim, hidden_dim, num_heads):

 
  Wq_mask, Wk_mask, Wv_mask, Wo_mask = initialize_multihead_weights(
      embed_dim,
      num_heads
  )

  
  Wq_cross, Wk_cross, Wv_cross, Wo_cross = initialize_multihead_weights(
      embed_dim,
      num_heads
  )

  
  W1, b1, W2, b2 = initialize_ffn(embed_dim, hidden_dim)

  
  gamma1, beta1 = initialize_layer_norm(embed_dim)
  gamma2, beta2 = initialize_layer_norm(embed_dim)
  gamma3, beta3 = initialize_layer_norm(embed_dim)

  return (
      Wq_mask, Wk_mask, Wv_mask, Wo_mask,
      Wq_cross, Wk_cross, Wv_cross, Wo_cross,
      W1, b1, W2, b2,
      gamma1, beta1,
      gamma2, beta2,
      gamma3, beta3
  )


def decoder(
    X,
    encoder_output,

    Wq_mask, Wk_mask, Wv_mask, Wo_mask,

    Wq_cross, Wk_cross, Wv_cross, Wo_cross,

    W1, b1, W2, b2,

    gamma1, beta1,
    gamma2, beta2,
    gamma3, beta3
):

  
  mask_output = masked_multihead_attention(
      X,
      Wq_mask,
      Wk_mask,
      Wv_mask,
      Wo_mask
  )

  skip1 = skip_connection(X, mask_output)
  norm1 = layer_norm(skip1, gamma1, beta1)

  
  cross_output = cross_attention(
      encoder_output,
      norm1,
      Wq_cross,
      Wk_cross,
      Wv_cross,
      Wo_cross
  )

  skip2 = skip_connection(norm1, cross_output)
  norm2 = layer_norm(skip2, gamma2, beta2)

  
  ffd = feed_ffd(
      norm2,
      W1,
      b1,
      W2,
      b2
  )

  skip3 = skip_connection(norm2, ffd)
  norm3 = layer_norm(skip3, gamma3, beta3)

  return norm3