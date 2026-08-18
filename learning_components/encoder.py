from multihead_attention import *
from residual import *
from layer_norm import *
from feed_forward import *


def initialize_encoder(embed_dim, hidden_dim, num_heads):

    # Multi-Head Attention
  Wq, Wk, Wv, Wo = initialize_multihead_weights(
        embed_dim,
        num_heads
    )

    # Feed Forward
  W1, b1, W2, b2 = initialize_ffn(
        embed_dim,
        hidden_dim
    )

    # LayerNorms
  gamma1, beta1 = initialize_layer_norm(embed_dim)
  gamma2, beta2 = initialize_layer_norm(embed_dim)

  return (
        Wq, Wk, Wv, Wo,
        W1, b1, W2, b2,
        gamma1, beta1,
        gamma2, beta2
    )

def encoder_block(
    X,
    Wq, Wk, Wv, Wo,
    W1, b1, W2, b2,
    gamma1, beta1,
    gamma2, beta2
):

  attention = multihead_attention(
        X,
        Wq,
        Wk,
        Wv,
        Wo
    )

  skip1 = skip_connection(X, attention)

  norm1 = layer_norm(skip1, gamma1, beta1)

  ffd = feed_ffd(
        norm1,
        W1,
        b1,
        W2,
        b2
    )

  skip2 = skip_connection(norm1, ffd)

  norm2 = layer_norm(skip2, gamma2, beta2)

  return norm2

