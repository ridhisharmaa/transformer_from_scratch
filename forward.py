"""
All forward-pass functions, written in the same beginner-friendly style
as your original component files.

Each layer returns its output AND a `cache` — the values its matching
backward function (in backprop.py) will need.
"""

import numpy as np


# =====================================================================
# Small attention helpers  (same as your self_attention.py)
# =====================================================================

def compute_qkv(X, Wq, Wk, Wv):
  Q = X @ Wq
  K = X @ Wk
  V = X @ Wv
  return Q, K, V


def compute_score(Q, K):
  return Q @ K.T


def scale_score(score, head_dim):
  return score / np.sqrt(head_dim)


def softmax(score):
  # subtract the row max for numerical stability, then normalise each row
  score = score - np.max(score, axis=1, keepdims=True)
  exp_score = np.exp(score)
  return exp_score / np.sum(exp_score, axis=1, keepdims=True)


def positional_encoding(seq_len, embed_dim):
  position = np.zeros((seq_len, embed_dim))
  for pos in range(seq_len):
    for i in range(embed_dim):
      if i % 2 == 0:
        position[pos, i] = np.sin(pos / (10000 ** (i / embed_dim)))
      else:
        position[pos, i] = np.cos(pos / (10000 ** ((i - 1) / embed_dim)))
  return position


# =====================================================================
# LayerNorm
# =====================================================================

def ln_forward(X, gamma, beta, eps=1e-8):
  mean = X.mean(axis=1, keepdims=True)
  std = np.sqrt(X.var(axis=1, keepdims=True) + eps)
  normalized = (X - mean) / std
  output = gamma * normalized + beta
  cache = (normalized, std)                 # (x_hat, sigma) for layernorm_backprop
  return output, cache


# =====================================================================
# Feed-forward network
# =====================================================================

def ffn_forward(X, W1, b1, W2, b2):
  pre_activation = X @ W1 + b1              # before ReLU
  hidden = np.maximum(0, pre_activation)    # after ReLU
  output = hidden @ W2 + b2
  cache = (pre_activation, hidden)          # (a, h2) for ffn_backprop
  return output, cache


# =====================================================================
# Multi-head self-attention  (mask=True gives the decoder's causal mask)
# =====================================================================

def mha_forward(X, Wq, Wk, Wv, Wo, num_heads, head_dim, mask=False):
  seq_len = X.shape[0]
  causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1) if mask else 0.0

  outputs = []
  head_caches = []
  for head in range(num_heads):
    Q, K, V = compute_qkv(X, Wq[head], Wk[head], Wv[head])
    score = compute_score(Q, K)
    score = scale_score(score, head_dim)
    attention_weights = softmax(score + causal_mask)
    outputs.append(attention_weights @ V)
    head_caches.append((Q, K, V, attention_weights))

  concat = np.concatenate(outputs, axis=1)
  output = concat @ Wo
  return output, concat, head_caches


# =====================================================================
# Cross-attention  (Q from decoder, K and V from encoder output)
# =====================================================================

def cross_forward(decoder_in, encoder_out, Wq, Wk, Wv, Wo, num_heads, head_dim):
  outputs = []
  head_caches = []
  for head in range(num_heads):
    Q = decoder_in @ Wq[head]
    K = encoder_out @ Wk[head]
    V = encoder_out @ Wv[head]
    score = compute_score(Q, K)
    score = scale_score(score, head_dim)
    attention_weights = softmax(score)
    outputs.append(attention_weights @ V)
    head_caches.append((Q, K, V, attention_weights))

  concat = np.concatenate(outputs, axis=1)
  output = concat @ Wo
  return output, concat, head_caches


# =====================================================================
# One encoder block  (self-attention -> add & norm -> ffn -> add & norm)
# =====================================================================

def enc_layer_forward(X, p, num_heads, head_dim):
  attention, concat, head_caches = mha_forward(
    X, p['Wq'], p['Wk'], p['Wv'], p['Wo'], num_heads, head_dim)
  skip1 = X + attention
  norm1, (x_hat1, sigma1) = ln_forward(skip1, p['ln1_g'], p['ln1_b'])

  ffn_out, (ffn_pre, ffn_hidden) = ffn_forward(
    norm1, p['ff_W1'], p['ff_b1'], p['ff_W2'], p['ff_b2'])
  skip2 = norm1 + ffn_out
  norm2, (x_hat2, sigma2) = ln_forward(skip2, p['ln2_g'], p['ln2_b'])

  cache = dict(
    inp=X, concat=concat, head_caches=head_caches,
    x_hat1=x_hat1, sigma1=sigma1, x_hat2=x_hat2, sigma2=sigma2,
    ffn_input=norm1, ffn_pre=ffn_pre, ffn_hidden=ffn_hidden)
  return norm2, cache


# =====================================================================
# One decoder block  (masked self-attn -> cross-attn -> ffn)
# =====================================================================

def dec_layer_forward(X, encoder_out, p, num_heads, head_dim):
  masked_out, masked_concat, masked_caches = mha_forward(
    X, p['mWq'], p['mWk'], p['mWv'], p['mWo'], num_heads, head_dim, mask=True)
  skip1 = X + masked_out
  norm1, (x_hat1, sigma1) = ln_forward(skip1, p['ln1_g'], p['ln1_b'])

  cross_out, cross_concat, cross_caches = cross_forward(
    norm1, encoder_out, p['cWq'], p['cWk'], p['cWv'], p['cWo'], num_heads, head_dim)
  skip2 = norm1 + cross_out
  norm2, (x_hat2, sigma2) = ln_forward(skip2, p['ln2_g'], p['ln2_b'])

  ffn_out, (ffn_pre, ffn_hidden) = ffn_forward(
    norm2, p['ff_W1'], p['ff_b1'], p['ff_W2'], p['ff_b2'])
  skip3 = norm2 + ffn_out
  norm3, (x_hat3, sigma3) = ln_forward(skip3, p['ln3_g'], p['ln3_b'])

  cache = dict(
    inp=X, encoder_out=encoder_out,
    masked_concat=masked_concat, masked_caches=masked_caches,
    cross_concat=cross_concat, cross_caches=cross_caches,
    x_hat1=x_hat1, sigma1=sigma1, x_hat2=x_hat2, sigma2=sigma2, x_hat3=x_hat3, sigma3=sigma3,
    ffn_input=norm2, ffn_pre=ffn_pre, ffn_hidden=ffn_hidden, norm1=norm1)
  return norm3, cache
