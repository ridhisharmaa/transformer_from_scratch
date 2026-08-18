"""
The full transformer, tied together.

 init_params        (from initializers.py)  -> every weight
 transformer_forward                        -> run all forwards, collect caches
 cross_entropy_loss                         -> one scalar loss
 transformer_backward                       -> chain all backwards in reverse

Forward helpers live in forward.py, backward functions in backprop.py.
"""

import numpy as np

from initializers import init_params
from forward import (
  positional_encoding, softmax,
  enc_layer_forward, dec_layer_forward,
)
from backprop import (
  softmax_cross_entropy, linear_backprop,
  encoder_backprop, decoder_backprop,
)


# =====================================================================
# FORWARD PASS  (runs everything, collects every cache)
# =====================================================================

def transformer_forward(enc_ids, dec_ids, params):
  E = params['E']; num_heads = params['num_heads']; head_dim = params['embed'] // num_heads

  encoder_input = E[enc_ids] + positional_encoding(len(enc_ids), params['embed'])
  decoder_input = E[dec_ids] + positional_encoding(len(dec_ids), params['embed'])

  # encoder stack
  x = encoder_input
  encoder_caches = []
  for p in params['enc']:
    x, cache = enc_layer_forward(x, p, num_heads, head_dim)
    encoder_caches.append(cache)
  encoder_output = x

  # decoder stack
  y = decoder_input
  decoder_caches = []
  for p in params['dec']:
    y, cache = dec_layer_forward(y, encoder_output, p, num_heads, head_dim)
    decoder_caches.append(cache)
  decoder_output = y

  logits = decoder_output @ params['W_out'] + params['b_out']
  probabilities = softmax(logits)

  caches = dict(enc_ids=enc_ids, dec_ids=dec_ids,
         encoder_caches=encoder_caches, decoder_caches=decoder_caches,
         decoder_output=decoder_output, encoder_output=encoder_output)
  return probabilities, caches


# =====================================================================
# LOSS  (one scalar = how wrong the model is)
# =====================================================================

def cross_entropy_loss(probabilities, target_ids):
  correct = probabilities[np.arange(len(target_ids)), target_ids]
  correct = np.clip(correct, 1e-12, 1.0)
  return np.mean(-np.log(correct))


# =====================================================================
# BACKWARD PASS  (reverse order; reuses the verified block backprops)
# =====================================================================

def transformer_backward(probabilities, target_ids, caches, params):
  num_heads = params['num_heads']; head_dim = params['embed'] // num_heads
  L = params['num_layers']
  grads = {'enc': [None] * L, 'dec': [None] * L}

  # ----- loss gradient + output linear -----
  dlogits = softmax_cross_entropy(probabilities, target_ids)              # p - y
  d_decoder_output, grads['W_out'], grads['b_out'] = linear_backprop(
    caches['decoder_output'], params['W_out'], dlogits)

  # ----- decoder stack (reverse), accumulate gradient to the encoder output -----
  d_encoder_output_total = np.zeros_like(caches['encoder_output'])
  dy = d_decoder_output
  for i in reversed(range(L)):
    p, c = params['dec'][i], caches['decoder_caches'][i]
    out = decoder_backprop(
      dy, c['inp'], c['encoder_out'],
      p['mWq'], p['mWk'], p['mWv'], p['mWo'], c['masked_concat'], c['masked_caches'],
      p['cWq'], p['cWk'], p['cWv'], p['cWo'], c['cross_concat'], c['cross_caches'],
      head_dim, num_heads,
      p['ln1_g'], c['x_hat1'], c['sigma1'],
      p['ln2_g'], c['x_hat2'], c['sigma2'],
      p['ln3_g'], c['x_hat3'], c['sigma3'],
      p['ff_W1'], c['ffn_input'], p['ff_W2'], c['ffn_hidden'], c['ffn_pre'], c['norm1'])
    (dX, d_encoder_output, dmWq, dmWk, dmWv, dmWo, dcWq, dcWk, dcWv, dcWo,
    dW1, db1, dW2, db2, dg1, dbe1, dg2, dbe2, dg3, dbe3) = out
    dy = dX
    d_encoder_output_total += d_encoder_output    # fan-out: enc output feeds every decoder layer
    grads['dec'][i] = dict(
      mWq=dmWq, mWk=dmWk, mWv=dmWv, mWo=dmWo,
      cWq=dcWq, cWk=dcWk, cWv=dcWv, cWo=dcWo,
      ln1_g=dg1, ln1_b=dbe1, ln2_g=dg2, ln2_b=dbe2, ln3_g=dg3, ln3_b=dbe3,
      ff_W1=dW1, ff_b1=db1, ff_W2=dW2, ff_b2=db2)
  d_decoder_input = dy

  # ----- encoder stack (reverse), starting from the accumulated gradient -----
  dx = d_encoder_output_total
  for i in reversed(range(L)):
    p, c = params['enc'][i], caches['encoder_caches'][i]
    out = encoder_backprop(
      dx, c['inp'],
      p['Wq'], p['Wk'], p['Wv'], p['Wo'], c['concat'], c['head_caches'], head_dim, num_heads,
      p['ln1_g'], c['x_hat1'], c['sigma1'],
      p['ln2_g'], c['x_hat2'], c['sigma2'],
      p['ff_W1'], c['ffn_input'], p['ff_W2'], c['ffn_hidden'], c['ffn_pre'])
    (dX, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2, dg1, dbe1, dg2, dbe2) = out
    dx = dX
    grads['enc'][i] = dict(
      Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo,
      ln1_g=dg1, ln1_b=dbe1, ln2_g=dg2, ln2_b=dbe2,
      ff_W1=dW1, ff_b1=db1, ff_W2=dW2, ff_b2=db2)
  d_encoder_input = dx

  # ----- embeddings (positional encoding is constant, gradient passes to E rows) -----
  dE = np.zeros_like(params['E'])
  for j, token_id in enumerate(caches['enc_ids']):
    dE[token_id] += d_encoder_input[j]
  for j, token_id in enumerate(caches['dec_ids']):
    dE[token_id] += d_decoder_input[j]
  grads['E'] = dE

  return grads
