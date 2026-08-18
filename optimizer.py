"""
SGD optimizer.

The whole idea is one line:   weight = weight - lr * gradient

Everything else here is just walking the nested `params` structure
(dicts, lists of layers, and per-head lists of weights) so we apply
that one rule to every weight array.
"""

import numpy as np


def sgd_step(params, grads, lr):
  # ---- top-level arrays (embeddings + output linear) ----
  params['E']     -= lr * grads['E']
  params['W_out'] -= lr * grads['W_out']
  params['b_out'] -= lr * grads['b_out']

  # ---- every encoder and decoder layer ----
  for i in range(params['num_layers']):
    update_layer(params['enc'][i], grads['enc'][i], lr)
    update_layer(params['dec'][i], grads['dec'][i], lr)


def update_layer(layer_params, layer_grads, lr):
  # layer_grads only holds weight keys, so we can loop over it directly
  for key in layer_grads:
    p = layer_params[key]
    g = layer_grads[key]

    if isinstance(p, list):
      # Wq / Wk / Wv are lists (one array per head) -> update each head
      for h in range(len(p)):
        p[h] -= lr * g[h]
    else:
      # everything else is a single array
      layer_params[key] -= lr * g




