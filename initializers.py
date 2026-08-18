"""All weight initialization lives here."""

import numpy as np


# ---------------------------------------------------------------------
# Xavier / Glorot init.
# Picks the weight size from how many numbers get summed in a matmul,
# so the signal doesn't explode or vanish as it flows through layers.
#   std = sqrt( 2 / (fan_in + fan_out) )
# ---------------------------------------------------------------------
def xavier(fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std


# ---------------------------------------------------------------------
# One encoder layer's weights (its OWN weights, not shared with other layers).
# ---------------------------------------------------------------------
def init_encoder_layer(embed, hidden, num_heads):
    hd = embed // num_heads
    return {
        'Wq': [xavier(embed, hd) for _ in range(num_heads)],
        'Wk': [xavier(embed, hd) for _ in range(num_heads)],
        'Wv': [xavier(embed, hd) for _ in range(num_heads)],
        'Wo': xavier(embed, embed),
        'ln1_g': np.ones(embed), 'ln1_b': np.zeros(embed),
        'ln2_g': np.ones(embed), 'ln2_b': np.zeros(embed),
        'ff_W1': xavier(embed, hidden), 'ff_b1': np.zeros(hidden),
        'ff_W2': xavier(hidden, embed), 'ff_b2': np.zeros(embed),
    }


# ---------------------------------------------------------------------
# One decoder layer's weights (masked self-attn + cross-attn + ffn + 3 LNs).
# ---------------------------------------------------------------------
def init_decoder_layer(embed, hidden, num_heads):
    hd = embed // num_heads
    return {
        'mWq': [xavier(embed, hd) for _ in range(num_heads)],
        'mWk': [xavier(embed, hd) for _ in range(num_heads)],
        'mWv': [xavier(embed, hd) for _ in range(num_heads)],
        'mWo': xavier(embed, embed),
        'cWq': [xavier(embed, hd) for _ in range(num_heads)],
        'cWk': [xavier(embed, hd) for _ in range(num_heads)],
        'cWv': [xavier(embed, hd) for _ in range(num_heads)],
        'cWo': xavier(embed, embed),
        'ln1_g': np.ones(embed), 'ln1_b': np.zeros(embed),
        'ln2_g': np.ones(embed), 'ln2_b': np.zeros(embed),
        'ln3_g': np.ones(embed), 'ln3_b': np.zeros(embed),
        'ff_W1': xavier(embed, hidden), 'ff_b1': np.zeros(hidden),
        'ff_W2': xavier(hidden, embed), 'ff_b2': np.zeros(embed),
    }


# ---------------------------------------------------------------------
# The whole model's parameters in one dict.
# ---------------------------------------------------------------------
def init_params(vocab, embed, hidden, num_heads, num_layers):
    return {
        'E': np.random.randn(vocab, embed) * 0.02,          # token embeddings
        'enc': [init_encoder_layer(embed, hidden, num_heads) for _ in range(num_layers)],
        'dec': [init_decoder_layer(embed, hidden, num_heads) for _ in range(num_layers)],
        'W_out': xavier(embed, vocab), 'b_out': np.zeros(vocab),
        'embed': embed, 'num_heads': num_heads, 'num_layers': num_layers, 'vocab': vocab,
    }
