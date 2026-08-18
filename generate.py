"""
Greedy text generation (autoregressive decoding).

Give the model a few seed words. It predicts the most likely next word,
appends it, and repeats -- feeding its own output back in each time.
"""

import numpy as np

from model import transformer_forward


def generate(seed_words, params, word_to_id, id_to_word, max_words=5):
  # turn the seed words into token ids
  ids = [word_to_id[w] for w in seed_words]

  for _ in range(max_words):
    # encoder and decoder both see the current sequence
    probs, _ = transformer_forward(ids, ids, params)

    # the most likely next word = argmax of the LAST position's distribution
    next_id = int(np.argmax(probs[-1]))

    # greedy decoding has no "stop" token, so stop if the model repeats itself
    if next_id == ids[-1]:
      break

    ids.append(next_id)

  return " ".join(id_to_word[i] for i in ids)
