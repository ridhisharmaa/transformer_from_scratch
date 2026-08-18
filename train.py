import numpy as np

from tokenizer import load, tokenize, build_word_to_id, build_id_to_word, encode
from initializers import init_params
from model import transformer_forward, cross_entropy_loss, transformer_backward
from optimizer import sgd_step
from generate import generate


# ==========================
# 1. Load + tokenize data
# ==========================

corpus = load("data/corpus.txt")

vocab = tokenize(corpus)
word_to_id = build_word_to_id(vocab)
id_to_word = build_id_to_word(word_to_id)
vocab_size = len(vocab)


# ==========================
# 2. Teacher-forcing examples  (input, decoder input, target)
# ==========================

examples = []
for sentence in corpus:
  ids = encode(sentence, word_to_id)
  if len(ids) < 2:
    continue
  enc_ids    = ids[:-1]     # input sentence
  dec_ids    = ids[:-1]     # decoder input
  target_ids = ids[1:]      # shifted by one -> "predict the next word"
  examples.append((enc_ids, dec_ids, target_ids))


# ==========================
# 3. Hyperparameters
# ==========================

embed_dim  = 32
hidden_dim = 64
num_heads  = 4
num_layers = 2
lr         = 0.05
epochs     = 300


# ==========================
# 4. Initialize the model
# ==========================

np.random.seed(0)
params = init_params(vocab_size, embed_dim, hidden_dim, num_heads, num_layers)


# ==========================
# 5. Training loop
# ==========================

for epoch in range(epochs):
  total_loss = 0.0
  for enc_ids, dec_ids, target_ids in examples:
    probs, caches = transformer_forward(enc_ids, dec_ids, params)   # forward
    total_loss   += cross_entropy_loss(probs, target_ids)           # how wrong
    grads         = transformer_backward(probs, target_ids, caches, params)  # backward
    sgd_step(params, grads, lr)                                     # learn

  if epoch % 40 == 0 or epoch == epochs - 1:
    avg_loss = total_loss / len(examples)
    print(f"epoch {epoch:3d}   avg loss = {avg_loss:.4f}")


# ==========================
# 6. Generate text from the trained model
# ==========================

print("\n--- generated text ---")
for seed in [["machine"], ["i", "love"], ["neural", "networks"], ["transformers", "are"], ["deep"]]:
  print(f"  {' '.join(seed):18s} ->  {generate(seed, params, word_to_id, id_to_word)}")
