# 🤖 Transformer From Scratch (NumPy)

> The original **Encoder–Decoder Transformer** from **"Attention Is All You Need"**, implemented from scratch in **pure NumPy** — including a fully hand-derived, **gradient-checked backward pass**. No PyTorch, no TensorFlow, no autograd.

---

## 📖 Overview

This project implements the complete Transformer architecture — **forward pass and backward pass** — using only NumPy. Every gradient is derived by hand and verified numerically.

The goal is not just to make it work, but to understand **every mathematical operation** behind Transformers: attention, the softmax Jacobian, LayerNorm's coupled gradients, backprop through multi-head and cross-attention, and how gradients flow across an encoder–decoder stack.

---

## ✨ Highlights

- **Full backward pass, from scratch** — cross-entropy, softmax, linear, feed-forward, LayerNorm, residual, multi-head attention, masked self-attention, and cross-attention, all wired through encoder and decoder blocks.
- **Every gradient is gradient-checked** against numerical (central-difference) gradients — the whole model agrees to **~1e-10**.
- **Trains and learns** — loss decreases with SGD on a toy corpus.
- **Clean, modular code** — forward, backward, initialization, and the optimizer each live in their own file.

---

## 🛠 Tech Stack

- Python + NumPy only
- Object-oriented & modular design
- Manual backpropagation + numerical gradient checking

---

## 📂 Project Structure

```text
transformer-from-scratch/
│
├── initializers.py     # Xavier init + all model parameters (per-layer weights)
├── forward.py          # every forward function (each returns its cache)
├── backprop.py         # every backward function (hand-derived gradients)
├── model.py            # transformer_forward → cross_entropy_loss → transformer_backward
├── optimizer.py        # SGD parameter update
│
├── tokenizer.py        # vocab, encode/decode
├── predict.py          # argmax decoding
│
├── data/corpus.txt     # tiny training corpus
│
└── learning_components/ # original component-by-component build (kept as learning notes)
```

The `learning_components/` folder holds the original, one-file-per-component implementation used while building and understanding each piece. The top-level files are the consolidated, trainable model.

---

## 🏗️ Architecture

```text
        INPUT SENTENCE                    TARGET (shifted)
              │                                 │
              ▼                                 ▼
        Tokenization                      Tokenization
              │                                 │
        Word Embeddings + Positional Encoding (both sides)
              │                                 │
              ▼                                 ▼
        Encoder Stack (N) ──────────►  Decoder Stack (N)
                              (cross-attention)
                                                │
                                                ▼
                                          Linear + Softmax
                                                │
                                                ▼
                                       Next-Token Probabilities
                                                │
                                                ▼
                                          Cross-Entropy Loss
                                                │
                                                ▼
                        Backward Pass ◄─── (gradients flow all the way back,
                                            including decoder → encoder)
```

---

## 🚀 Training

```python
from initializers import init_params
from model import transformer_forward, cross_entropy_loss, transformer_backward
from optimizer import sgd_step

params = init_params(vocab=15, embed=16, hidden=32, num_heads=2, num_layers=2)

for step in range(200):
    probs, caches = transformer_forward(enc_ids, dec_ids, params)   # forward
    loss  = cross_entropy_loss(probs, target_ids)                    # how wrong
    grads = transformer_backward(probs, target_ids, caches, params)  # backward
    sgd_step(params, grads, lr=0.1)                                  # learn
```

The loss falls as the model trains — e.g. from ~3.5 (random) toward ~1.0 on a small example.

---

## ✅ Progress

| Module | Forward | Backward |
|---|:---:|:---:|
| Tokenizer / Embeddings / Positional Encoding | ✅ | — |
| Scaled Dot-Product Attention | ✅ | ✅ |
| Multi-Head Attention | ✅ | ✅ |
| Masked Self-Attention | ✅ | ✅ |
| Cross-Attention | ✅ | ✅ |
| Residual + LayerNorm | ✅ | ✅ |
| Feed-Forward Network | ✅ | ✅ |
| Encoder Block / Stack | ✅ | ✅ |
| Decoder Block / Stack | ✅ | ✅ |
| Linear + Softmax + Cross-Entropy | ✅ | ✅ |
| Full model (forward + loss + backward) | ✅ | ✅ |
| SGD optimizer + training loop | ✅ | — |
| Gradient checking (whole model ~1e-10) | ✅ | ✅ |
| Text generation / greedy decode | 🚧 | — |

---

## 📚 What I Learned

- Scaled dot-product and multi-head attention, and the softmax Jacobian
- Why softmax + cross-entropy collapses to `p − y`
- Backprop through LayerNorm (mean/variance coupling → three-term gradient)
- Residual connections as a gradient highway (fan-out rule)
- How cross-attention carries gradient from the decoder back into the encoder
- Numerical gradient checking to verify every derivative
- Wiring a full model: caching activations forward, chaining backward in reverse

---

## 🔮 Next

- Greedy / beam-search text generation
- Adam optimizer
- Mini-batch training on a larger dataset
- Weight saving & loading
- Attention visualization

---

## 📖 References

1. **Attention Is All You Need** — Vaswani et al. (2017)
2. The Annotated Transformer
3. NumPy documentation

---

## ⭐ Purpose

An educational implementation of the original Transformer, built entirely from scratch in NumPy — with the goal of understanding the internal mathematics of both the forward and the backward pass, not production performance.
