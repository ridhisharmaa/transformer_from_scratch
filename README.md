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
- **Trains and generates** — loss decreases with SGD, and the trained model produces text via greedy decoding (e.g. `"machine"` → `"machine learning is powerful"`).
- **Clean, modular code** — forward, backward, initialization, the optimizer, training, and generation each live in their own file.

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
├── train.py            # entry point: load data → train → print loss → generate
├── generate.py         # greedy autoregressive text generation
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

## 🚀 Run it

```bash
python train.py
```

This loads the corpus, trains the model with SGD, and then generates text from the trained weights:

```text
epoch   0   avg loss = 3.75      # random model
epoch 160   avg loss = 0.19
epoch 299   avg loss = 0.15      # learned

--- generated text ---
  machine            ->  machine learning is powerful
  transformers are   ->  transformers are amazing
  deep               ->  deep learning is powerful
```

The core training step is four lines — the same loop for every example:

```python
probs, caches = transformer_forward(enc_ids, dec_ids, params)   # forward
loss  = cross_entropy_loss(probs, target_ids)                    # how wrong
grads = transformer_backward(probs, target_ids, caches, params)  # backward
sgd_step(params, grads, lr)                                      # learn
```

Because the corpus is tiny and decoding is greedy (no end-of-sentence token), generations are coherent for the first few words and then drift — the expected behaviour at this scale.

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
| Text generation / greedy decode | ✅ | — |

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

- Beam-search / temperature sampling, and an end-of-sentence token
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
