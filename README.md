# 🤖 Transformer From Scratch (NumPy)

> Building the original **Encoder-Decoder Transformer** architecture from the paper **"Attention Is All You Need"** using only **NumPy**.

---

# 📖 Project Overview

This project aims to implement the complete **Transformer architecture** from scratch without using deep learning frameworks like PyTorch or TensorFlow.

The goal is **not just to make it work**, but to understand every mathematical operation behind Transformers by implementing each module manually.

---

# 🎯 Objectives

- Build the complete Transformer from scratch
- Understand the mathematics behind attention
- Implement every component manually
- Learn how Encoder and Decoder work internally
- Create an interview-ready educational project

---

# 🛠 Tech Stack

- Python
- NumPy
- Object-Oriented & Modular Design

---

# 📂 Project Structure

```text
Transformer-From-Scratch/
│
├── tokenizer.py
├── embeddings.py
├── positional_embedding.py
│
├── self_attention.py
├── multihead_attention.py
│
├── skip_connection.py
├── layer_norm.py
├── feed_forward.py
│
├── encoder.py
├── encoder_stack.py
│
├── masked_multihead_attention.py     (Upcoming)
├── cross_attention.py                (Upcoming)
├── decoder.py                        (Upcoming)
├── decoder_stack.py                  (Upcoming)
│
├── transformer.py                    (Upcoming)
├── train.py                          (Upcoming)
│
└── README.md
```

---

# 🏗️ Architecture

```text
                 INPUT SENTENCE
                        │
                        ▼
                 Tokenization
                        │
                        ▼
                 Word Embeddings
                        │
                        ▼
             Positional Encoding
                        │
                        ▼
               Encoder Stack (N)
                        │
                        ▼
                Encoder Output
                        │
                        ▼
              Decoder Stack (N)
                        │
                        ▼
                  Linear Layer
                        │
                        ▼
                     Softmax
                        │
                        ▼
               Next Token Prediction
```

---

# 📚 Learning Outcomes

By completing this project, I gained a deep understanding of:

- Tokenization
- Word Embeddings
- Positional Encoding
- Scaled Dot Product Attention
- Multi Head Attention
- Residual Connections
- Layer Normalization
- Feed Forward Networks
- Encoder Architecture
- Decoder Architecture
- Complete Transformer Pipeline
- Training a Transformer from Scratch

---

# 🚀 Current Progress

| Module | Status |
|---------|--------|
| Tokenizer | ✅ |
| Embeddings | ✅ |
| Positional Encoding | ✅ |
| Self Attention | ✅ |
| Multi Head Attention | ✅ |
| Skip Connection | ✅ |
| Layer Normalization | ✅ |
| Feed Forward Network | ✅ |
| Encoder Block | ✅ |
| Encoder Stack | ✅ |
| Decoder | 🚧 |
| Transformer | 🚧 |
| Training | 🚧 |

---

# 🔮 Future Improvements

- Numerically Stable Softmax
- Learnable LayerNorm Parameters (γ, β)
- Batch Processing
- Mini-Batch Training
- Beam Search
- Greedy Decoding
- Weight Saving & Loading
- Attention Visualization
- Training on a Real Dataset

---

# 📖 References

1. **Attention Is All You Need** — Vaswani et al. (2017)
2. The Annotated Transformer
3. NumPy Documentation

---

# ⭐ Purpose

This repository is an educational implementation of the **original Transformer architecture** built entirely from scratch using **NumPy**.

The focus is on understanding the internal mathematics and engineering behind Transformers rather than achieving production-level performance.