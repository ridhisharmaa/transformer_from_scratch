from tokenizer import *
from transformer import *
from predict import *

# ==========================
# Load Data
# ==========================

corpus = load("data/corpus.txt")

vocab = tokenize(corpus)

word_to_id = build_word_to_id(vocab)

id_to_word = build_id_to_word(word_to_id)

# ==========================
# Example Sentence
# ==========================

text = corpus[0]

ids = encode(text, word_to_id)

encoder_id = ids[:-1]

target_id = ids[1:]

decoder_id = encoder_id

# ==========================
# Hyperparameters
# ==========================

vocab_size = len(vocab)

embed_dim = 512

hidden_dim = 2048

num_heads = 8

num_layers = 6

# ==========================
# Initialize Transformer
# ==========================

embedding_matrix, encoder_weights, decoder_weights, linear_weights = initialize_transformer(
    vocab_size,
    embed_dim,
    hidden_dim,
    num_heads
)

# ==========================
# Forward Pass
# ==========================

probabilities = transformer(
    encoder_id,
    decoder_id,
    embedding_matrix,
    encoder_weights,
    decoder_weights,
    linear_weights,
    num_layers
)

# ==========================
# Output
# ==========================

print("Output Shape :", probabilities.shape)

print(probabilities)

predicted_words=predict(probabilities, id_to_word)

print(predicted_words)