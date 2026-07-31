# from tokenizer import *
# from embeddings import *

# corpus=load()
# vocab=tokenize(corpus)
# print(corpus)
# print(vocab)
# word_to_id=build_word_to_id(vocab)

# id_to_word = build_id_to_word(word_to_id)

# print(word_to_id)
# print(id_to_word)

# print(encode("I Love AI",word_to_id))

# print(decode([0,1,5,8],id_to_word))

# embedding=embedding_matrix(vocab, 4)

# print(embedding_lookup([0,1,3],embedding))

# from positional_embedding import pos_encod

# print(pos_encod(3, 4))

# for sentence in corpus:

#   ids = encode(sentence, word_to_id)

#   embeddings = embedding_lookup(ids, embedding)

#   position = pos_encod(len(ids), 4)

#   x = embeddings + position

#   print(x)

#     # pass x into the Transformer


from tokenizer import *
from embeddings import *
from positional_embedding import *
from self_attention import *

# ==========================
# Configuration
# ==========================

EMBED_DIM = 4

# ==========================
# Load Data
# ==========================

corpus = load()

# ==========================
# Build Vocabulary
# ==========================

vocab = tokenize(corpus)

word_to_id = build_word_to_id(vocab)
id_to_word = build_id_to_word(word_to_id)

# ==========================
# Create Embedding Matrix
# ==========================

embedding = embedding_matrix(vocab, EMBED_DIM)

# ==========================
# Display Tokenizer Results
# ==========================

print("Corpus:")
print(corpus)

print("\nVocabulary:")
print(vocab)

print("\nWord → ID")
print(word_to_id)

print("\nID → Word")
print(id_to_word)

print("\nEncoding Example:")
print(encode("I Love AI", word_to_id))

print("\nDecoding Example:")
print(decode([0, 1, 5, 8], id_to_word))

print("\nEmbedding Lookup Example:")
print(embedding_lookup([0, 1, 3], embedding))

print("\nPositional Encoding Example:")
print(pos_encod(3, EMBED_DIM))

# ==========================
# Build Transformer Input
# ==========================

print("\n==============================")
print("Transformer Inputs")
print("==============================")

for sentence in corpus:

    print("\nSentence:")
    print(sentence)

    ids = encode(sentence, word_to_id)
    print("Token IDs:", ids)

    x_embed = embedding_lookup(ids, embedding)

    x_pos = pos_encod(len(ids), EMBED_DIM)

    x = x_embed + x_pos

    print("Input to Transformer (X):")
    print(x)


Wq, Wk, Wv = initialize_weights(4)

Q, K, V = compute_qkv(x, Wq, Wk, Wv)

print("Q")
print(Q)

print("K")
print(K)

print("V")
print(V)


score=compute_score(Q,K)



score=scale_score(score,4)

print("Attention scores:")
print(score)

