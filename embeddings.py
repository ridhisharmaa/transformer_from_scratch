import numpy as np

def embedding_matrix(vocab_size, dim):

  embedding_matrix=np.random.rand(vocab_size,dim)

  return embedding_matrix


def embedding_lookup(ids, embedding):

  embedded_tokens=[]

  for token_id in ids:
    embedded_tokens.append(embedding[token_id])

  return np.array(embedded_tokens)