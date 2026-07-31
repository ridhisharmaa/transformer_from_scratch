import numpy as np

def embedding_matrix(vocab, dim):

  embedding=np.random.rand(len(vocab),dim)

  return embedding


def embedding_lookup(ids, embedding):

  matrix=[]

  for x in ids:
    matrix.append(embedding[x])

  return np.array(matrix)