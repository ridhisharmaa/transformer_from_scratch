import numpy as np

from embeddings import *
from positional_embedding import *

from encoder import *
from encoder_stack import *

from decoder import *
from decoder_stack import *

from linear import *

def initialize_transformer(
    vocab_size,
    embed_dim,
    hidden_dim,
    num_heads
):
  embedding = embedding_matrix(
    vocab_size,
    embed_dim
)

  encoder_weights = initialize_encoder(
    embed_dim,
    hidden_dim,
    num_heads
)

  decoder_weights = initialize_decoder(
    embed_dim,
    hidden_dim,
    num_heads
)

  linear_weights = initialize_linear(
    embed_dim,
    vocab_size
)

def transformer(encoder_id, decoder_id, embedding_matrix, encoder_weights, decoder_weights, linear_weights):

  embedding_lookup(encoder_id, embedding_matrix)

  


  return
  