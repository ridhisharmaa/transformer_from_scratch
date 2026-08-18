import numpy as np

from embeddings import *
from positional_embedding import *

from encoder import *
from encoder_stack import *

from decoder import *
from decoder_stack import *

from linear import *
from output_softmax import *

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

  return (
    embedding,
    encoder_weights,
    decoder_weights,
    linear_weights
)

def transformer(encoder_id, decoder_id, embedding_matrix, encoder_weights, decoder_weights, linear_weights, num_layers):

  encoder_embedding=embedding_lookup(encoder_id, embedding_matrix)

  encoder_position = positional_encoding(
    len(encoder_id),
    encoder_embedding.shape[1]
)

  encoder_input = (
    encoder_embedding
    +
    encoder_position
)

  encoder_output = encoder_stack(
    encoder_input,
    num_layers,
    *encoder_weights #argument unpacking from tuple to individual numbers
)

  decoder_embedding = embedding_lookup(
    decoder_id,
    embedding_matrix
)

  decoder_position=positional_encoding(
    len(decoder_id),
    decoder_embedding.shape[1]
  )

  decoder_input = (
    decoder_embedding
    +
    decoder_position
)

  decoder_output = decoder_stack(
    decoder_input,
    encoder_output,
    num_layers,
    *decoder_weights
)

  logits=linear(decoder_output, *linear_weights)

  probabilities=output_softmax(logits)

  return probabilities




  