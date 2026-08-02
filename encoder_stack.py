from encoder import *


def encoder_stack(
    X,
    num_layers,

    Wq, Wk, Wv, Wo,

    W1, b1, W2, b2,

    gamma1, beta1,
    gamma2, beta2
):

  for _ in range(num_layers):

    X = encoder_block(
        X,

        Wq, Wk, Wv, Wo,

        W1, b1, W2, b2,

        gamma1, beta1,

        gamma2, beta2
    )

  return X