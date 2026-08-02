from decoder import *


def decoder_stack(

    X,
    encoder_output,

    num_layers,

    Wq_mask, Wk_mask, Wv_mask, Wo_mask,

    Wq_cross, Wk_cross, Wv_cross, Wo_cross,

    W1, b1, W2, b2,

    gamma1, beta1,

    gamma2, beta2,

    gamma3, beta3

):

  for _ in range(num_layers):

    X = decoder(

        X,
        encoder_output,
 
        Wq_mask,
        Wk_mask,
        Wv_mask,
        Wo_mask,

        Wq_cross,
        Wk_cross,
        Wv_cross,
        Wo_cross,

        W1,
        b1,
        W2,
        b2,

        gamma1,
        beta1,

        gamma2,
        beta2,

        gamma3,
        beta3

    )

  return X