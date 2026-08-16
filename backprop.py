import numpy as np

def softmax_cross_entropy(probabilities, target_id):
  grad=probabilities.copy()

  grad[np.arange(len(target_id)),target_id] -= 1
  #we did p-y

  return grad/len(target_id)

# Why don't you compute the full softmax Jacobian in code? → Because when softmax feeds cross-entropy, the Jacobian cancels analytically to p − y. Computing the full (vocab × vocab) Jacobian per position would be wasteful — the closed form is one subtraction.

# What's the shape of this gradient and why? → Same as the logits, (seq_len, vocab_size) — one gradient per logit, because every logit influenced the loss.


def linear_backprop(h, W, dOut):
  dW=h.T @ dOut
  dh= dOut @ W.T
  db= np.sum(dOut,axis=0)

  return dh, dW, db

#z=output of this layer=the logits=Wh+b

#h=decoder_output coming from forward pass that is the input to this layer

#dOut=softmax_backprop_output that is coming from back..output of previous back prop

#b was copied to every position in the forward pass. So in the backward pass, every position sends a gradient back to b, and b collects all of them (adds them up). One b, many outputs → many gradients coming home → sum.
#Contrast: why isn't dW a sum too?

# Good thing to notice — W is also reused across positions! And in fact dW = Xᵀ @ dOut is secretly a sum — the matrix multiply is summing the contributions across all positions internally. The bias sum is just more visible because b is a plain vector, so we write the sum explicitly with np.sum. Same rule underneath.


def ffn_backprop(dy, W1, h1, W2, h2, a):

  dW2=h2.T @ dy
  dh2= dy @ W2.T
  db2=np.sum(dy,axis=0)

  drelu = (a > 0)

  da=dh2 * drelu

  dW1=h1.T @ da
  dh1= da @ W1.T
  db1=np.sum(da,axis=0)

  return dh1, dW1, db1, dW2, db2

#x is input into the first linear layer
#linear1 => a=h1W1+b1
#then relu => h2=relU(a)
#linear2 => y=h2W2+b2
# y is output into the next layer in ffn that is another linear
#this was forward pass

  
  
def layernorm_backprop(dy, gamma , x_hat, s ):

  dbeta = np.sum(dy, axis=0)

  dgamma = np.sum(dy*x_hat, axis=0)

  dx_hat= dy * gamma

  mean1= np.mean(dx_hat, axis=1, keepdims=True)

  mean2=np.mean(dx_hat * x_hat, axis=1, keepdims=True)

  dx = (dx_hat - mean1 - x_hat * mean2)/s

  return dx, dgamma, dbeta




# gamma and beta are shared across all rows, so every row contributes
# to their gradients → we SUM across rows (axis=0).
# beta is added → dBeta = sum(dy); gamma multiplies x_hat → dGamma = sum(dy * x_hat).

# Remember why axis=1: the normalization used the mean/std of each row's features, and features live along axis=1. And keepdims=True so the result is (seq, 1) and broadcasts back across the D features.

# "Why is LayerNorm backward harder than a linear layer's?" → "Because mean and variance are computed over the whole row, so every output depends on every input in that row — the elements are coupled. That's why dx has three terms: a direct path plus corrections through the mean and the variance."
# "Sanity check for LayerNorm backward?" → "The row-mean of dx should be ~0, because normalization constrains each row to mean 0."


def residual_backprop(dOut):
    dX = dOut
    dsublayer = dOut
    return dX, dsublayer



  


