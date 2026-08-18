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


def attention_backprop(dO, A, V, K, Q, dk):

  dV= A.T @ dO

  dA= dO @ V.T

  dZ= A * (dA - np.sum(A * dA, axis=1, keepdims=True))

  dQ = (dZ @ K) / np.sqrt(dk)

  dK = (dZ.T @ Q )/ np.sqrt(dk)#d small k sub script

  return dV, dA, dZ, dQ, dK # differentiation of capital K

# dO is output from previous layer of backprop
#beautiful derivaion and explanation i copy please refer

# "How do you know your backprop is correct?" → "Gradient checking: compare the analytic gradient to a numerical one via central differences, (L(x+h) − L(x−h))/2h. If they agree to ~1e-7, the analytic gradient is right."
# "Why central difference (f(x+h)−f(x−h))/2h instead of (f(x+h)−f(x))/h?" → "Central difference has error O(h²) vs O(h) — far more accurate for the same h."
# "Softmax backward in general vs with cross-entropy?" → "General: dz = s ⊙ (g − Σ(g·s)) per row. With CE it collapses to p − y. The attention case needs the general form because softmax feeds into @V, not the loss."


  

# this one is too confusing but refer copy for derivations and all
def multihead_backprop(dOut, X, Wq, Wk, Wv, Wo, concat, caches, head_dim, num_heads):

  # caches[h] = (Q_h, K_h, V_h, A_h)  saved from the forward pass, one per head

    # ④ Wo backward  (output = concat @ Wo)

  dWo = concat.T @ dOut
  dconcat = dOut @ Wo.T

  #accumulators
  dX = np.zeros_like(X) #dX will sum over heads(fan-out)

  dWq, dWk, dWv = [], [], [] #one weight-grad per head
  #"I need a gradient array dW that has the same shape as W, so create one filled with 0."

# Then while looping through things, you add gradients into it:

# dW += ...

# np.zeros(5) → 5 zeros
# np.zeros((3,4)) → 3×4 zeros
# np.zeros_like(W) → zeros with exactly the same shape as W

  #loop over heads
  for h in range(num_heads):
    Q_h, K_h, V_h, A_h = caches[h]

    #split this head'd slice of dconcat (columns h*head_dim : (h+1)*head_dim)
    dheadOut = dconcat[:, h*head_dim : (h+1)*head_dim]


    # ① Q_h = X @ Wq[h] etc → weight grads (X.T @ dQ) + dX contribution (dQ @ Wq.T)
    dV_h, dA_h, dZ_h, dQ_h, dK_h = attention_backprop(dheadOut, A_h, V_h, K_h, Q_h, head_dim)

    dWq.append(X.T @ dQ_h)
    dWk.append(X.T @ dK_h)
    dWv.append(X.T @ dV_h)

    dX += dQ_h @ Wq[h].T + dK_h @ Wk[h].T + dV_h @ Wv[h].T

  return dX, dWq, dWk, dWv, dWo



#dy is dout

#Cache = values that we saved during the forward pass because we'll need them during backprop.
#During forward pass, we calculate:
# Q, K, V, A, O
# Then later, during backward, we need those values to calculate gradients.
# So instead of recalculating them, we save them.
# That saved stuff is called the cache.

# cache = [
#     (Q1, K1, V1, A1),
#     (Q2, K2, V2, A2)
# ]
# That's a cache.
# So: cache[0]
# means: Give me everything I saved for head 1.
# and: cache[1]
# means: Give me everything I saved for head 2.




def encoder_backprop(dY, X, Wq, Wk, Wv, Wo, concat, caches, head_dim, num_heads, gamma1, xhat1, s1, gamma2, xhat2, s2, W1, h1, W2, h2, a):

  # final LayerNorm
  dR2, dgamma2, dbeta2 = layernorm_backprop(
    dY, gamma2, xhat2, s2
  )

  # residual: R2 = N1 + F
  dN1_direct, dF = residual_backprop(dR2)

  #FFN
  dh1, dW1, db1, dW2, db2 = ffn_backprop(
    dF, W1, h1, W2, h2, a)

  # two paths reached N1
  dN1 = dN1_direct + dh1

  # first LayerNorm
  dR1, dgamma1, dbeta1 = layernorm_backprop(
    dN1, gamma1, xhat1, s1
)

  # first residual
  dX_direct, dA = residual_backprop(dR1)


  # attention
  dX_attention, dWq, dWk, dWv, dWo = multihead_backprop(
    dA, X, Wq, Wk, Wv, Wo, concat, caches, head_dim, num_heads
)

  # two paths reached X
  dX = dX_direct + dX_attention

  return dX, dWq, dWk, dWv, dWo, dW1, db1, dW2, db2, dgamma1, dbeta1, dgamma2, dbeta2
	

#Multiple paths meeting → gradients add
#Residual addition → gradient splits


# cross-attention has two inputs:
# from decoder and encoder

# So its gradient has to flow both backward into the decoder and backward into the encoder.

# That's why doing the encoder first makes things much easier.

def cross_attention_backprop(dOut, dec_in, enc_out, Wq, Wk, Wv, Wo, concat, caches, head_dim, num_heads):
    dWo = concat.T @ dOut
    dconcat = dOut @ Wo.T

    dDec = np.zeros_like(dec_in)    # gradient to decoder  (via Q)
    dEnc = np.zeros_like(enc_out)   # gradient to encoder  (via K, V)
    dWq, dWk, dWv = [], [], []

    for h in range(num_heads):
        Q_h, K_h, V_h, A_h = caches[h]
        dHead = dconcat[:, h*head_dim:(h+1)*head_dim]
        dV_h, dA_h, dZ_h, dQ_h, dK_h = attention_backprop(dHead, A_h, V_h, K_h, Q_h, head_dim)

        # Q = dec_in @ Wq[h]   → decoder gradient
        dWq.append(dec_in.T @ dQ_h)
        dDec += dQ_h @ Wq[h].T

        # K, V = enc_out @ Wk[h], Wv[h]   → encoder gradient
        dWk.append(enc_out.T @ dK_h)
        dWv.append(enc_out.T @ dV_h)
        dEnc += dK_h @ Wk[h].T + dV_h @ Wv[h].T

    return dDec, dEnc, dWq, dWk, dWv, dWo

#Ye wahi hai jo pichhle multihead backprop func mein diya tha — multihead_backprop jaisa, bas dX ki jagah do accumulators (dDec decoder ke liye, dEnc encoder ke liye):

# multihead_backprop mein Q, K, V teeno X se aate the → ek hi dX.
# Cross-attention mein Q decoder se, K/V encoder se → isliye do alag gradients: dDec (Q path) aur dEnc (K+V path).
# Baaki sab (Wo backward, split, attention_backprop loop, weight grads) bilkul same hai.


def decoder_backprop(dY, X, enc_out,
                     Wq_m,Wk_m,Wv_m,Wo_m, concat_m, caches_m,      # masked self-attn
                     Wq_c,Wk_c,Wv_c,Wo_c, concat_c, caches_c,      # cross-attn
                     head_dim, num_heads,
                     gamma1,xhat1,s1, gamma2,xhat2,s2, gamma3,xhat3,s3,  # 3 LNs
                     W1,h1,W2,h2,a,     # FFN (h1 = N2, the ffn input)
                     N1):               # cross-attn's decoder input

  # ⑨ LN3
  dR3, dgamma3, dbeta3 = layernorm_backprop(dY, gamma3, xhat3, s3)
  # ⑧ residual3: R3 = N2 + F
  dN2_direct, dF = residual_backprop(dR3)
  # ⑦ FFN
  dh1, dW1, db1, dW2, db2 = ffn_backprop(dF, W1, h1, W2, h2, a)
  dN2 = dN2_direct + dh1              # ← fan-out (N2)

  # ⑥ LN2
  dR2, dgamma2, dbeta2 = layernorm_backprop(dN2, gamma2, xhat2, s2)
  # ⑤ residual2: R2 = N1 + cross
  dN1_direct, dCross = residual_backprop(dR2)
  # ④ CROSS-ATTENTION → two gradients
  dN1_cross, dEncOut, dWq_c, dWk_c, dWv_c, dWo_c = cross_attention_backprop(
      dCross, N1, enc_out, Wq_c,Wk_c,Wv_c,Wo_c, concat_c, caches_c, head_dim, num_heads)
  dN1 = dN1_direct + dN1_cross        # ← fan-out (N1)

  # ③ LN1
  dR1, dgamma1, dbeta1 = layernorm_backprop(dN1, gamma1, xhat1, s1)
  # ② residual1: R1 = X + mask
  dX_direct, dMask = residual_backprop(dR1)
  # ① MASKED SELF-ATTENTION → reuse multihead_backprop!
  dX_mask, dWq_m, dWk_m, dWv_m, dWo_m = multihead_backprop(
      dMask, X, Wq_m,Wk_m,Wv_m,Wo_m, concat_m, caches_m, head_dim, num_heads)
  dX = dX_direct + dX_mask            # ← fan-out (X)

  return (dX, dEncOut,
          dWq_m,dWk_m,dWv_m,dWo_m, dWq_c,dWk_c,dWv_c,dWo_c,
          dW1,db1,dW2,db2,
          dgamma1,dbeta1, dgamma2,dbeta2, dgamma3,dbeta3)


# "Cross-attention backward is special how?" → "Q comes from the decoder but K and V come from the encoder output, so the gradient splits: the Q-path flows back to the decoder, the K/V-paths flow back to the encoder. That's the channel through which the decoder's loss reaches and trains the encoder."
# "Masked self-attention backward — need a special function?" → "No. The mask is applied in the forward softmax (future positions get −∞ → weight 0). Since those weights are 0, the softmax backward naturally gives them 0 gradient. So the same multihead_backprop works — the cached attention matrix already carries the mask."