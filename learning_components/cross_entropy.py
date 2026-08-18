import numpy as np

#probabilities → shape (seq_len, vocab_size)
#np.clip(value, low, high) clamps a number into a range beacuse if value is zero then log(0) will give infinity

def cross_entropy(probabilities, target_id):

  total=0

  for i in range(len(probabilities)):

    p = probabilities[i][target_id[i]]

    p=np.clip(p,1e-12,1.0)

    p=-np.log(p)

    total=total+p


    

  return total/len(target_id)


# def cross_entropy(probabilities, target_id):
#   correct = probabilities[np.arange(len(target_id)), target_id]
#   return np.mean(-np.log(np.clip(correct, 1e-12, 1.0)))