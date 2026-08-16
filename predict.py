import numpy as np

def predict(probabilities, id_to_word):

  output=[]

  for row in range(len(probabilities)):
    id=np.argmax(probabilities[row])

    word=id_to_word[id]

    output.append(word)

  return output
