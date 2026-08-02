import string

def load(path):
  with open(path) as file:
   corpus=file.readlines()
  
  return corpus



def tokenize(corpus):

  vocab=[]

  for sentence in corpus:

   sentence=sentence.lower()

   sentence=sentence.translate(str.maketrans('','',string.punctuation))

   words=sentence.split()

   

   for word in words:
     if word not in vocab:
       vocab.append(word)

  return vocab



def build_word_to_id(vocab):

  word_to_id={}

  for index,word in enumerate(vocab):

    word_to_id[word]=index

  return word_to_id


def build_id_to_word(word_to_id):

  id_to_word={}

  for word,id in word_to_id.items():
    id_to_word[id]=word

  return id_to_word


def encode(text, word_to_id):
  encoded=[]
  for word in text.lower().split():
    encoded.append(word_to_id[word])

  return encoded

def decode(ids, id_to_word):
  decoded=[]
  for x in ids:
    decoded.append(id_to_word[x])

  return " ".join(decoded)