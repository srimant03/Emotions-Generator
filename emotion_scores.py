import numpy as np
import math
import random

class BigramLM:
    def __init__(self, corpus):
        self.corpus = corpus
        self.lines = self.read_corpus()
        self.con = self.tokenize_corpus()
        self.dic_count, self.word_count, self.mat_prob_init, self.ind, self.opp = self.create_bigram()
        self.mat_prob = self.compute_bigram_prob()
        self.mat_prob_laplace = self.compute_bigram_prob_smoothed_laplace()
        self.mat_prob_knessar = self.knessar()
        self.ans = self.print_sent('<s>')

    def read_corpus(self):
        with open(self.corpus, 'r') as f:
            lines = [line.rstrip() for line in f]
        return lines

    def tokenize_corpus(self):
        con = [i.strip("''").split(" ") for i in self.lines]
        for i in range(len(con)):
          con[i] = [word.strip('.,?!') for word in con[i]]
          con[i] = [word for word in con[i] if word]
          con[i] = [word.lower() for word in con[i]]
          con[i] += ['<e>']
          con[i].insert(0, '<s>')
        return con

    def create_bigram(self):
        dic_count = {}
        for i in self.con:
            for a,b in zip(i, i[1:]):
                dic_count[(a, b)] = dic_count.get((a, b), 0) + 1

        word_count = {}
        for i in self.con:
            for a in i:
                word_count[(a)] = word_count.get((a), 0) + 1

        keys = list(word_count.keys())
        keys.sort()
        word_count = {i: word_count[i] for i in keys}

        count = len(word_count)
        mat_prob_init = np.zeros((count, count))

        ind = {}
        opp = {}
        for s,i in enumerate(word_count.keys()):
            ind[i] = s
            opp[s] = i

        return dic_count, word_count, mat_prob_init, ind, opp

    def compute_bigram_prob(self):
        #dic_freq = {}
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            self.mat_prob_init[i][j] = self.dic_count[(a,b)]/self.word_count[a]
        return self.mat_prob_init

    def compute_bigram_prob_smoothed_laplace(self, alpha=1):
        '''mat_prob_smoothed = np.zeros((len(self.word_count), len(self.word_count)))
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            mat_prob_smoothed[i][j] = (self.dic_count[(a,b)] + alpha) / (self.word_count[a] + alpha * len(self.word_count))
        return mat_prob_smoothed'''

        mat = self.mat_prob
        mat = mat + alpha
        mat = mat/mat.sum(axis=1, keepdims=True)
        mat_laplace = mat
        return mat_laplace


    def knessar(self, d=0.75):
        '''mat_prob_knessar = np.zeros((len(self.word_count), len(self.word_count)))
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            first_term = max(self.dic_count[(a,b)] - d, 0) / self.word_count[a]
            continuation_probability = len([k for k in self.dic_count if k[0]==b]) / len(self.dic_count)
            alpha = d * len([k for k in self.dic_count if k[0]==a]) / self.word_count[a]
            mat_prob_knessar[i][j] = first_term + alpha * continuation_probability
        return mat_prob_knessar'''

        mat = self.mat_prob
        first_term = np.maximum(mat - d, 0)/mat.sum(axis=1, keepdims=True)
        cont_prob = np.count_nonzero(mat, axis=0, keepdims=True)
        cont_prob = cont_prob/cont_prob.sum()
        alpha_term = (d/(mat.sum(axis=1, keepdims=True)))*(np.count_nonzero(mat, axis=0, keepdims=True))
        mat_knessar = first_term + (alpha_term)*cont_prob
        return mat_knessar


    def predict_next(self, word):
      ind_word = self.ind[word]
      maxi=0
      ans=-1

      ch=np.random.choice(len(self.mat_prob_laplace[ind_word]), p=self.mat_prob_laplace[ind_word])

      return self.opp[ch]

    def print_sent(self, int_word):
      ans=""
      if(int_word!='<s>'):
          ans+=int_word
      word = self.predict_next(int_word)
      stop=0
      while(stop<20 and word!='<e>'):
          if(ans!=""):
              ans+=" "
          ans+=word
          word = self.predict_next(word)
          stop+=1
      return ans+'.'

x = BigramLM('/content/drive/MyDrive/NLPA1/corpus.txt')

ans = x.print_sent('<s>')
print(ans)

import sys
sys.path.append('/content/drive/MyDrive/NLPA1')
from utils import emotion_scores

class BigramLME:
    def __init__(self, corpus):
        self.corpus = corpus
        self.lines = self.read_corpus()
        self.con = self.tokenize_corpus()
        self.dic_count, self.word_count, self.mat_prob_init, self.ind, self.opp = self.create_bigram()
        self.mat_prob = self.compute_bigram_prob()
        self.mat_prob_laplace = self.compute_bigram_prob_smoothed_laplace()
        self.mat_prob_knessar = self.knessar()
        self.mat_emo = self.calc_emotions()
        self.ans = self.print_sent('<s>')

    def read_corpus(self):
        with open(self.corpus, 'r') as f:
            lines = [line.rstrip() for line in f]
        return lines

    def tokenize_corpus(self):
        con = [i.strip("''").split(" ") for i in self.lines]
        for i in range(len(con)):
          con[i] = [word.strip('.,?!') for word in con[i]]
          con[i] = [word for word in con[i] if word]
          con[i] = [word.lower() for word in con[i]]
          con[i] += ['<e>']
          con[i].insert(0, '<s>')
        return con

    def create_bigram(self):
        dic_count = {}
        for i in self.con:
            for a,b in zip(i, i[1:]):
                dic_count[(a, b)] = dic_count.get((a, b), 0) + 1

        word_count = {}
        for i in self.con:
            for a in i:
                word_count[(a)] = word_count.get((a), 0) + 1

        keys = list(word_count.keys())
        keys.sort()
        word_count = {i: word_count[i] for i in keys}

        count = len(word_count)
        mat_prob_init = np.zeros((count, count))

        ind = {}
        opp = {}
        for s,i in enumerate(word_count.keys()):
            ind[i] = s
            opp[s] = i

        return dic_count, word_count, mat_prob_init, ind, opp

    def compute_bigram_prob(self):
        #dic_freq = {}
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            self.mat_prob_init[i][j] = self.dic_count[(a,b)]/self.word_count[a]
        return self.mat_prob_init

    def compute_bigram_prob_smoothed_laplace(self, alpha=1):
        '''mat_prob_smoothed = np.zeros((len(self.word_count), len(self.word_count)))
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            mat_prob_smoothed[i][j] = (self.dic_count[(a,b)] + alpha) / (self.word_count[a] + alpha * len(self.word_count))
        return mat_prob_smoothed'''

        mat = self.mat_prob
        mat = mat + alpha
        mat = mat/mat.sum(axis=1, keepdims=True)
        mat_laplace = mat
        return mat_laplace


    def knessar(self, d=0.75):
        '''mat_prob_knessar = np.zeros((len(self.word_count), len(self.word_count)))
        for a,b in self.dic_count:
            i = self.ind[a]
            j = self.ind[b]
            first_term = max(self.dic_count[(a,b)] - d, 0) / self.word_count[a]
            continuation_probability = len([k for k in self.dic_count if k[0]==b]) / len(self.dic_count)
            alpha = d * len([k for k in self.dic_count if k[0]==a]) / self.word_count[a]
            mat_prob_knessar[i][j] = first_term + alpha * continuation_probability
        return mat_prob_knessar'''

        mat = self.mat_prob
        first_term = np.maximum(mat - d, 0)/mat.sum(axis=1, keepdims=True)
        cont_prob = np.count_nonzero(mat, axis=0, keepdims=True)
        cont_prob = cont_prob/cont_prob.sum()
        alpha_term = (d/(mat.sum(axis=1, keepdims=True)))*(np.count_nonzero(mat, axis=0, keepdims=True))
        mat_knessar = first_term + (alpha_term)*cont_prob
        return mat_knessar

    # def calc_emotions(self):
    #     emo_mat = {}

    #     emotion_list = [i['label'] for i in emotion_scores('i')]

    #     count = len(self.word_count)

    #     for i in emotion_list:
    #       emo_mat[i] = np.zeros((count, count))

    #     for a,b in self.dic_count:
    #       i = self.ind[a]
    #       j = self.ind[b]
    #       scores = emotion_scores(a+' '+b)

    #       for idx in range(len(emotion_list)):
    #         emo_mat[emotion_list[idx]][i][j] = scores[idx]['score']

    #     return emo_mat


    def predict_next(self, word):
      ind_word = self.ind[word]
      maxi=0
      ans=-1

      ch=np.random.choice(len(self.mat_prob_laplace[ind_word]), p=self.mat_prob_laplace[ind_word])

      return self.opp[ch]

    def print_sent(self, int_word):
      ans=""
      if(int_word!='<s>'):
          ans+=int_word
      word = self.predict_next(int_word)
      stop=0
      while(stop<20 and word!='<e>'):
          if(ans!=""):
              ans+=" "
          ans+=word
          word = self.predict_next(word)
          stop+=1
      return ans+'.'

y = BigramLME('/content/drive/MyDrive/NLPA1/corpus.txt')

import pickle
file_name = '/content/drive/MyDrive/NLPA1/y.pkl'

# Writing the object to a file using pickle
with open(file_name, 'wb') as file:
    pickle.dump(y, file)
    print(f'Object successfully saved to "{file_name}"')

# Reading the object back from the file
with open(file_name, "rb") as file:
    y = pickle.load(file)

print(f"Deserialized Student Object: {y.mat_prob_laplace}")

word_emo = {}
emotion_list = [i['label'] for i in emotion_scores('i')]

for i in emotion_list:
  word_emo[i] = np.zeros((len(y.word_count)))

for a in y.word_count:
  i = y.ind[a]
  scores = emotion_scores(a)

  for idx in range(len(emotion_list)):
    word_emo[emotion_list[idx]][i] = scores[idx]['score']

for i in emotion_list:
  word_emo[i] = word_emo[i]/word_emo[i].sum()

test_mat_emo = y.mat_emo

def mod_emo_mat(alpha,emotion):
  count = len(y.word_count)
  temp = np.zeros((count, count))

  for a,b in y.dic_count:
    i = y.ind[a]
    j = y.ind[b]
    temp[i][j] = y.mat_prob[i][j]
    if(y.mat_emo[emotion][i][j]>0):
      temp[i][j] = (temp[i][j] + alpha * y.mat_emo[emotion][i][j]/y.mat_emo[emotion][i].sum())/(alpha+1)

  return temp

emotion_list = [i['label'] for i in emotion_scores('i')]
for i in emotion_list:
  test_mat_emo[i] = mod_emo_mat(50,i)

def pred_next(word, emotion):
  ind_word = y.ind[word]
  maxi=0
  ans=-1

  ch=np.random.choice(len(test_mat_emo[emotion][ind_word]), p=test_mat_emo[emotion][ind_word])

  return y.opp[ch]

def genSen(int_word, emotion):
  ans=""
  if(int_word!='<s>'):
      ans+=int_word
  word = pred_next(int_word, emotion)
  stop=0
  while(stop<20 and word!='<e>'):
      if(ans!=""):
          ans+=" "
      ans+=word
      word = pred_next(word, emotion)
      stop+=1
  return ans+'.'

emotion_list = [i['label'] for i in emotion_scores('i')]

def find_emotion(sent):
    scores = [i['score'] for i in emotion_scores(sent)]
    mx = max(scores)
    return emotion_list[scores.index(mx)]

print("Results: ")

for i in emotion_list:
  results = []

  for j in range(100):
    sent = genSen('<s>',i)
    results.append(find_emotion(sent))

  print(i,results.count(i)/len(results) * 100, " percent")

def pred_next_uni(word, emotion):
  ind_word = y.ind[word]
  maxi=0
  ans=-1

  ch=np.random.choice(len(word_emo[emotion]), p=word_emo[emotion])

  return y.opp[ch]

def genSen_uni(int_word, emotion):
  ans=""
  if(int_word!='<s>'):
      ans+=int_word
  word = pred_next_uni(int_word, emotion)
  stop=0
  while(stop<20 and word!='<e>'):
      if(ans!=""):
          ans+=" "
      ans+=word
      word = pred_next_uni(word, emotion)
      stop+=1
  return ans+'.'

print("Results: ")

for i in emotion_list:
  results = []

  for j in range(100):
    sent = genSen_uni('<s>',i)
    results.append(find_emotion(sent))

  print(i,results.count(i)/len(results) * 100, " percent")

print("Example sentences: ")
print()

for i in emotion_list:
  print(i)

  for j in range(5):
    sent = genSen('<s>',i)
    print(sent,"| emotion: ",find_emotion(sent))
    print()

for i in emotion_list:
  filelocation = '/content/drive/MyDrive/NLPA1/unigram_samples/'
  filename = filelocation + 'gen_'+i+'.txt'

  with open(filename, 'w') as f:
    cnt = 0

    while(cnt<50):
      sent = genSen_uni('<s>',i)

      if(len(sent.split())>8):
        f.write(sent)
        f.write('\n')
        cnt+=1

  print('Samples for',i,' generated.')