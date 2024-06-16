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
        mat_knessar = mat_knessar/mat_knessar.sum(axis=1, keepdims=True)
        return mat_knessar

    def top_5_bigrams_by_prob(self, mat_prob):
        flat_mat_prob = mat_prob.flatten()
        flat_mat_prob = flat_mat_prob[~np.isnan(flat_mat_prob)]
        top_5_indices = np.argsort(flat_mat_prob)[-5:]
        top_5_bigrams = [(self.opp[i // len(mat_prob)], self.opp[i % len(mat_prob)]) for i in top_5_indices]
        top_5_probabilities = [flat_mat_prob[i] for i in top_5_indices]
        return list(zip(top_5_bigrams, top_5_probabilities))

    def print_top_5_bigrams_by_prob(self):
        print("Top 5 Bigrams with Highest Probabilities Before Smoothing:")
        print(self.top_5_bigrams_by_prob(self.mat_prob_init))
        print("\nTop 5 Bigrams with Highest Probabilities After Laplace Smoothing:")
        print(self.top_5_bigrams_by_prob(self.mat_prob_laplace))
        print("\nTop 5 Bigrams with Highest Probabilities After Kneser-Ney Smoothing:")
        print(self.top_5_bigrams_by_prob(self.mat_prob_knessar))

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

x = BigramLM('/content/drive/MyDrive/NLP/data-20240130T044130Z-001/data/corpus.txt')

ans = x.print_sent('<s>')
print(ans)
x.print_top_5_bigrams_by_prob()
