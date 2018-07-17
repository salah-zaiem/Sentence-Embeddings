# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:19:43 2018

@author: zaiemjjj
"""

import numpy as np 
import pandas as pd 
import gensim
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk import FreqDist
from process import Process 
from sklearn.utils.extmath import randomized_svd

import sys


class sentenceVectorizer : 
    
    def __init__(self, filename, vectors_name, sentences, parameter_a):
        self.a = parameter_a
        self.filename = filename
        self.vectors_name = vectors_name
        self.sentences = sentences 
        self.word_vectors = KeyedVectors.load_word2vec_format(self.vectors_name, binary=True)
    def get_frequencies (self) : 
        process =Process(self.filename, self.sentences)
        return process.get_frequencies()
    
    def get_sequences(self) : 
        f = open(self.sentences, 'r')
        sentences = f.readlines()
        sentences = [i[0:-1] for i in sentences]
        return sentences
    def sentence_first_vector(self,sentence, a, frequencies) : 
        tokens = word_tokenize(sentence)
        counter = 0 
        vector = [0]*500
        for i in tokens : 
            if i in self.word_vectors.vocab : 
                counter +=1
                weight = a/(a+frequencies.freq(i))
                word_embedding = self.word_vectors.get_vector(i).tolist() 
                for i in range(500): 
                    vector[i] += word_embedding[i] * weight
        vector = [x / counter for x in vector]
        return vector
    
    def vectorize(self) : 
        sentences = self.get_sequences()
        frequencies = self.get_frequencies()
        sentences_matrix= [self.sentence_first_vector(i, self.a, frequencies) for i in sentences]
        sentences_array = np.array(sentences_matrix).transpose()
        U, S, Vt = randomized_svd(sentences_array, n_components=1)
        sentences_vectors = [i - U.dot(U.transpose()).dot(np.array(i)) for i in sentences_matrix]
        return sentences_vectors 
    
    
def main(filename, vectors_name, sentences, parameter_a):
    sv = sentenceVectorizer(filename, vectors_name, sentences, parameter_a)
    return sv.vectorize()
    
if __name__ == "__main__":
   print ('Number of arguments:', len(sys.argv), 'arguments.')
   print ('Argument List:', str(sys.argv[1:]))

   print(main(sys.argv[1], sys.argv[2], sys.argv[3],float(sys.argv[4])))
   