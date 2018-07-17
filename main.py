# %load main.py
"""
Created on Mon May 28 16:19:43 2018

@author: zaiem
"""

import numpy as np 
import pandas as pd 
import gensim
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk import FreqDist
from process import Process 
from sklearn.utils.extmath import randomized_svd
import time

import sys


class sentenceVectorizer : 
    
    def __init__(self, filename, vectors_name, sentences, parameter_a):
        self.a = parameter_a
        self.filename = filename
        self.vectors_name = vectors_name
        self.sentences = sentences 
        t3=time.time()
        print('loading word embeddings')
        self.word_vectors = KeyedVectors.load_word2vec_format(self.vectors_name, binary=True, encoding='latin-1')
        print('word embeddings loaded')
        t4=time.time()
        print(t4-t3)
        self.dimension = self.word_vectors.vector_size
        print(self.dimension)
    def get_frequencies (self) : 
        process =Process(self.filename, self.sentences)
        return process.get_frequencies()
    
    def get_sequences(self) : 
        f = open(self.sentences, 'r', encoding='utf8')
        sentences = f.readlines()
        sentences = [i[0:-1] for i in sentences]
        return sentences
    def sentence_first_vector(self,sentence, a, frequencies) : 
        tokens = word_tokenize(sentence)
        counter = 0 
        vector = [0]*self.dimension
        for i in tokens : 
            if i in self.word_vectors.vocab : 
                counter +=1
                weight = a/(a+frequencies.freq(i))
                word_embedding = self.word_vectors.get_vector(i).tolist() 
                for i in range(self.dimension): 
                    vector[i] += word_embedding[i] * weight

        vector = [x / counter for x in vector]
        return vector
    
    def vectorize(self) : 
        sentences = self.get_sequences()
        print('sentences loaded')
        frequencies = self.get_frequencies()
        print('frequencies done-')
        sentences_matrix= [self.sentence_first_vector(i, self.a, frequencies) for i in sentences]
        sentences_array = np.array(sentences_matrix).transpose()
        
        U, S, Vt = randomized_svd(sentences_array, n_components=1)
        sentences_vectors = [i - U.dot(U.transpose()).dot(np.array(i)) for i in sentences_matrix]
        return sentences_vectors 
    
    
def main(filename, vectors_name, sentences, parameter_a):
    sv = sentenceVectorizer(filename, vectors_name, sentences, parameter_a)
    return sv.vectorize()
    

    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv[1:]))
if __name__ == "__main__":
    t1=time.time()
    results = main(sys.argv[1], sys.argv[2], sys.argv[3],float(sys.argv[4]))
    results_array = np.array(results)
    print('matrix shape')
    print(results_array.shape)
    np.savetxt("french_withqa_vectors.csv", results_array, delimiter=",")
    print('total_time = ' + str(time.time() -t1))
