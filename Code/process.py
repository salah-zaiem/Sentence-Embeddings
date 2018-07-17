# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:34:15 2018

@author: zaiem
"""

import numpy as np 
import pandas as pd 
import gensim
from gensim.models import KeyedVectors
from nltk import word_tokenize
from nltk import FreqDist
class Process :
    
    def __init__(self, filename, sentences) :
        self.filename = filename
        self.sentences= sentences
        

    def get_frequencies(self):
        f = open(self.sentences, 'r')
        document = f.read()
        document = document.replace("\n", " ")
        tokens = word_tokenize(document)
        frequencies= FreqDist(tokens)
        return frequencies
    

        