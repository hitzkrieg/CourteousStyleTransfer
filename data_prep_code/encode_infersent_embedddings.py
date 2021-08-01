# -*- coding: utf-8 -*-

""" Use DeepMoji to encode sentences into emotional feature vectors.
"""
from __future__ import print_function, division
import sys
from random import randint
import matplotlib
import pickle
import numpy as np
import torch

GLOVE_PATH = '../../../../Resources/InferSent/InferSent/dataset/GloVe/glove.840B.300d.txt'
MODEL_PATH = '../../../../Resources/InferSent/InferSent/encoder/infersent.allnli.pickle'


def unpickle(filename):
	with open(filename, 'rb') as fp:
		return pickle.load(fp)

def pickelize(obj, filename):
	with open(filename, 'wb') as fp:
		pickle.dump(obj, fp)



SENTENCES = unpickle('pickle_files/sentences.pickle')
# model = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
model = torch.load(MODEL_PATH)
model.set_glove_path(GLOVE_PATH)
model.build_vocab_k_words(K=1000000)

for i in range(2, 39):
	print("**********")
	print("Batch number ", i+1)
	sentences = SENTENCES[i*100000: (i+1)*100000 ]
	print('Encoding texts..')
	encoding = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
	pickelize(encoding, 'pickle_files/infersent_encodings{}.pickle'.format(i+1))
	print("No of encodings and the shape : ", len(encoding), encoding.shape)

